# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
from loguru import logger
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from model_usage import ModelUsageRecorder
except ImportError:
    from .model_usage import ModelUsageRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    str(LOGS_DIR / "stage2_{time:YYYYMMDD_HHmmss}.log"),
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)


SYSTEM_PROMPT = """你是一个极其严谨的水排序游戏剪辑大师。我将按时间顺序提供一段视频的 12 张关键帧截图（编号 1 到 12）。

【视觉判定常识（铁律）】：
1. **静止状态**：所有瓶子平稳放于桌面，无白边。
2. **动作状态**：瓶子有白边、悬空或正在倒水。

【任务 1：识别关卡】识别画面上方当前是第几关。
【任务 2：提取所有闭环动作】仔细扫描这 12 帧，找出**所有**独立且完整的核心操作。
动作定性标准：
   - `success_step`: 成功的倒水。
   - `trial_error`: 完整的试错（选中/提起 -> 无法倒水 -> 放下）。
   - `victory`: 出现通关结算。

【任务 3：精准裁剪】为找到的**每一个**动作提供起止帧。
铁律：截取的首尾帧，必须是绝对的“静止状态”！
   - `start_frame`: 动作开始前的最后一刻（静止状态）。
   - `end_frame`: 动作彻底结束的一刻（静止状态，且无新瓶子被选中）。
    - 等待时间规则：如果画面连续超过 2 秒没有任何操作（无人点击、选中、提起、移动、倒水或放下），这段等待不需要保留。不要把长时间空等包含在动作片段的开头、结尾或中间；如果某段只是在等待且没有完整操作，请不要输出 action。对于 6 秒、12 帧均匀采样的视频，超过约 4 个采样间隔的纯静止等待应被裁掉或跳过。

强制输出合法的 JSON 格式。如果发现多个动作，请放入 actions 数组；如果没有发现任何完整动作，actions 留空。格式如下：
{
  "level": "Level_X",
  "actions": [
    {
      "action_type": "success_step",
      "start_frame": 1,
      "end_frame": 4,
      "reasoning": "简述..."
    },
    {
      "action_type": "trial_error",
      "start_frame": 9,
      "end_frame": 12,
      "reasoning": "简述..."
    }
  ]
}"""


class VideoFineFilter:
    """Stage 2: 12-frame VLM closed-loop trimming and global asset pooling."""

    SAMPLE_FRAME_COUNT = 12
    VALID_ACTION_TYPES = {"success_step", "trial_error", "victory"}

    def __init__(
        self,
        interim_dir: str = "data/interim",
        processed_assets_dir: str = "data/processed_assets",
        global_assets_path: str = "data/global_assets.json",
        run_id: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_side: int = 720,
        jpeg_quality: int = 80,
    ) -> None:
        self.interim_dir = Path(interim_dir)
        self.clips_root = self.interim_dir / "clips"
        self.processed_assets_dir = Path(processed_assets_dir)
        self.global_assets_path = Path(global_assets_path)
        self.run_id = self._safe_name(run_id or self._timestamp())
        self.run_output_dir = self.processed_assets_dir / self.run_id
        self.model = (
            model
            or os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview").strip()
            or "openai/gpt-4o-mini"
        )
        self.max_side = max(256, int(max_side))
        self.jpeg_quality = min(95, max(40, int(jpeg_quality)))

        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY", "sk-or-v1-5794a8b038307965ef5bcdfea40fcfc18").strip()
        if not resolved_api_key:
            raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.")

        resolved_base_url = (
            base_url
            or os.getenv("OPENROUTER_BASE_URL", "").strip()
            or "https://apirouter.zhiqiteai.cn/ApiRouterServ/v1"
        )
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        self.global_assets_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.run_output_dir / "stage2.log"
        self.usage_recorder = ModelUsageRecorder(self.run_output_dir / "model_usage.json", stage="stage2")
        logger.add(
            str(self.run_log_path),
            rotation="10 MB",
            level="INFO",
            encoding="utf-8",
        )

    def run(self, only_video: str | None = None) -> list[dict[str, Any]]:
        if not self.clips_root.exists():
            logger.warning("Clips root does not exist: {}", self.clips_root)
            return []

        segments_files = sorted(self.clips_root.glob("*/segments.json"))
        if only_video:
            wanted = only_video.strip()
            segments_files = [path for path in segments_files if path.parent.name == wanted]

        if not segments_files:
            logger.warning("No segments.json files found under {}", self.clips_root)
            return []

        created_assets: list[dict[str, Any]] = []
        logger.info(
            "Stage 2 started | segment groups {} | model {} | run {} | output {} | global index {}",
            len(segments_files),
            self.model,
            self.run_id,
            self.run_output_dir,
            self.global_assets_path,
        )
        logger.info("Run log path: {}", self.run_log_path)

        for segments_path in segments_files:
            video_name = segments_path.parent.name
            logger.info("Processing segment group: {}", video_name)
            created_assets.extend(self._process_segments_file(video_name, segments_path))

        self.usage_recorder.write()
        logger.info("Stage 2 finished | created assets {}", len(created_assets))
        return created_assets

    def _process_segments_file(
        self,
        video_name: str,
        segments_path: Path,
    ) -> list[dict[str, Any]]:
        records = self._load_segments(segments_path)
        if not records:
            logger.warning("Empty segments.json: {}", segments_path)
            return []

        assets: list[dict[str, Any]] = []
        total = len(records)
        for index, record in enumerate(records, start=1):
            clip_path = self._resolve_clip_path(record, base_dir=segments_path.parent)
            clip_tag = f"[{video_name} #{index:03d}/{total:03d}]"

            try:
                frames_b64, source_duration = self._sample_frames_as_base64(clip_path)
                analysis = self._analyze_clip_with_retry(
                    frames_b64,
                    request_name=clip_tag,
                    metadata={
                        "video_name": video_name,
                        "segment_index": index,
                        "segment_total": total,
                        "clip_path": self._json_path(clip_path),
                        "source_duration": round(source_duration, 3),
                    },
                )
                level = analysis["level"]
                actions = analysis["actions"]
                logger.info(
                    "{} VLM result | level={} | actions={}",
                    clip_tag,
                    level,
                    len(actions),
                )

                if not actions:
                    logger.info("{} skipped clip with no complete actions: {}", clip_tag, clip_path.name)
                    continue

                for action_index, action in enumerate(actions, start=1):
                    crop_start, crop_end = self._frame_range_to_crop_times(
                        start_frame=action["start_frame"],
                        end_frame=action["end_frame"],
                        duration=source_duration,
                    )
                    output_path = self._build_asset_output_path(
                        level=level,
                        action_type=action["action_type"],
                        part_index=action_index,
                    )
                    self._trim_clip(
                        input_path=clip_path,
                        start_time=crop_start,
                        duration=crop_end - crop_start,
                        output_path=output_path,
                    )
                    final_duration = self._get_video_duration(output_path) or (crop_end - crop_start)
                    asset = self._build_asset_record(
                        level=level,
                        action=action,
                        source_record=record,
                        clip_path=clip_path,
                        output_path=output_path,
                        duration=final_duration,
                    )
                    self._append_global_asset(asset)
                    assets.append(asset)
                    logger.info(
                        "{} created action asset [{}/{}]: {} | {} frames {}-{} | {}",
                        clip_tag,
                        action_index,
                        len(actions),
                        output_path,
                        action["action_type"],
                        action["start_frame"],
                        action["end_frame"],
                        action["reasoning"],
                    )
            except Exception as exc:
                logger.exception("{} failed: {} | {}", clip_tag, clip_path, exc)

        return assets

    def _sample_frames_as_base64(self, clip_path: Path) -> tuple[list[str], float]:
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip does not exist: {clip_path}")

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0 or frame_count <= 0:
                raise RuntimeError(f"Cannot read FPS or frame count: {clip_path}")

            duration = frame_count / fps
            frame_interval = duration / self.SAMPLE_FRAME_COUNT
            max_time = max(0.0, duration - (1.0 / fps))
            sampled_images: list[str] = []

            for frame_number in range(1, self.SAMPLE_FRAME_COUNT + 1):
                sample_time = min(max_time, max(0.0, (frame_number - 0.5) * frame_interval))
                frame_index = min(frame_count - 1, max(0, int(round(sample_time * fps))))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to sample frame {frame_number}: {clip_path}")

                sampled_images.append(self._to_base64_jpeg(self._compress_frame(frame)))

            return sampled_images, duration
        finally:
            cap.release()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _analyze_clip_with_retry(
        self,
        frames_b64: list[str],
        request_name: str = "stage2_vlm_analysis",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(frames_b64) != self.SAMPLE_FRAME_COUNT:
            raise ValueError(f"Expected 12 frames, got {len(frames_b64)}")

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "下面是同一段 6 秒安全区视频按时间顺序均匀抽取的 12 张关键帧。"
                    "每张图片前的文字标注就是帧编号，请严格按照编号 1 到 12 判断闭环动作边界。"
                ),
            }
        ]
        for index, image_b64 in enumerate(frames_b64, start=1):
            content.append({"type": "text", "text": f"Frame {index}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=768,
        )
        self.usage_recorder.record_chat_completion(
            response,
            step="stage2_vlm_analysis",
            request_name=request_name,
            model=self.model,
            metadata=metadata,
        )

        raw_text = (response.choices[0].message.content or "").strip()
        if not raw_text:
            raise RuntimeError("VLM returned an empty response.")

        return self._parse_and_validate_response(raw_text)

    def _parse_and_validate_response(self, raw_text: str) -> dict[str, Any]:
        parsed = self._parse_json_object(raw_text)

        level = self._normalize_level(parsed.get("level"))
        raw_actions = parsed.get("actions", [])
        if not isinstance(raw_actions, list):
            raw_actions = []

        actions: list[dict[str, Any]] = []
        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                continue

            action_type = str(raw_action.get("action_type") or "").strip()
            if action_type not in self.VALID_ACTION_TYPES:
                continue

            start_frame = self._parse_frame_number(raw_action.get("start_frame"), default=1)
            end_frame = self._parse_frame_number(
                raw_action.get("end_frame"),
                default=self.SAMPLE_FRAME_COUNT,
            )
            if action_type == "victory":
                end_frame = self.SAMPLE_FRAME_COUNT
            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame

            reasoning = str(raw_action.get("reasoning") or "").strip()
            if not reasoning:
                reasoning = "模型未提供明确推理。"

            actions.append(
                {
                    "action_type": action_type,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "reasoning": reasoning,
                }
            )

        return {
            "level": level,
            "actions": actions,
        }

    def _frame_range_to_crop_times(
        self,
        *,
        start_frame: int,
        end_frame: int,
        duration: float,
    ) -> tuple[float, float]:
        frame_interval = duration / float(self.SAMPLE_FRAME_COUNT)
        crop_start = max(0.0, (float(start_frame) - 1.5) * frame_interval)
        crop_end = min(duration, (float(end_frame) + 0.5) * frame_interval)
        if crop_end <= crop_start:
            crop_start = 0.0
            crop_end = duration
        return crop_start, crop_end

    def _build_asset_output_path(self, *, level: str, action_type: str, part_index: int) -> Path:
        output_dir = self.run_output_dir / level
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._timestamp()
        suffix = uuid.uuid4().hex[:8]
        return output_dir / f"{action_type}_{timestamp}_part{part_index:02d}_{suffix}.mp4"

    def _trim_clip(
        self,
        *,
        input_path: Path,
        start_time: float,
        duration: float,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            self._format_seconds(start_time),
            "-i",
            str(input_path),
            "-t",
            self._format_seconds(duration),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]
        self._run_ffmpeg(command, f"Failed to trim clip: {output_path}")

    @staticmethod
    def _get_video_duration(video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot read video duration: {}", video_path)
            return 0.0

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0 or frame_count <= 0:
                return 0.0
            return frame_count / fps
        finally:
            cap.release()

    def _build_asset_record(
        self,
        *,
        level: str,
        action: dict[str, Any],
        source_record: dict[str, Any],
        clip_path: Path,
        output_path: Path,
        duration: float,
    ) -> dict[str, Any]:
        original_video = (
            str(source_record.get("source_file") or "").strip()
            or str(source_record.get("source_video") or "").strip()
            or clip_path.name
        )
        return {
            "id": uuid.uuid4().hex,
            "level": level,
            "action_type": action["action_type"],
            "original_video": original_video,
            "final_clip_path": self._json_path(output_path),
            "duration": round(max(0.0, float(duration)), 3),
            "created_at": self._iso_timestamp(),
        }

    def _append_global_asset(self, asset: dict[str, Any]) -> None:
        assets = self._load_global_assets()
        assets.append(asset)
        self._write_json_list(self.global_assets_path, assets)

    def _load_global_assets(self) -> list[dict[str, Any]]:
        if not self.global_assets_path.exists():
            return []

        data = json.loads(self.global_assets_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict) and isinstance(data.get("assets"), list):
            return [item for item in data["assets"] if isinstance(item, dict)]
        raise ValueError(f"global_assets.json must be a JSON list: {self.global_assets_path}")

    def _compress_frame(self, frame: Any) -> Any:
        height, width = frame.shape[:2]
        max_edge = max(height, width)
        if max_edge > self.max_side:
            scale = self.max_side / float(max_edge)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def _to_base64_jpeg(self, frame: Any) -> str:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed.")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    @staticmethod
    def _parse_json_object(raw_text: str) -> dict[str, Any]:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, flags=re.S)
            if not match:
                raise
            data = json.loads(match.group(0))

        if not isinstance(data, dict):
            raise ValueError("VLM output JSON must be an object.")
        return data

    @classmethod
    def _parse_frame_number(cls, raw_value: Any, *, default: int) -> int:
        try:
            frame_number = int(round(float(raw_value)))
        except (TypeError, ValueError):
            frame_number = default
        return max(1, min(cls.SAMPLE_FRAME_COUNT, frame_number))

    @staticmethod
    def _normalize_level(raw_level: Any) -> str:
        text = str(raw_level or "").strip()
        match = re.search(r"(\d+)", text)
        if match:
            return f"Level_{int(match.group(1))}"
        if re.fullmatch(r"Level_[A-Za-z0-9_-]+", text):
            return text
        return "Level_Unknown"

    @staticmethod
    def _load_segments(path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"segments.json must be a list: {path}")
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _resolve_clip_path(record: dict[str, Any], base_dir: Path) -> Path:
        raw_path = str(record.get("clip_path", "")).strip()
        if not raw_path:
            raise ValueError("Segment record is missing clip_path.")

        path = Path(raw_path)
        if path.is_absolute():
            return path
        if path.exists():
            return path

        project_candidate = PROJECT_ROOT / path
        if project_candidate.exists():
            return project_candidate

        base_candidate = base_dir / path
        if base_candidate.exists():
            return base_candidate

        return base_dir / path.name

    @staticmethod
    def _run_ffmpeg(command: list[str], error_prefix: str) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg was not found. Please install it and add it to PATH.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip() or "ffmpeg did not return detailed stderr."
            raise RuntimeError(f"{error_prefix}\n{detail}") from exc

    @staticmethod
    def _write_json_list(path: Path, data: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    @staticmethod
    def _json_path(path: Path) -> str:
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        return f"{max(0.0, float(seconds)):.3f}"

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._-")
        return safe or time.strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _iso_timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")


VideoFineScorer = VideoFineFilter


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: 12-frame VLM closed-loop trimming and global asset pooling"
    )
    parser.add_argument(
        "--interim-dir",
        default="data/interim",
        help="Directory containing stage-1 clips and segments.json files.",
    )
    parser.add_argument(
        "--processed-assets-dir",
        default="data/processed_assets",
        help="Output directory for classified trimmed assets.",
    )
    parser.add_argument(
        "--global-assets-path",
        default="data/global_assets.json",
        help="Global asset pool JSON path.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run folder under processed assets. Defaults to current timestamp.",
    )
    parser.add_argument(
        "--video-name",
        default=None,
        help="Only process a specific clips subdirectory name.",
    )
    parser.add_argument("--model", default=None, help="OpenRouter model name.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenRouter API base URL. Defaults to OPENROUTER_BASE_URL.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=720,
        help="Maximum image side before sending frames to the VLM.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for sampled frames, clamped to 40-95.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    load_env_file(PROJECT_ROOT / ".env")

    started_at = time.perf_counter()
    logger.info("=== Stage 2 started: 12-frame VLM trimming and global asset pooling ===")
    logger.info("Project root: {}", PROJECT_ROOT)

    fine_filter = VideoFineFilter(
        interim_dir=args.interim_dir,
        processed_assets_dir=args.processed_assets_dir,
        global_assets_path=args.global_assets_path,
        run_id=args.run_id,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
    )
    assets = fine_filter.run(only_video=args.video_name)
    elapsed = time.perf_counter() - started_at

    logger.info(
        "=== Stage 2 finished: created {} assets, elapsed {:.2f}s ===",
        len(assets),
        elapsed,
    )


if __name__ == "__main__":
    main()
