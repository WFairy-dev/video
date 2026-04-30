# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
from loguru import logger
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    str(LOGS_DIR / "stage2_{time:YYYYMMDD_HHmmss}.log"),
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)


SYSTEM_PROMPT = """你是一个专业的水排序游戏审核员和剪辑导演。我将提供一段游戏动作视频（长短不一）的 5 张按时间顺序的关键帧截图。
请仔细观察瓶子的倾斜动作和水位变化。

判定标准打分（1-10分）：
- 1分（试错/废片）：无有效动作，首尾水位无变化。
- 3分（普通倒水）：发生倾斜和倒水，但没装满任何一瓶。
- 5分（高光满瓶）：成功凑齐单色，某一个瓶子被彻底装满。
- 10分（通关胜利）：出现结算界面或满屏特效。

【剪辑导演任务 - 决定播放速度】
你还需要为保留下来的片段决定播放速度（speed）。
- 取值范围：严格限制在 0.9 到 2.0 之间（1.0 为原速）。
- 建议原则：如果是普通动作想快点过，可以给 1.1~1.3；如果是极其解压的满瓶高光，可以给 0.9 进行慢放强调；胜利画面建议 1.0。

强制输出合法 JSON，格式如下：
{"reasoning": "推理过程...", "score": 分数, "selected": true/false, "speed": 浮点数}
注意：score 为 1 时 selected 必须为 false，speed 可默认设 1.0。不要输出 JSON 以外的任何文本。
"""


class VideoFineFilter:
    """阶段二精筛：按真实片段时长抽 5 帧 + VLM 语义过滤。"""

    SAMPLE_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
    VALID_LABELS = {"试错", "有效倒水", "通关胜利"}

    def __init__(
        self,
        interim_dir: str = "data/interim",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_side: int = 720,
        jpeg_quality: int = 80,
    ) -> None:
        self.interim_dir = Path(interim_dir)
        self.clips_root = self.interim_dir / "clips"
        self.model = model or os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview").strip() or "openai/gpt-4o-mini"
        self.max_side = max(256, int(max_side))
        self.jpeg_quality = min(95, max(40, int(jpeg_quality)))

        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        if not resolved_api_key:
            raise ValueError("缺少 OpenRouter API Key，请设置 OPENROUTER_API_KEY 或 --api-key。")

        resolved_base_url = (
            base_url
            or os.getenv("OPENROUTER_BASE_URL", "").strip()
            or "https://apirouter.zhiqiteai.cn/ApiRouterServ/v1"
        )
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )

    def run(self, only_video: str | None = None) -> list[dict[str, Any]]:
        """遍历 clips 子目录，读取 segments.json 并输出 scored_segments.json。"""
        if not self.clips_root.exists():
            logger.warning("未找到 clips 目录: {}", self.clips_root)
            return []

        segments_files = sorted(self.clips_root.glob("*/segments.json"))
        if only_video:
            segments_files = [
                path for path in segments_files if path.parent.name == only_video.strip()
            ]

        if not segments_files:
            logger.warning("未找到可处理的 segments.json，目录: {}", self.clips_root)
            return []

        all_scored: list[dict[str, Any]] = []
        logger.info("阶段二启动 | 待处理视频数 {} | 模型 {}", len(segments_files), self.model)

        for segments_path in segments_files:
            video_name = segments_path.parent.name
            logger.info("开始处理视频子目录: {}", video_name)
            scored = self._process_video_segments(video_name, segments_path)
            all_scored.extend(scored)

        logger.info("阶段二结束 | 总判定片段 {}", len(all_scored))
        return all_scored

    def _process_video_segments(
        self,
        video_name: str,
        segments_path: Path,
    ) -> list[dict[str, Any]]:
        records = self._load_segments(segments_path)
        if not records:
            logger.warning("segments.json 为空: {}", segments_path)
            output_path = segments_path.parent / "scored_segments.json"
            self._write_json_list(output_path, [])
            return []

        scored_records: list[dict[str, Any]] = []
        total = len(records)
        for idx, record in enumerate(records, start=1):
            clip_path = self._resolve_clip_path(record, base_dir=segments_path.parent)
            clip_name = clip_path.name
            clip_tag = f"[Video {video_name} - Clip {idx:03d}]"
            duration = self._record_duration(record, clip_path)
            try:
                frames_b64 = self._sample_five_frames_as_base64(clip_path, duration)
                classification = self._classify_clip_with_retry(frames_b64)
                merged = dict(record)
                merged.update(classification)
                merged["duration"] = round(duration, 3)
                scored_records.append(merged)

                logger.info(
                    "{} 判定完成 ({}/{}): {} | duration={:.3f}s | score={} | selected={} | speed={:.2f} | 理由: {}",
                    clip_tag,
                    idx,
                    total,
                    classification["label"],
                    duration,
                    classification["score"],
                    classification["selected"],
                    classification["speed"],
                    classification["reason"],
                )
            except Exception as exc:
                logger.exception("{} 判定失败: {} | 错误: {}", clip_tag, clip_name, exc)
                fallback = dict(record)
                fallback.update(
                    {
                        "label": "试错",
                        "reasoning": f"判定失败，自动降级为试错：{type(exc).__name__}",
                        "reason": f"判定失败，自动降级为试错：{type(exc).__name__}",
                        "score": 1,
                        "selected": False,
                        "speed": 1.0,
                        "duration": round(duration, 3),
                    }
                )
                scored_records.append(fallback)

        output_path = segments_path.parent / "scored_segments.json"
        self._write_json_list(output_path, scored_records)
        logger.info("已写入精筛结果: {}", output_path)
        return scored_records

    def _sample_five_frames_as_base64(self, clip_path: Path, duration: float | None = None) -> list[str]:
        """按照真实时长的 10%/30%/50%/70%/90% 采样 5 张关键帧。"""
        if not clip_path.exists():
            raise FileNotFoundError(f"片段文件不存在: {clip_path}")

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开片段: {clip_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0 or frame_count <= 0:
                raise RuntimeError(f"无法读取视频帧率或帧总数: {clip_path}")

            physical_duration = float(duration or 0.0)
            if physical_duration <= 0:
                physical_duration = frame_count / fps

            sampled_images: list[str] = []
            max_time = max(0.0, min(physical_duration, frame_count / fps) - (1.0 / fps))
            for ratio in self.SAMPLE_RATIOS:
                sample_time = max(0.0, min(max_time, physical_duration * ratio))
                frame_index = min(frame_count - 1, max(0, int(round(sample_time * fps))))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"截帧失败: {clip_path.name} @ ratio={ratio:.1f}, time={sample_time:.3f}s")

                compressed = self._compress_frame(frame)
                sampled_images.append(self._to_base64_jpeg(compressed))

            return sampled_images
        finally:
            cap.release()

    def _compress_frame(self, frame: Any) -> Any:
        """压缩截图尺寸，降低 token 和网络传输开销。"""
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
            raise RuntimeError("JPEG 编码失败。")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _classify_clip_with_retry(self, frames_b64: list[str]) -> dict[str, Any]:
        """调用 OpenRouter VLM 并解析为结构化分类结果。"""
        if len(frames_b64) != 5:
            raise ValueError(f"期望 5 帧，实际 {len(frames_b64)} 帧。")

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "以下是同一段游戏动作视频（长短不一）在 10%、30%、50%、70%、90% "
                    "位置按时间顺序抽取的 5 张关键帧截图。请作为审核员和剪辑导演，"
                    "严格按系统要求输出包含 reasoning、score、selected、speed 的 JSON。"
                ),
            }
        ]
        for image_b64 in frames_b64:
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
            max_tokens=320,
        )

        raw_text = (response.choices[0].message.content or "").strip()
        if not raw_text:
            raise RuntimeError("VLM 返回为空。")

        return self._parse_and_validate_response(raw_text)

    def _parse_and_validate_response(self, raw_text: str) -> dict[str, Any]:
        """兼容解析模型输出，并补齐内部字段。"""
        parsed = self._parse_json_strict(raw_text)

        score = self._parse_score(parsed.get("score", 1))
        label = self._label_from_score(score)

        reasoning = str(parsed.get("reasoning") or parsed.get("reason") or "").strip()
        if not reasoning:
            reasoning = "模型未提供明确原因。"

        selected = bool(parsed.get("selected", score != 1))
        if score == 1:
            selected = False
        elif not selected:
            selected = True

        speed = self._clamp_speed(parsed.get("speed", 1.0))
        if not selected:
            speed = 1.0

        return {
            "label": label,
            "reasoning": reasoning,
            "reason": reasoning,
            "score": score,
            "selected": selected,
            "speed": speed,
        }

    @staticmethod
    def _parse_score(raw_score: Any) -> int:
        try:
            score = int(round(float(raw_score)))
        except (TypeError, ValueError):
            score = 1
        return max(1, min(10, score))

    @staticmethod
    def _label_from_score(score: int) -> str:
        if score >= 10:
            return "通关胜利"
        if score <= 1:
            return "试错"
        return "有效倒水"

    @staticmethod
    def _clamp_speed(raw_speed: Any) -> float:
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            speed = 1.0
        return max(0.9, min(2.0, speed))

    @staticmethod
    def _parse_json_strict(raw_text: str) -> dict[str, Any]:
        """优先直接解析 JSON；失败则提取首个 JSON 对象片段。"""
        try:
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("模型输出 JSON 不是对象。")
            return data
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, flags=re.S)
            if not match:
                raise
            data = json.loads(match.group(0))
            if not isinstance(data, dict):
                raise ValueError("模型输出 JSON 不是对象。")
            return data

    def _record_duration(self, record: dict[str, Any], clip_path: Path) -> float:
        try:
            duration = float(record.get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            return duration

        probed = self._probe_video_duration(clip_path)
        if probed > 0:
            return probed

        try:
            start_time = float(record["source_start_time"])
            end_time = float(record["source_end_time"])
            return max(0.0, end_time - start_time)
        except (KeyError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _probe_video_duration(clip_path: Path) -> float:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return 0.0
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0 or frame_count <= 0:
                return 0.0
            return frame_count / fps
        finally:
            cap.release()

    @staticmethod
    def _load_segments(path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"segments.json 格式错误（应为列表）: {path}")
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _resolve_clip_path(record: dict[str, Any], base_dir: Path) -> Path:
        raw_path = str(record.get("clip_path", "")).strip()
        if not raw_path:
            raise ValueError("记录缺少 clip_path 字段。")
        path = Path(raw_path)
        if path.is_absolute():
            return path
        if path.exists():
            return path

        base_candidate = base_dir / path
        if base_candidate.exists():
            return base_candidate

        return base_dir / path.name

    @staticmethod
    def _write_json_list(path: Path, data: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)


VideoFineScorer = VideoFineFilter


def load_env_file(env_path: Path) -> None:
    """轻量读取 .env，避免强制引入 python-dotenv 依赖。"""
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
    parser = argparse.ArgumentParser(description="阶段二：VLM 语义过滤")
    parser.add_argument(
        "--interim-dir",
        default="data/interim",
        help="中间目录路径（默认 data/interim）",
    )
    parser.add_argument(
        "--video-name",
        default=None,
        help="仅处理指定视频子目录（例如 level3）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter 模型名（例如 google/gemini-flash-1.5 / openai/gpt-4o-mini）",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API Key（可选，默认读取 OPENROUTER_API_KEY）",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenRouter API Base URL（可选，默认读取 OPENROUTER_BASE_URL）",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=720,
        help="采样图缩放后的最长边（默认 720）",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG 压缩质量 40~95（默认 80）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    load_env_file(project_root / ".env")

    started_at = time.perf_counter()
    logger.info("=== 阶段二启动：VLM 语义过滤 ===")
    logger.info("项目根目录: {}", project_root)

    fine_filter = VideoFineFilter(
        interim_dir=args.interim_dir,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
    )
    scored = fine_filter.run(only_video=args.video_name)
    elapsed = time.perf_counter() - started_at

    logger.info("=== 阶段二完成：共判定 {} 条，耗时 {:.2f}s ===", len(scored), elapsed)


if __name__ == "__main__":
    main()
