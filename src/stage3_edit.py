# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    str(LOGS_DIR / "stage3_{time:YYYYMMDD_HHmmss}.log"),
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)


class VideoHighlightAssembler:
    """阶段三：基于 scored_segments.json 生成多节奏高光成片。"""

    NORMAL_CLIP_COUNT = 6
    BLIND_BOX_COUNT = 7

    def __init__(
        self,
        interim_dir: str = "data/interim",
        processed_dir: str = "data/processed",
        seed: int | None = None,
    ) -> None:
        self.interim_dir = Path(interim_dir)
        self.clips_root = self.interim_dir / "clips"
        self.processed_dir = Path(processed_dir)
        self.random = random.Random(seed)

        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run(self, only_video: str | None = None) -> list[dict[str, Any]]:
        """遍历 scored_segments.json，为每个原始视频生成 10 个成品。"""
        if not self.clips_root.exists():
            logger.warning("未找到 clips 目录: {}", self.clips_root)
            return []

        scored_files = sorted(self.clips_root.glob("*/scored_segments.json"))
        if only_video:
            scored_files = [
                path for path in scored_files if path.parent.name == only_video.strip()
            ]

        if not scored_files:
            logger.warning("未找到可组装的 scored_segments.json，目录: {}", self.clips_root)
            return []

        outputs: list[dict[str, Any]] = []
        logger.info("阶段三启动 | 待组装视频数 {}", len(scored_files))
        for scored_path in scored_files:
            video_outputs = self._process_video(scored_path)
            outputs.extend(video_outputs)

        logger.info("阶段三结束 | 生成成品数 {}", len(outputs))
        return outputs

    def _process_video(self, scored_path: Path) -> list[dict[str, Any]]:
        video_name = scored_path.parent.name
        records = self._load_records(scored_path)
        selected = [
            record
            for record in records
            if record.get("selected") is True and self._resolve_clip_path(record, scored_path.parent).exists()
        ]

        if not selected:
            logger.warning("无可用 selected 片段，跳过: {}", scored_path)
            return []

        selected.sort(key=self._record_sort_key)
        victory_clips = [record for record in selected if "胜利" in str(record.get("label", ""))]
        valid_clips = [record for record in selected if record not in victory_clips]
        victory_clip = victory_clips[-1] if victory_clips else None

        if victory_clip:
            logger.info(
                "{} 使用最晚胜利片段压轴: {}",
                video_name,
                victory_clip.get("id", self._resolve_clip_path(victory_clip, scored_path.parent).name),
            )
        else:
            logger.warning("{} 未找到胜利片段，将仅使用普通有效倒水片段。", video_name)

        plans = self._build_plans(valid_clips, victory_clip)
        outputs: list[dict[str, Any]] = []

        for order, plan in enumerate(plans, start=1):
            clip_paths = [
                self._resolve_clip_path(record, scored_path.parent)
                for record in plan["segments"]
            ]
            if not clip_paths:
                logger.warning("{} 组合为空，跳过: {}", video_name, plan["name"])
                continue

            output_path = self.processed_dir / f"{video_name}_{order:02d}_{plan['name']}.mp4"
            self._concat_videos_ffmpeg(clip_paths, output_path)
            output_record = {
                "source_video": video_name,
                "strategy": plan["name"],
                "output_path": self._json_path(output_path),
                "clip_count": len(clip_paths),
                "clip_ids": [str(record.get("id", "")) for record in plan["segments"]],
            }
            outputs.append(output_record)
            logger.info(
                "成品生成完成: {} | 策略 {} | 片段数 {}",
                output_path,
                plan["name"],
                len(clip_paths),
            )

        return outputs

    def _build_plans(
        self,
        valid_clips: list[dict[str, Any]],
        victory_clip: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []
        take_count = min(self.NORMAL_CLIP_COUNT, len(valid_clips))

        earliest = valid_clips[:take_count]
        plans.append({"name": "sequential", "segments": self._with_victory(earliest, victory_clip)})

        latest = valid_clips[-take_count:] if take_count else []
        plans.append({"name": "comeback", "segments": self._with_victory(latest, victory_clip)})

        panoramic = self._pick_panoramic(valid_clips, take_count)
        plans.append({"name": "panoramic", "segments": self._with_victory(panoramic, victory_clip)})

        seen_orders = {
            self._plan_key(plan["segments"])
            for plan in plans
        }
        blind_box_plans = self._build_blind_box_plans(valid_clips, victory_clip, seen_orders)
        plans.extend(blind_box_plans)
        return plans

    def _build_blind_box_plans(
        self,
        valid_clips: list[dict[str, Any]],
        victory_clip: dict[str, Any] | None,
        seen_orders: set[tuple[str, ...]],
    ) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []
        take_count = min(self.NORMAL_CLIP_COUNT, len(valid_clips))
        max_attempts = 200

        for version in range(1, self.BLIND_BOX_COUNT + 1):
            selected: list[dict[str, Any]] = []
            for _ in range(max_attempts):
                if take_count == 0:
                    candidate = []
                elif len(valid_clips) >= self.NORMAL_CLIP_COUNT:
                    candidate = self.random.sample(valid_clips, self.NORMAL_CLIP_COUNT)
                    self.random.shuffle(candidate)
                else:
                    # 片段不足时按发生顺序拼接，避免把短素材进一步打乱。
                    candidate = valid_clips[:take_count]

                segments = self._with_victory(candidate, victory_clip)
                key = self._plan_key(segments)
                selected = segments
                if key not in seen_orders or len(valid_clips) < self.NORMAL_CLIP_COUNT:
                    seen_orders.add(key)
                    break

            plans.append({"name": f"blindbox_{version:02d}", "segments": selected})

        return plans

    def _pick_panoramic(
        self,
        valid_clips: list[dict[str, Any]],
        take_count: int,
    ) -> list[dict[str, Any]]:
        if take_count <= 0:
            return []
        if len(valid_clips) <= take_count:
            return valid_clips[:take_count]

        picked: list[dict[str, Any]] = []
        total = len(valid_clips)
        for index in range(take_count):
            start = int(index * total / take_count)
            end = int((index + 1) * total / take_count)
            bucket = valid_clips[start:max(start + 1, end)]
            picked.append(bucket[len(bucket) // 2])
        return picked

    @staticmethod
    def _with_victory(
        clips: list[dict[str, Any]],
        victory_clip: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if victory_clip is None:
            return list(clips)
        return [*clips, victory_clip]

    def _concat_videos_ffmpeg(self, clips: list[Path], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as file_obj:
            concat_list_path = Path(file_obj.name)
            for clip in clips:
                file_obj.write(f"file '{self._escape_concat_path(clip.resolve())}'\n")

        try:
            command = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(output_path),
            ]
            self._run_ffmpeg(command, f"无损拼接失败: {output_path}")
        finally:
            concat_list_path.unlink(missing_ok=True)

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
            raise RuntimeError("未找到 ffmpeg，请确认已安装并加入 PATH。") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip() or "ffmpeg 未返回详细错误。"
            raise RuntimeError(f"{error_prefix}\n{detail}") from exc

    @staticmethod
    def _load_records(path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"scored_segments.json 格式错误（应为列表）: {path}")
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
    def _record_sort_key(record: dict[str, Any]) -> tuple[float, str]:
        raw_start = record.get("start_time", 0.0)
        try:
            start_time = float(raw_start)
        except (TypeError, ValueError):
            start_time = 0.0
        return start_time, str(record.get("id", ""))

    @staticmethod
    def _plan_key(segments: list[dict[str, Any]]) -> tuple[str, ...]:
        return tuple(str(record.get("id", record.get("clip_path", ""))) for record in segments)

    @staticmethod
    def _escape_concat_path(path: Path) -> str:
        return path.as_posix().replace("'", "'\\''")

    @staticmethod
    def _json_path(path: Path) -> str:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段三：极速组装高光成片")
    parser.add_argument(
        "--interim-dir",
        default="data/interim",
        help="中间目录路径（默认: data/interim）",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="成品输出目录（默认: data/processed）",
    )
    parser.add_argument(
        "--video-name",
        default=None,
        help="仅处理指定视频子目录（例如 level3）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="盲盒策略随机种子（默认: 不固定）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    started_at = time.perf_counter()
    logger.info("=== 阶段三启动：极速组装高光成片 ===")
    logger.info("项目根目录: {}", project_root)

    assembler = VideoHighlightAssembler(
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        seed=args.seed,
    )
    outputs = assembler.run(only_video=args.video_name)
    elapsed = time.perf_counter() - started_at

    logger.info("=== 阶段三完成：生成 {} 个成品，耗时 {:.2f}s ===", len(outputs), elapsed)


if __name__ == "__main__":
    main()
