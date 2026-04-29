# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
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


class VideoAssembler:
    """阶段三：按 VLM 给出的 speed 动态统筹时长并批量组装成片。"""

    BASE_DURATION = 4.0
    TARGET_DURATION = 30.0
    RANDOM_VERSION_COUNT = 7

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
        """遍历 scored_segments.json，为每个原始视频生成 10 个动态时长版本。"""
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
            outputs.extend(self._process_video(scored_path))

        logger.info("阶段三结束 | 生成成品数 {}", len(outputs))
        return outputs

    def _process_video(self, scored_path: Path) -> list[dict[str, Any]]:
        video_name = scored_path.parent.name
        records = self._load_records(scored_path)
        selected = [
            record
            for record in records
            if record.get("selected") is True
            and self._resolve_clip_path(record, scored_path.parent).exists()
        ]

        if not selected:
            logger.warning("无可用 selected 片段，跳过: {}", scored_path)
            return []

        selected.sort(key=self._record_sort_key)
        victory_clips = [record for record in selected if self._is_victory_clip(record)]
        valid_clips = [record for record in selected if not self._is_victory_clip(record)]
        victory_clip = victory_clips[-1] if victory_clips else None
        victory_time = self._effective_time(victory_clip) if victory_clip else 0.0
        budget = max(0.0, self.TARGET_DURATION - victory_time)

        if victory_clip:
            logger.info(
                "{} 使用最晚胜利片段压轴: {} | 胜利有效时长 {:.2f}s | 普通片段预算 {:.2f}s",
                video_name,
                victory_clip.get("id", self._resolve_clip_path(victory_clip, scored_path.parent).name),
                victory_time,
                budget,
            )
        else:
            logger.warning("{} 未找到胜利片段，将全部 30 秒预算用于普通有效片段。", video_name)

        timestamp = self._timestamp()
        output_dir = self.processed_dir / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = scored_path.parent / f"tmp_speed_{timestamp}_{os.getpid()}_{self.random.randint(1000, 9999)}"
        tmp_dir.mkdir(parents=True, exist_ok=False)

        try:
            plans = self._build_plans(valid_clips, victory_clip, budget)
            outputs: list[dict[str, Any]] = []

            for plan in plans:
                segments = plan["segments"]
                if not segments:
                    logger.warning("{} {} 组合为空，跳过。", video_name, plan["log_name"])
                    continue

                total_time = self._total_effective_time(segments)
                logger.info(
                    "正在生成 {}，共选中 {} 个片段，预计合成时长 {:.2f} 秒",
                    plan["log_name"],
                    len(segments),
                    total_time,
                )

                output_path = output_dir / f"{plan['file_stem']}_{timestamp}.mp4"
                plan_tmp_dir = tmp_dir / plan["file_stem"]
                self._render_plan(
                    segments=segments,
                    base_dir=scored_path.parent,
                    temp_dir=plan_tmp_dir,
                    output_path=output_path,
                )

                output_record = {
                    "source_video": video_name,
                    "strategy": plan["strategy"],
                    "output_path": self._json_path(output_path),
                    "clip_count": len(segments),
                    "estimated_duration": round(total_time, 3),
                    "clip_ids": [str(record.get("id", "")) for record in segments],
                    "speeds": [self._safe_speed(record.get("speed", 1.0)) for record in segments],
                }
                outputs.append(output_record)
                logger.info(
                    "成品生成完成: {} | 策略 {} | 片段数 {} | 预计时长 {:.2f}s",
                    output_path,
                    plan["strategy"],
                    len(segments),
                    total_time,
                )

            return outputs
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("已清理阶段三临时目录: {}", tmp_dir)

    def _build_plans(
        self,
        valid_clips: list[dict[str, Any]],
        victory_clip: dict[str, Any] | None,
        budget: float,
    ) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []

        sequential = self.fill_budget(
            sorted(valid_clips, key=self._record_sort_key),
            budget,
        )
        plans.append(
            self._make_plan(
                version=1,
                strategy="sequential",
                log_name="版本1_顺产型",
                file_stem="v01_sequential",
                clips=sequential,
                victory_clip=victory_clip,
            )
        )

        reverse = self.fill_budget(
            sorted(valid_clips, key=self._record_sort_key, reverse=True),
            budget,
        )
        reverse.sort(key=self._record_sort_key)
        plans.append(
            self._make_plan(
                version=2,
                strategy="reverse",
                log_name="版本2_逆袭型",
                file_stem="v02_reverse",
                clips=reverse,
                victory_clip=victory_clip,
            )
        )

        highscore = self.fill_budget(
            sorted(valid_clips, key=self._score_sort_key),
            budget,
        )
        highscore.sort(key=self._record_sort_key)
        plans.append(
            self._make_plan(
                version=3,
                strategy="highscore",
                log_name="版本3_高分型",
                file_stem="v03_highscore",
                clips=highscore,
                victory_clip=victory_clip,
            )
        )

        for random_index in range(1, self.RANDOM_VERSION_COUNT + 1):
            shuffled = list(valid_clips)
            self.random.shuffle(shuffled)
            random_clips = self.fill_budget(shuffled, budget)
            random_clips.sort(key=self._record_sort_key)
            version = random_index + 3
            plans.append(
                self._make_plan(
                    version=version,
                    strategy=f"random_{random_index}",
                    log_name=f"版本{version}_盲盒型_{random_index}",
                    file_stem=f"v{version:02d}_random_{random_index}",
                    clips=random_clips,
                    victory_clip=victory_clip,
                )
            )

        return plans

    def fill_budget(
        self,
        clip_list: list[dict[str, Any]],
        budget: float,
    ) -> list[dict[str, Any]]:
        """按候选顺序累加有效时长，超过预算立即停止。"""
        selected: list[dict[str, Any]] = []
        used_time = 0.0

        for clip in clip_list:
            effective_time = self._effective_time(clip)
            if used_time + effective_time <= budget:
                selected.append(clip)
                used_time += effective_time
            else:
                break

        return selected

    def _make_plan(
        self,
        *,
        version: int,
        strategy: str,
        log_name: str,
        file_stem: str,
        clips: list[dict[str, Any]],
        victory_clip: dict[str, Any] | None,
    ) -> dict[str, Any]:
        segments = list(clips)
        if victory_clip:
            segments.append(victory_clip)

        return {
            "version": version,
            "strategy": strategy,
            "log_name": log_name,
            "file_stem": file_stem,
            "segments": segments,
        }

    def _render_plan(
        self,
        segments: list[dict[str, Any]],
        base_dir: Path,
        temp_dir: Path,
        output_path: Path,
    ) -> None:
        temp_dir.mkdir(parents=True, exist_ok=False)

        speed_adjusted_clips: list[Path] = []
        for index, segment in enumerate(segments, start=1):
            input_clip = self._resolve_clip_path(segment, base_dir)
            temp_output = temp_dir / f"speed_clip_{index:03d}.mp4"
            safe_speed = self._safe_speed(segment.get("speed", 1.0))
            logger.info(
                "片段变速中: {} | speed {:.2f} | 预计有效时长 {:.2f}s",
                input_clip.name,
                safe_speed,
                self.BASE_DURATION / safe_speed,
            )
            self._render_speed_adjusted_clip(input_clip, temp_output, safe_speed)
            speed_adjusted_clips.append(temp_output)

        concat_list_path = temp_dir / "concat_list.txt"
        with concat_list_path.open("w", encoding="utf-8") as file_obj:
            for clip in speed_adjusted_clips:
                file_obj.write(f"file '{self._escape_concat_path(clip.resolve())}'\n")

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]
        self._run_ffmpeg(command, f"最终拼接失败: {output_path}")

    def _render_speed_adjusted_clip(
        self,
        input_clip: Path,
        temp_output: Path,
        safe_speed: float,
    ) -> None:
        temp_output.parent.mkdir(parents=True, exist_ok=True)
        v_pts = 1.0 / safe_speed
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_clip),
            "-filter_complex",
            f"[0:v]setpts={v_pts:.6f}*PTS[v];[0:a]atempo={safe_speed:.6f}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            str(temp_output),
        ]
        self._run_ffmpeg(command, f"片段变速失败: {input_clip}")

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

    def _score_sort_key(self, record: dict[str, Any]) -> tuple[int, float, str]:
        raw_score = record.get("score", 0)
        try:
            score = int(round(float(raw_score)))
        except (TypeError, ValueError):
            score = 0

        start_time, record_id = self._record_sort_key(record)
        return -score, start_time, record_id

    def _effective_time(self, record: dict[str, Any] | None) -> float:
        if record is None:
            return 0.0
        return self.BASE_DURATION / self._safe_speed(record.get("speed", 1.0))

    def _total_effective_time(self, segments: list[dict[str, Any]]) -> float:
        return sum(self._effective_time(segment) for segment in segments)

    @staticmethod
    def _is_victory_clip(record: dict[str, Any]) -> bool:
        label = str(record.get("label", ""))
        score = record.get("score")
        try:
            numeric_score = int(round(float(score)))
        except (TypeError, ValueError):
            numeric_score = 0
        return "胜利" in label or numeric_score >= 10

    @staticmethod
    def _safe_speed(speed: Any) -> float:
        try:
            parsed_speed = float(speed)
        except (TypeError, ValueError):
            parsed_speed = 1.0
        return max(0.5, min(2.0, parsed_speed))

    @staticmethod
    def _escape_concat_path(path: Path) -> str:
        return path.as_posix().replace("'", "'\\''")

    @staticmethod
    def _json_path(path: Path) -> str:
        return path.as_posix()

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y%m%d_%H%M%S")


VideoHighlightAssembler = VideoAssembler


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段三：动态时长统筹与变速组装")
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
        help="仅处理指定视频子目录（例如 level4）",
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
    logger.info("=== 阶段三启动：动态时长统筹与变速组装 ===")
    logger.info("项目根目录: {}", project_root)

    assembler = VideoAssembler(
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        seed=args.seed,
    )
    outputs = assembler.run(only_video=args.video_name)
    elapsed = time.perf_counter() - started_at

    logger.info("=== 阶段三完成：生成 {} 个成品，耗时 {:.2f}s ===", len(outputs), elapsed)


if __name__ == "__main__":
    main()
