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
    """阶段三：按真实动态时长统筹片段，并批量组装成 30 秒内成片版本。"""

    TARGET_DURATION = 30.0
    RANDOM_VERSION_COUNT = 6

    def __init__(
        self,
        interim_dir: str = "data/interim",
        processed_dir: str = "data/processed",
        seed: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self.interim_dir = Path(interim_dir)
        self.clips_root = self.interim_dir / "clips"
        self.processed_dir = Path(processed_dir)
        self.random = random.Random(seed)
        self.run_id = run_id or self._timestamp()

        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run(self, only_video: str | None = None) -> list[dict[str, Any]]:
        """遍历 scored_segments.json，为每个素材集合生成 10 个动态时长版本。"""
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
            and self._record_media_available(record, scored_path.parent)
        ]

        if not selected:
            logger.warning("无可用 selected 片段，跳过: {}", scored_path)
            return []

        selected.sort(key=self._record_sort_key)
        victory_clips = [
            record for record in selected if self._is_victory_clip(record)
        ]
        valid_clips = [
            record for record in selected if not self._is_victory_clip(record)
        ]
        all_clips = valid_clips + victory_clips

        victory_clip = victory_clips[-1] if victory_clips else None
        victory_time = self._effective_time(victory_clip) if victory_clip else 0.0
        forced_win_budget = max(0.0, self.TARGET_DURATION - victory_time)

        logger.info(
            "{} 数据准备完成 | 普通有效片段 {} 个 | 胜利片段 {} 个 | 总池 {} 个",
            video_name,
            len(valid_clips),
            len(victory_clips),
            len(all_clips),
        )
        if victory_clip:
            logger.info(
                "{} 强制胜利尾缀使用片段 {} | 胜利有效时长 {:.2f}s | 普通片段预算 {:.2f}s",
                video_name,
                victory_clip.get(
                    "id",
                    self._resolve_clip_path(victory_clip, scored_path.parent).name,
                ),
                victory_time,
                forced_win_budget,
            )
        else:
            logger.warning(
                "{} 未找到胜利片段，带胜利尾缀策略将退化为普通拼接，不追加尾缀",
                video_name,
            )

        timestamp = self.run_id
        output_dir = self.processed_dir / video_name / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = (
            scored_path.parent
            / f"tmp_speed_{timestamp}_{os.getpid()}_{self.random.randint(1000, 9999)}"
        )
        tmp_dir.mkdir(parents=True, exist_ok=False)

        try:
            plans = self._build_plans(
                valid_clips=valid_clips,
                victory_clip=victory_clip,
                all_clips=all_clips,
                forced_win_budget=forced_win_budget,
            )
            outputs: list[dict[str, Any]] = []

            for plan in plans:
                segments = plan["segments"]
                if not segments:
                    logger.warning("{} {} 组合为空，跳过", video_name, plan["log_name"])
                    continue

                total_time = self._total_effective_time(segments)
                logger.info(
                    "{} 正在生成 {} | 策略 {} | 选中片段 {} 个 | 预计时长 {:.2f}s",
                    video_name,
                    plan["log_name"],
                    plan["strategy"],
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
                    "speeds": [
                        self._safe_speed(record.get("speed", 1.0))
                        for record in segments
                    ],
                }
                outputs.append(output_record)
                logger.info(
                    "{} 成品生成完成: {} | 策略 {} | 片段数 {} | 预计时长 {:.2f}s",
                    video_name,
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
        *,
        valid_clips: list[dict[str, Any]],
        victory_clip: dict[str, Any] | None,
        all_clips: list[dict[str, Any]],
        forced_win_budget: float,
    ) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []

        sequential = self.fill_budget(
            sorted(valid_clips, key=self._record_sort_key),
            forced_win_budget,
        )
        plans.append(
            self._make_forced_win_plan(
                version=1,
                strategy="sequential_win",
                log_name="版本1_顺产型_强制胜利尾缀",
                file_stem="v01_sequential_win",
                clips=sequential,
                victory_clip=victory_clip,
            )
        )

        reverse = self.fill_budget(
            sorted(valid_clips, key=self._record_sort_key, reverse=True),
            forced_win_budget,
        )
        reverse.sort(key=self._record_sort_key)
        plans.append(
            self._make_forced_win_plan(
                version=2,
                strategy="reverse_win",
                log_name="版本2_逆袭型_强制胜利尾缀",
                file_stem="v02_reverse_win",
                clips=reverse,
                victory_clip=victory_clip,
            )
        )

        highscore = self.fill_budget(
            sorted(valid_clips, key=self._score_sort_key),
            forced_win_budget,
        )
        highscore.sort(key=self._record_sort_key)
        plans.append(
            self._make_forced_win_plan(
                version=3,
                strategy="highscore_win",
                log_name="版本3_高分型_强制胜利尾缀",
                file_stem="v03_highscore_win",
                clips=highscore,
                victory_clip=victory_clip,
            )
        )

        pure_sequential = self.fill_budget(
            sorted(all_clips, key=self._record_sort_key),
            self.TARGET_DURATION,
        )
        plans.append(
            self._make_plain_plan(
                version=4,
                strategy="pure_sequential",
                log_name="版本4_纯享顺序型_无强制胜利尾缀",
                file_stem="v04_pure_sequential",
                clips=pure_sequential,
            )
        )

        for random_index in range(1, self.RANDOM_VERSION_COUNT + 1):
            shuffled = list(valid_clips)
            self.random.shuffle(shuffled)
            random_clips = self.fill_budget(shuffled, forced_win_budget)
            random_clips.sort(key=self._record_sort_key)
            version = random_index + 4
            plans.append(
                self._make_forced_win_plan(
                    version=version,
                    strategy=f"random_{random_index}_win",
                    log_name=f"版本{version}_盲盒型_{random_index}_强制胜利尾缀",
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
        """按候选顺序累计真实有效时长，下一段超过预算时立即停止。"""
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

    def _make_forced_win_plan(
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

        logger.info(
            "{} 计划完成 | 普通片段 {} 个 | 是否追加胜利尾缀 {} | 最终片段 {} 个",
            log_name,
            len(clips),
            bool(victory_clip),
            len(segments),
        )
        return {
            "version": version,
            "strategy": strategy,
            "log_name": log_name,
            "file_stem": file_stem,
            "segments": segments,
        }

    @staticmethod
    def _make_plain_plan(
        *,
        version: int,
        strategy: str,
        log_name: str,
        file_stem: str,
        clips: list[dict[str, Any]],
    ) -> dict[str, Any]:
        logger.info("{} 计划完成 | 无强制尾缀 | 最终片段 {} 个", log_name, len(clips))
        return {
            "version": version,
            "strategy": strategy,
            "log_name": log_name,
            "file_stem": file_stem,
            "segments": list(clips),
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
            temp_output = temp_dir / f"speed_clip_{index:03d}.mp4"
            safe_speed = self._safe_speed(segment.get("speed", 1.0))
            media_name = self._describe_segment_media(segment, base_dir)
            logger.info(
                "片段变速中: {} | speed {:.2f} | 预计有效时长 {:.2f}s",
                media_name,
                safe_speed,
                self._effective_time(segment),
            )
            self._render_speed_adjusted_segment(
                segment,
                base_dir,
                temp_output,
                safe_speed,
            )
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

    def _render_speed_adjusted_segment(
        self,
        segment: dict[str, Any],
        base_dir: Path,
        temp_output: Path,
        safe_speed: float,
    ) -> None:
        temp_output.parent.mkdir(parents=True, exist_ok=True)
        source_path = self._resolve_source_path(segment)
        source_range = self._segment_source_range(segment)
        if source_path is not None and source_path.exists() and source_range is not None:
            start_time, duration = source_range
            self._render_speed_adjusted_source_range(
                source_path=source_path,
                start_time=start_time,
                duration=duration,
                temp_output=temp_output,
                safe_speed=safe_speed,
            )
            return

        input_clip = self._resolve_clip_path(segment, base_dir)
        logger.warning(
            "片段缺少可用原视频映射，回退使用阶段一短片: {}",
            input_clip,
        )
        self._render_speed_adjusted_clip(input_clip, temp_output, safe_speed)

    def _render_speed_adjusted_source_range(
        self,
        source_path: Path,
        start_time: float,
        duration: float,
        temp_output: Path,
        safe_speed: float,
    ) -> None:
        v_pts = 1.0 / safe_speed
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-filter_complex",
            (
                f"[0:v]trim=start={start_time:.6f}:duration={duration:.6f},"
                f"setpts=PTS-STARTPTS,setpts={v_pts:.6f}*PTS[v];"
                f"[0:a]atrim=start={start_time:.6f}:duration={duration:.6f},"
                f"asetpts=PTS-STARTPTS,atempo={safe_speed:.6f},"
                "aresample=async=1:first_pts=0[a]"
            ),
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
            "-b:a",
            "128k",
            str(temp_output),
        ]
        self._run_ffmpeg(command, f"原视频映射剪辑失败: {source_path}")

    def _render_speed_adjusted_clip(
        self,
        input_clip: Path,
        temp_output: Path,
        safe_speed: float,
    ) -> None:
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
    def _resolve_source_path(record: dict[str, Any]) -> Path | None:
        raw_path = str(record.get("source_path", "")).strip()
        if not raw_path:
            return None

        path = Path(raw_path)
        if path.is_absolute() or path.exists():
            return path

        project_candidate = PROJECT_ROOT / path
        if project_candidate.exists():
            return project_candidate

        return path

    @staticmethod
    def _segment_source_range(record: dict[str, Any]) -> tuple[float, float] | None:
        try:
            start_time = float(record["source_start_time"])
            end_time = float(record["source_end_time"])
        except (KeyError, TypeError, ValueError):
            return None

        duration = max(0.0, end_time - start_time)
        if duration <= 0:
            return None
        return max(0.0, start_time), duration

    def _describe_segment_media(self, record: dict[str, Any], base_dir: Path) -> str:
        source_path = self._resolve_source_path(record)
        source_range = self._segment_source_range(record)
        if source_path is not None and source_path.exists() and source_range is not None:
            start_time, duration = source_range
            return f"{source_path.name} @ {start_time:.3f}s + {duration:.3f}s"
        return self._resolve_clip_path(record, base_dir).name

    def _record_media_available(self, record: dict[str, Any], base_dir: Path) -> bool:
        source_path = self._resolve_source_path(record)
        if (
            source_path is not None
            and source_path.exists()
            and self._segment_source_range(record) is not None
        ):
            return True

        try:
            return self._resolve_clip_path(record, base_dir).exists()
        except ValueError:
            return False

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
        return self._record_duration(record) / self._safe_speed(
            record.get("speed", 1.0)
        )

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

    def _record_duration(self, record: dict[str, Any]) -> float:
        try:
            duration = float(record.get("duration", 4.0))
        except (TypeError, ValueError):
            source_range = self._segment_source_range(record)
            if source_range is not None:
                _, duration = source_range
            else:
                duration = 4.0

        if duration <= 0:
            return 4.0
        return duration

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
    parser = argparse.ArgumentParser(
        description="阶段三：动态时长统筹与变速组装"
    )
    parser.add_argument(
        "--interim-dir",
        default="data/interim",
        help="中间目录路径（默认 data/interim）",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="成品输出目录（默认 data/processed）",
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
        help="盲盒策略随机种子（默认不固定）",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="本次运行输出目录名（默认: 当前时间 YYYYMMDD_HHMMSS）",
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
        run_id=args.run_id,
    )
    outputs = assembler.run(only_video=args.video_name)
    elapsed = time.perf_counter() - started_at

    logger.info(
        "=== 阶段三完成：生成 {} 个成品，耗时 {:.2f}s ===",
        len(outputs),
        elapsed,
    )


if __name__ == "__main__":
    main()
