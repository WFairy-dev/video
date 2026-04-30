# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    str(LOGS_DIR / "stage1_{time:YYYYMMDD}.log"),
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)


class VideoCoarseFilter:
    """阶段一粗筛：MOG2 运动信号提取 + 动态启停阈值切片。"""

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    RESIZE_WIDTH = 640
    TOP_CROP_RATIO = 0.15
    BOTTOM_CROP_RATIO = 0.15
    ENERGY_THRESHOLD = 20000
    PRE_BUFFER_SECONDS = 0.5
    POST_BUFFER_SECONDS = 0.5
    QUIET_SECONDS = 1.0
    MOG2_HISTORY = 100
    MOG2_VAR_THRESHOLD = 50

    def __init__(
        self,
        raw_dir: str,
        interim_dir: str,
        fps: int = 5,
        *,
        raw_subdir: str | None = None,
        recursive: bool = False,
        energy_threshold: float | None = None,
    ):
        self.raw_path = Path(raw_dir)
        if self.raw_path.is_file():
            collection_source = self.raw_path.stem
        elif raw_subdir:
            collection_source = Path(raw_subdir).name
        else:
            collection_source = self.raw_path.name
        self.collection_name = self._safe_stem(collection_source)

        if raw_subdir:
            if self.raw_path.is_file():
                logger.warning("raw_subdir 已忽略：raw_dir 当前是单文件 {}", self.raw_path)
            else:
                self.raw_path = self.raw_path / Path(raw_subdir)

        self.recursive = recursive
        self.interim_dir = Path(interim_dir)
        self.clips_dir = self.interim_dir / "clips"
        self.frames_dir = self.interim_dir / "frames"
        self.target_fps = max(1, int(fps))
        self.energy_threshold = (
            float(energy_threshold)
            if energy_threshold is not None
            else float(self.ENERGY_THRESHOLD)
        )
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> list[dict[str, Any]]:
        """执行阶段一：提取一维运动能量，按动态启停阈值导出不定长片段。"""
        started_at = time.perf_counter()
        video_files = self._iter_video_files()
        all_records: list[dict[str, Any]] = []

        if not video_files:
            logger.warning("未找到可处理的视频: {}", self.raw_path)
            return all_records

        logger.info("阶段一启动 | 待处理视频数 {}", len(video_files))

        collection_name = self.collection_name
        clip_output_dir = self.clips_dir / collection_name
        frame_output_dir = self.frames_dir / collection_name
        self._prepare_collection_output_dirs(collection_name, clip_output_dir, frame_output_dir)

        next_index = 1
        timeline_offset = 0.0
        for video_file in video_files:
            logger.info("开始处理视频 {}", video_file)
            duration = self._get_video_duration(video_file)
            signal_started = time.perf_counter()
            energy_signal, timestamps, analysis_fps, _ = self._extract_motion_signal(video_file)
            signal_elapsed = time.perf_counter() - signal_started

            segments, signal_peak = self._select_segments_with_state_machine(
                energy_signal=energy_signal,
                timestamps=timestamps,
                analysis_fps=analysis_fps,
                duration=duration,
            )
            logger.info(
                "视频分析完成: {} | 分析耗时 {:.2f}s | 信号最大峰值 {:.1f} | 最终片段数 {}",
                video_file.name,
                signal_elapsed,
                signal_peak,
                len(segments),
            )

            records = self._extract_and_save(
                video_file,
                segments,
                collection_name=collection_name,
                clip_output_dir=clip_output_dir,
                frame_output_dir=frame_output_dir,
                start_index=next_index,
                timeline_offset=timeline_offset,
            )
            all_records.extend(records)
            next_index += len(records)
            timeline_offset += max(0.0, duration)

        self._write_collection_segments_json(clip_output_dir, all_records)
        elapsed = time.perf_counter() - started_at
        logger.info(
            "阶段一结束 | 素材集合 {} | 导出总片段 {} | 总耗时 {:.2f}s",
            collection_name,
            len(all_records),
            elapsed,
        )
        return all_records

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """缩放并裁掉顶部/底部 UI 区域，降低运动检测噪声。"""
        if frame is None or frame.size == 0:
            raise ValueError("输入帧为空，无法预处理。")

        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("输入帧尺寸异常，无法预处理。")

        scale = self.RESIZE_WIDTH / float(width)
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            frame,
            (self.RESIZE_WIDTH, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        top = int(resized_height * self.TOP_CROP_RATIO)
        bottom = int(resized_height * (1.0 - self.BOTTOM_CROP_RATIO))
        if bottom <= top:
            return resized
        return resized[top:bottom, :]

    def _extract_motion_signal(
        self,
        video_path: Path,
    ) -> tuple[list[int], list[float], float, float]:
        """使用 MOG2 提取每个采样帧的非零像素数，形成一维运动能量曲线。"""
        started_at = time.perf_counter()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {video_path}")

        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            if source_fps <= 0:
                source_fps = 30.0

            frame_skip = max(1, int(round(source_fps / self.target_fps)))
            analysis_fps = source_fps / frame_skip

            mog2 = cv2.createBackgroundSubtractorMOG2(
                history=self.MOG2_HISTORY,
                varThreshold=self.MOG2_VAR_THRESHOLD,
                detectShadows=False,
            )
            energy_signal: list[int] = []
            timestamps: list[float] = []

            frame_index = 0
            while True:
                if not cap.grab():
                    break
                if frame_index % frame_skip != 0:
                    frame_index += 1
                    continue

                ok, frame = cap.retrieve()
                if not ok:
                    frame_index += 1
                    continue

                processed = self._preprocess_frame(frame)
                fg_mask = mog2.apply(processed)
                cleaned_mask = cv2.morphologyEx(
                    fg_mask,
                    cv2.MORPH_OPEN,
                    self.morph_kernel,
                )
                energy_signal.append(int(cv2.countNonZero(cleaned_mask)))
                timestamps.append(frame_index / source_fps)
                frame_index += 1

            elapsed = time.perf_counter() - started_at
            logger.info(
                "MOG2 信号提取完成: {} | 采样帧数 {} | 分析 FPS {:.3f} | 耗时 {:.2f}s",
                video_path.name,
                len(energy_signal),
                analysis_fps,
                elapsed,
            )
            return energy_signal, timestamps, analysis_fps, elapsed
        finally:
            cap.release()

    def _select_segments_with_state_machine(
        self,
        energy_signal: list[int],
        timestamps: list[float],
        analysis_fps: float,
        duration: float,
    ) -> tuple[list[tuple[float, float]], float]:
        """按能量阈值启停录制，静默超过 1 秒后结束片段。"""
        if not energy_signal or not timestamps or analysis_fps <= 0:
            return [], 0.0

        signal_array = np.asarray(energy_signal, dtype=np.float64)
        signal_peak = float(signal_array.max()) if signal_array.size else 0.0

        candidate_intervals: list[tuple[float, float]] = []
        is_recording = False
        start_time = 0.0
        quiet_started_at: float | None = None

        for raw_energy, raw_timestamp in zip(energy_signal, timestamps, strict=False):
            energy = float(raw_energy)
            timestamp = float(raw_timestamp)

            if energy > self.energy_threshold:
                if not is_recording:
                    is_recording = True
                    start_time = max(0.0, timestamp - self.PRE_BUFFER_SECONDS)
                    logger.debug(
                        "Trigger On: {:.3f}s | buffered start {:.3f}s | energy {:.1f}",
                        timestamp,
                        start_time,
                        energy,
                    )
                quiet_started_at = None
                continue

            if not is_recording:
                continue

            if quiet_started_at is None:
                quiet_started_at = timestamp
                continue

            if timestamp - quiet_started_at >= self.QUIET_SECONDS:
                end_time = timestamp + self.POST_BUFFER_SECONDS
                if duration > 0:
                    end_time = min(duration, end_time)
                if end_time > start_time:
                    candidate_intervals.append((start_time, end_time))
                    logger.debug(
                        "Trigger Off: {:.3f}s | buffered end {:.3f}s | duration {:.3f}s",
                        timestamp,
                        end_time,
                        end_time - start_time,
                    )
                is_recording = False
                quiet_started_at = None

        if is_recording:
            end_time = duration if duration > 0 else float(timestamps[-1]) + self.POST_BUFFER_SECONDS
            end_time = max(end_time, float(timestamps[-1]))
            if end_time > start_time:
                candidate_intervals.append((start_time, end_time))

        merged_intervals = self._merge_intervals(candidate_intervals)

        if not merged_intervals:
            logger.info(
                "无动态片段超过阈值 threshold {:.1f} | 信号峰值 {:.1f}",
                self.energy_threshold,
                signal_peak,
            )
            return [], signal_peak

        logger.info(
            "动态阈值筛选完成 | 阈值 {:.1f} | 信号峰值 {:.1f} | 片段数 {}",
            self.energy_threshold,
            signal_peak,
            len(merged_intervals),
        )
        return merged_intervals, signal_peak

    @staticmethod
    def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        sorted_intervals = sorted(intervals, key=lambda interval: interval[0])
        merged_intervals: list[tuple[float, float]] = []

        for start_time, end_time in sorted_intervals:
            if not merged_intervals or start_time > merged_intervals[-1][1]:
                merged_intervals.append((start_time, end_time))
                continue

            last_start_time, last_end_time = merged_intervals[-1]
            merged_intervals[-1] = (last_start_time, max(last_end_time, end_time))

        return merged_intervals

    def _extract_and_save(
        self,
        video_path: Path,
        segments: list[tuple[float, float]],
        *,
        collection_name: str,
        clip_output_dir: Path,
        frame_output_dir: Path,
        start_index: int,
        timeline_offset: float,
    ) -> list[dict[str, Any]]:
        """用 FFmpeg 导出动态片段，并写出每段真实 duration。"""
        started_at = time.perf_counter()
        video_name = self._safe_stem(video_path.stem)

        if not segments:
            logger.info("无可导出片段: {}", video_path.name)
            return []

        total = len(segments)
        records: list[dict[str, Any]] = []
        logger.info("开始 FFmpeg 导出: {} | 片段总数 {}", video_path.name, total)

        for index, (start_time, end_time) in enumerate(segments, start=1):
            duration = max(0.0, end_time - start_time)
            if duration <= 0:
                continue

            global_index = start_index + len(records)
            clip_path = clip_output_dir / f"{collection_name}_clip_{global_index:03d}.mp4"
            frame_path = frame_output_dir / f"{collection_name}_frame_{global_index:03d}.jpg"
            mid_time = start_time + (duration / 2.0)

            logger.info(
                "FFmpeg 进度 [{}/{}] 截取片段 {} ({:.3f}s ~ {:.3f}s, duration {:.3f}s)",
                index,
                total,
                clip_path.name,
                start_time,
                end_time,
                duration,
            )
            self._run_ffmpeg_clip(video_path, start_time, duration, clip_path)
            self._run_ffmpeg_frame(clip_path, max(0.0, mid_time - start_time), frame_path)

            records.append(
                {
                    "id": f"{collection_name}_{global_index:03d}",
                    "source_video": collection_name,
                    "source_file": video_name,
                    "source_path": self._json_path(video_path),
                    "source_start_time": round(start_time, 3),
                    "source_end_time": round(end_time, 3),
                    "start_time": round(timeline_offset + start_time, 3),
                    "end_time": round(timeline_offset + end_time, 3),
                    "duration": round(duration, 3),
                    "clip_path": self._json_path(clip_path),
                    "frame_path": self._json_path(frame_path),
                }
            )

        elapsed = time.perf_counter() - started_at
        logger.info(
            "FFmpeg 导出完成: {} | 成功 {} 段 | 耗时 {:.2f}s",
            video_path.name,
            len(records),
            elapsed,
        )
        return records

    def _iter_video_files(self) -> list[Path]:
        if self.raw_path.is_file():
            if self.raw_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                return [self.raw_path]
            logger.warning("输入文件格式不受支持: {}", self.raw_path)
            return []

        if not self.raw_path.exists():
            logger.warning("输入路径不存在: {}", self.raw_path)
            return []

        def _is_video(path: Path) -> bool:
            return path.is_file() and path.suffix.lower() in self.VIDEO_EXTENSIONS

        if self.recursive:
            return sorted(path for path in self.raw_path.rglob("*") if _is_video(path))
        return sorted(path for path in self.raw_path.iterdir() if _is_video(path))

    def _get_video_duration(self, video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("无法读取视频时长: {}", video_path)
            return 0.0

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0 or frame_count <= 0:
                return 0.0
            return frame_count / fps
        finally:
            cap.release()

    def _prepare_collection_output_dirs(
        self,
        collection_name: str,
        clip_output_dir: Path,
        frame_output_dir: Path,
    ) -> None:
        clip_output_dir.mkdir(parents=True, exist_ok=True)
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        for old_clip in clip_output_dir.glob(f"{collection_name}_clip_*.mp4"):
            old_clip.unlink()
        for old_frame in frame_output_dir.glob(f"{collection_name}_frame_*.jpg"):
            old_frame.unlink()
        for manifest_name in ("segments.json", "scored_segments.json"):
            manifest_path = clip_output_dir / manifest_name
            if manifest_path.exists():
                manifest_path.unlink()

    def _write_collection_segments_json(
        self,
        clip_output_dir: Path,
        records: list[dict[str, Any]],
    ) -> None:
        segments_path = clip_output_dir / "segments.json"
        self._write_json_list(segments_path, records)
        logger.info("已写入素材集合清单: {}", segments_path)

    def _run_ffmpeg_clip(
        self,
        input_path: Path,
        start_time: float,
        duration: float,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            self._format_seconds(start_time),
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
            "23",
            "-bsf:v",
            "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-af",
            "aresample=async=1:first_pts=0",
            str(output_path),
        ]
        self._run_ffmpeg(command, f"截取视频失败: {output_path}")

    def _run_ffmpeg_frame(
        self,
        input_path: Path,
        timestamp: float,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            self._format_seconds(timestamp),
            "-i",
            str(input_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
        try:
            self._run_ffmpeg(command, f"提取关键帧失败: {output_path}")
        except RuntimeError as exc:
            logger.warning(
                "FFmpeg 抽帧失败，回退 OpenCV 抽帧: {} | time={:.3f}s | reason={}",
                input_path,
                float(timestamp),
                str(exc),
            )
            self._extract_frame_with_opencv(input_path, timestamp, output_path)

    @staticmethod
    def _extract_frame_with_opencv(
        input_path: Path,
        timestamp: float,
        output_path: Path,
    ) -> None:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频文件: {input_path}")

        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if source_fps > 0:
                frame_index = max(0, int(round(float(timestamp) * source_fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)

            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(f"OpenCV 未读到有效帧: {input_path} @ {timestamp:.3f}s")

            suffix = output_path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                success = cv2.imwrite(
                    str(output_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
            elif suffix == ".png":
                success = cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            else:
                success = cv2.imwrite(str(output_path), frame)

            if not success:
                raise RuntimeError(f"OpenCV 写入关键帧失败: {output_path}")
        finally:
            cap.release()

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
    def _write_json_list(path: Path, data: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    @staticmethod
    def _safe_stem(stem: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
        return safe or "video"

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        return f"{max(0.0, float(seconds)):.3f}"

    @staticmethod
    def _json_path(path: Path) -> str:
        return path.as_posix()


if __name__ == "__main__":
    started_at = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    coarse_filter = VideoCoarseFilter(
        raw_dir=str(project_root / "data" / "raw"),
        interim_dir=str(project_root / "data" / "interim"),
        fps=5,
    )
    coarse_filter.run()
    elapsed = time.perf_counter() - started_at
    logger.info("阶段一独立执行完成 | 总耗时 {:.2f}s", elapsed)
