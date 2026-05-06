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
    """阶段一粗筛：MOG2 运动信号提取 + 1D 滑动窗口 + NMS 去重。"""

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    RESIZE_WIDTH = 640
    TOP_CROP_RATIO = 0.15
    BOTTOM_CROP_RATIO = 0.15
    WINDOW_SECONDS = 4
    ENERGY_THRESHOLD = 20000
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
        self.window_seconds = self.WINDOW_SECONDS
        self.energy_threshold = (
            float(energy_threshold)
            if energy_threshold is not None
            else float(self.ENERGY_THRESHOLD)
        )
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
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
        """使用 MOG2 + 形态学开运算，提取一维运动能量信号。"""
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

    def _select_segments_with_sliding_window(
        self,
        energy_signal: list[int],
        timestamps: list[float],
        analysis_fps: float,
        duration: float,
    ) -> tuple[list[tuple[float, float]], float]:
        """在一维信号上执行窗口打分，并通过 NMS 保证动作片段不重叠。"""
        if not energy_signal or not timestamps or analysis_fps <= 0:
            return [], 0.0

        signal_array = np.asarray(energy_signal, dtype=np.float64)
        signal_peak = float(signal_array.max()) if signal_array.size else 0.0

        window_frames = max(1, int(round(self.window_seconds * analysis_fps)))
        if signal_array.size < window_frames:
            logger.info(
                "信号帧数不足一个窗口: 信号长度 {} < 窗口长度 {}，跳过切片。",
                signal_array.size,
                window_frames,
            )
            return [], signal_peak

        prefix_sum = np.zeros(signal_array.size + 1, dtype=np.float64)
        prefix_sum[1:] = np.cumsum(signal_array)
        window_energies = prefix_sum[window_frames:] - prefix_sum[:-window_frames]
        window_peak = float(window_energies.max()) if window_energies.size else 0.0

        candidate_indexes = np.where(window_energies >= self.energy_threshold)[0]
        if candidate_indexes.size == 0:
            logger.info(
                "无候选窗口超过阈值: threshold {:.1f} | 信号峰值 {:.1f} | 窗口峰值 {:.1f}",
                self.energy_threshold,
                signal_peak,
                window_peak,
            )
            return [], signal_peak

        sorted_candidates = sorted(
            (
                (int(index), float(window_energies[index]))
                for index in candidate_indexes.tolist()
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        selected: list[tuple[int, float]] = []
        for index, energy in sorted_candidates:
            start_time = float(timestamps[index])
            should_keep = True
            for kept_index, _ in selected:
                kept_start = float(timestamps[kept_index])
                if abs(start_time - kept_start) < self.window_seconds:
                    should_keep = False
                    break
            if should_keep:
                selected.append((index, energy))

        selected.sort(key=lambda item: timestamps[item[0]])
        segments: list[tuple[float, float]] = []
        for index, _ in selected:
            start_time = max(0.0, float(timestamps[index]))
            end_time = start_time + self.window_seconds
            if duration > 0:
                end_time = min(duration, end_time)
            if end_time <= start_time:
                continue
            segments.append((start_time, end_time))

        logger.info(
            "窗口筛选完成: 阈值 {:.1f} | 信号峰值 {:.1f} | 窗口峰值 {:.1f} | 候选 {} | NMS后 {}",
            self.energy_threshold,
            signal_peak,
            window_peak,
            len(sorted_candidates),
            len(segments),
        )
        return segments, signal_peak

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
        """把单个源视频筛出的片段写入所属素材集合目录。"""
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
            mid_time = start_time + (self.window_seconds / 2.0)
            if end_time > start_time:
                mid_time = min(mid_time, end_time)

            logger.info(
                "FFmpeg 进度 [{}/{}] 截取片段 {} ({:.3f}s ~ {:.3f}s)",
                index,
                total,
                clip_path.name,
                start_time,
                end_time,
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

    def run(self) -> list[dict[str, Any]]:
        """执行阶段一：MOG2 提取、滑窗筛选、FFmpeg 导出。"""
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
            logger.info("开始处理视频: {}", video_file)
            duration = self._get_video_duration(video_file)
            signal_started = time.perf_counter()
            energy_signal, timestamps, analysis_fps, _ = self._extract_motion_signal(video_file)
            signal_elapsed = time.perf_counter() - signal_started

            segments, signal_peak = self._select_segments_with_sliding_window(
                energy_signal=energy_signal,
                timestamps=timestamps,
                analysis_fps=analysis_fps,
                duration=duration,
            )
            logger.info(
                "视频分析完成: {} | 分析耗时 {:.2f}s | 信号最大波峰 {:.1f} | 最终片段数 {}",
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

    def _iter_video_files(self) -> list[Path]:
        """根据输入路径返回待处理视频列表。"""
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
        """读取视频总时长，用于裁剪片段边界。"""
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
        """创建输出目录并清理当前素材集合的历史产物。"""
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
        """覆盖写入当前素材集合的 segments.json。"""
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
        """使用固定 4 秒窗口参数执行 ffmpeg 切片。"""
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
        """提取中间时刻关键帧，用于后续精筛。"""
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
        """FFmpeg 抽帧失败时，使用 OpenCV 兜底抽帧。"""
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
                raise RuntimeError(f"OpenCV 未读取到有效帧: {input_path} @ {timestamp:.3f}s")

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
        """执行 ffmpeg 命令，统一处理异常信息。"""
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
        """原子化写入 JSON，避免中断导致文件损坏。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    @staticmethod
    def _safe_stem(stem: str) -> str:
        """将视频文件名清洗为安全目录名。"""
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
