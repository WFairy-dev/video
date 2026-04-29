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


# 项目根目录固定为 water_sort_highlights/，无论从哪里执行脚本，都能把日志写到根目录 logs/。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Loguru 默认保留控制台输出；这里额外增加文件日志，记录 DEBUG 级别细节，便于回溯长视频处理过程。
logger.add(
    str(LOGS_DIR / "stage1_{time:YYYYMMDD_HHmmss}.log"),
    rotation="10 MB",
    level="DEBUG",
    encoding="utf-8",
)


class VideoCoarseFilter:
    """阶段一：使用 OpenCV 快速帧差法粗筛倒水动作，并用原生 ffmpeg 导出片段。"""

    # raw_dir 是目录时，只扫描这些常见视频格式。
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    # 帧差前先把画面宽度降到 320 像素，这是阶段一提速的关键。
    RESIZE_WIDTH = 320

    # 上下固定 UI 容易闪烁，参与帧差会造成误判，因此裁掉顶部 15% 和底部 15%。
    TOP_CROP_RATIO = 0.15
    BOTTOM_CROP_RATIO = 0.15

    # 每个动作片段前后各保留 1 秒，避免倒水动作开头或结尾被切掉。
    PADDING_SECONDS = 1.0

    # Active 状态下，短暂停顿不足 0.5 秒时不断开动作片段。
    STILL_GRACE_SECONDS = 0.5

    # 灰度差超过该阈值的像素才会被视为“变化像素”。
    DIFF_THRESHOLD = 25

    # ROI 中变化像素占比达到该值时，认为当前采样帧处于运动状态。
    MOTION_RATIO_THRESHOLD = 0.003

    # 过滤极短误触发片段，减少压缩噪声、UI 抖动等带来的脏数据。
    MIN_SEGMENT_SECONDS = 0.3

    def __init__(
        self,
        raw_dir: str,
        interim_dir: str,
        fps: int = 5,
        *,
        raw_subdir: str | None = None,
        recursive: bool = False,
    ):
        """初始化粗筛器。

        参数:
            raw_dir: 原始视频路径，可以是单个视频文件，也可以是视频目录。
            interim_dir: 中间产物目录，内部只管理 clips/ 和 frames/。
            fps: 目标分析帧率，默认约 5 FPS。
            raw_subdir: 可选 raw 子目录，例如只处理 data/raw/batch_a。
            recursive: 为 True 时递归扫描 raw 路径下所有子目录。
        """
        self.raw_path = Path(raw_dir)
        if raw_subdir:
            if self.raw_path.is_file():
                logger.warning("raw_subdir 已忽略：raw_dir 指向单个文件 {}", self.raw_path)
            else:
                self.raw_path = self.raw_path / Path(raw_subdir)

        self.recursive = recursive
        self.interim_dir = Path(interim_dir)
        self.clips_dir = self.interim_dir / "clips"
        self.frames_dir = self.interim_dir / "frames"
        self.target_fps = max(1, int(fps))

        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """把原始帧处理成适合帧差计算的低维灰度图。

        顺序固定为：
        1. 先将整帧等比缩放到 320 像素宽，降低后续计算量。
        2. 再裁掉顶部 15% 和底部 15%，只保留中间瓶子区域。
        3. 转为灰度图，减少颜色差异对运动判断的干扰。
        4. 使用高斯模糊去噪，降低录屏压缩噪声和边缘抖动影响。
        """
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

        resized_height = resized.shape[0]
        top = int(resized_height * self.TOP_CROP_RATIO)
        bottom = int(resized_height * (1.0 - self.BOTTOM_CROP_RATIO))
        roi = resized[top:bottom, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _detect_motion_segments(self, video_path: str) -> list[tuple[float, float]]:
        """检测单个视频中的动作时间段。

        速度优化策略：
        - 通过源 FPS 和 target_fps 计算 frame_skip。
        - 循环中先用 cap.grab() 快速推进视频流。
        - 只有命中采样点时才 cap.retrieve() 取出图像并执行帧差。
        - 取出的帧先降维到 320 宽，再进行裁剪、灰度、模糊、帧差。
        """
        started_at = time.perf_counter()
        video_file = Path(video_path)
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {video_file}")

        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            if source_fps <= 0:
                # 少数录屏可能缺失 FPS 元数据，使用 30 FPS 作为保守回退。
                source_fps = 30.0

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_count / source_fps if frame_count > 0 else 0.0
            frame_skip = max(1, int(round(source_fps / self.target_fps)))
            analysis_interval = frame_skip / source_fps

            logger.info(
                "开始侦测: {} | 源 FPS {:.2f} | 目标 FPS {} | frame_skip {}",
                video_file.name,
                source_fps,
                self.target_fps,
                frame_skip,
            )

            previous_frame: np.ndarray | None = None
            segments: list[tuple[float, float]] = []
            state = "Idle"
            active_start: float | None = None
            last_motion_time: float | None = None
            last_sample_time = 0.0
            analyzed_frames = 0
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

                timestamp = frame_index / source_fps
                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
                if pos_msec > 0:
                    timestamp = pos_msec / 1000.0

                last_sample_time = timestamp
                analyzed_frames += 1
                current_frame = self._preprocess_frame(frame)

                if previous_frame is not None:
                    diff = cv2.absdiff(previous_frame, current_frame)
                    _, mask = cv2.threshold(
                        diff,
                        self.DIFF_THRESHOLD,
                        255,
                        cv2.THRESH_BINARY,
                    )

                    changed_pixels = cv2.countNonZero(mask)
                    motion_ratio = changed_pixels / float(mask.size)
                    is_motion = motion_ratio >= self.MOTION_RATIO_THRESHOLD

                    if state == "Idle":
                        if is_motion:
                            state = "Active"
                            active_start = max(0.0, timestamp - analysis_interval)
                            last_motion_time = timestamp
                    else:
                        if is_motion:
                            last_motion_time = timestamp
                        else:
                            reference_time = last_motion_time or timestamp
                            still_seconds = timestamp - reference_time
                            if still_seconds >= self.STILL_GRACE_SECONDS:
                                segment_end = reference_time + analysis_interval
                                if duration > 0:
                                    segment_end = min(duration, segment_end)

                                if (
                                    active_start is not None
                                    and segment_end - active_start >= self.MIN_SEGMENT_SECONDS
                                ):
                                    segments.append((active_start, segment_end))

                                state = "Idle"
                                active_start = None
                                last_motion_time = None

                previous_frame = current_frame
                frame_index += 1

            if state == "Active" and active_start is not None:
                segment_end = (last_motion_time or last_sample_time) + analysis_interval
                if duration > 0:
                    segment_end = min(duration, segment_end)
                if segment_end - active_start >= self.MIN_SEGMENT_SECONDS:
                    segments.append((active_start, segment_end))

            elapsed = time.perf_counter() - started_at
            logger.info(
                "侦测完成: {} | 分析帧 {} | 发现片段 {} | 耗时 {:.2f}s",
                video_file.name,
                analyzed_frames,
                len(segments),
                elapsed,
            )
            return segments
        finally:
            cap.release()

    def _extract_and_save(self, video_path: str, segments: list) -> list[dict[str, Any]]:
        """使用 ffmpeg 导出短视频和关键帧，并覆盖当前视频的局部 segments.json。"""
        started_at = time.perf_counter()
        video_file = Path(video_path)
        video_name = self._safe_stem(video_file.stem)
        clip_output_dir = self.clips_dir / video_name
        frame_output_dir = self.frames_dir / video_name
        self._prepare_video_output_dirs(video_name, clip_output_dir, frame_output_dir)

        if not segments:
            self._write_video_segments_json(clip_output_dir, [])
            logger.info("未发现可导出的片段，已覆盖局部清单: {}", video_file.name)
            return []

        duration = self._get_video_duration(video_file)
        padded_segments = self._apply_padding_and_merge(segments, duration)
        if not padded_segments:
            self._write_video_segments_json(clip_output_dir, [])
            logger.info("Padding 后无有效片段，已覆盖局部清单: {}", video_file.name)
            return []

        total = len(padded_segments)
        records: list[dict[str, Any]] = []

        logger.info("开始生成切片: {} | 待导出 {} 个片段", video_file.name, total)

        for index, (start, end) in enumerate(padded_segments, start=1):
            start = max(0.0, float(start))
            end = min(float(end), duration) if duration > 0 else float(end)
            if end <= start:
                continue

            segment_id = f"{video_name}_{index:03d}"
            clip_path = clip_output_dir / f"{video_name}_clip_{index:03d}.mp4"
            frame_path = frame_output_dir / f"{video_name}_frame_{index:03d}.jpg"
            mid_time = (start + end) / 2.0

            logger.debug("[{}/{}] 正在提取 {}", index, total, clip_path)
            self._run_ffmpeg_clip(video_file, start, end, clip_path)

            logger.debug("[{}/{}] 正在提取关键帧 {}", index, total, frame_path)
            self._run_ffmpeg_frame(video_file, mid_time, frame_path)

            records.append(
                {
                    "id": segment_id,
                    "source_video": video_name,
                    "source_name": video_name,
                    "source_path": self._json_path(video_file),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "clip_path": self._json_path(clip_path),
                    "frame_path": self._json_path(frame_path),
                }
            )

        self._write_video_segments_json(clip_output_dir, records)

        elapsed = time.perf_counter() - started_at
        logger.info(
            "切片生成完成: {} | 成功导出 {} 个片段 | 局部清单 {} | 耗时 {:.2f}s",
            video_file.name,
            len(records),
            clip_output_dir / "segments.json",
            elapsed,
        )
        return records

    def run(self) -> list[dict[str, Any]]:
        """执行阶段一粗筛。

        本阶段不再维护跨视频总清单。每个视频的元数据只写入：
        data/interim/clips/{video_name}/segments.json。

        返回值只是本次进程内的汇总列表，供调用方临时查看；不会被写成全局 JSON。
        """
        started_at = time.perf_counter()
        video_files = self._iter_video_files()
        run_records: list[dict[str, Any]] = []

        if not video_files:
            logger.warning("未找到可处理的视频文件: {}", self.raw_path)
            return run_records

        logger.info("阶段一粗筛启动 | 待处理视频数: {}", len(video_files))

        for video_file in video_files:
            logger.info("开始处理视频: {}", video_file)
            detect_started_at = time.perf_counter()
            segments = self._detect_motion_segments(str(video_file))
            detect_elapsed = time.perf_counter() - detect_started_at
            logger.info(
                "视频侦测统计: {} | 发现 {} 个候选片段 | 耗时 {:.2f}s",
                video_file.name,
                len(segments),
                detect_elapsed,
            )

            records = self._extract_and_save(str(video_file), segments)
            run_records.extend(records)

        elapsed = time.perf_counter() - started_at
        logger.info(
            "阶段一粗筛完成 | 本次运行总片段数 {} | 数据已按视频隔离保存 | 总耗时 {:.2f}s",
            len(run_records),
            elapsed,
        )
        return run_records

    def _iter_video_files(self) -> list[Path]:
        """根据 raw_path 返回待处理视频列表。"""
        if self.raw_path.is_file():
            if self.raw_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                return [self.raw_path]
            logger.warning("输入文件不是支持的视频格式: {}", self.raw_path)
            return []

        if not self.raw_path.exists():
            logger.warning("原始视频路径不存在: {}", self.raw_path)
            return []

        def _is_video(path: Path) -> bool:
            return path.is_file() and path.suffix.lower() in self.VIDEO_EXTENSIONS

        if self.recursive:
            return sorted(path for path in self.raw_path.rglob("*") if _is_video(path))

        return sorted(path for path in self.raw_path.iterdir() if _is_video(path))

    def _get_video_duration(self, video_path: Path) -> float:
        """读取视频时长，用于 padding 后的边界裁剪。"""
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

    def _apply_padding_and_merge(
        self,
        segments: list[tuple[float, float]],
        duration: float,
    ) -> list[tuple[float, float]]:
        """给片段添加 padding，并合并 padding 后重叠或相接的片段。"""
        padded_segments: list[tuple[float, float]] = []

        for raw_start, raw_end in sorted(segments, key=lambda item: item[0]):
            start = max(0.0, float(raw_start) - self.PADDING_SECONDS)
            end = float(raw_end) + self.PADDING_SECONDS
            if duration > 0:
                end = min(duration, end)

            if end <= start:
                continue

            if padded_segments and start <= padded_segments[-1][1]:
                previous_start, previous_end = padded_segments[-1]
                padded_segments[-1] = (previous_start, max(previous_end, end))
            else:
                padded_segments.append((start, end))

        return padded_segments

    def _prepare_video_output_dirs(
        self,
        video_name: str,
        clip_output_dir: Path,
        frame_output_dir: Path,
    ) -> None:
        """创建当前视频输出目录，并清理本模块生成的旧同名产物。"""
        clip_output_dir.mkdir(parents=True, exist_ok=True)
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        for old_clip in clip_output_dir.glob(f"{video_name}_clip_*.mp4"):
            old_clip.unlink()
        for old_frame in frame_output_dir.glob(f"{video_name}_frame_*.jpg"):
            old_frame.unlink()

    def _write_video_segments_json(
        self,
        clip_output_dir: Path,
        records: list[dict[str, Any]],
    ) -> None:
        """将当前视频的片段清单覆盖写入 clips/{video_name}/segments.json。"""
        segments_path = clip_output_dir / "segments.json"
        self._write_json_list(segments_path, records)
        logger.debug("局部 segments.json 已保存: {}", segments_path)

    def _run_ffmpeg_clip(
        self,
        input_path: Path,
        start: float,
        end: float,
        output_path: Path,
    ) -> None:
        """调用 ffmpeg 截取视频片段。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            self._format_seconds(start),
            "-to",
            self._format_seconds(end),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            str(output_path),
        ]
        self._run_ffmpeg(command, f"截取视频片段失败: {output_path}")

    def _run_ffmpeg_frame(
        self,
        input_path: Path,
        timestamp: float,
        output_path: Path,
    ) -> None:
        """调用 ffmpeg 提取片段中间关键帧。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            self._format_seconds(timestamp),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
        self._run_ffmpeg(command, f"提取关键帧失败: {output_path}")

    @staticmethod
    def _run_ffmpeg(command: list[str], error_prefix: str) -> None:
        """执行 ffmpeg 命令，并将错误输出整理成可读异常。"""
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
            raise RuntimeError("未找到 ffmpeg，请确认 Conda 环境已安装 ffmpeg 并已激活。") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            detail = stderr if stderr else "ffmpeg 未返回详细错误。"
            raise RuntimeError(f"{error_prefix}\n{detail}") from exc

    @staticmethod
    def _write_json_list(path: Path, data: list[dict[str, Any]]) -> None:
        """原子化写入 JSON 列表，降低中途中断导致文件损坏的概率。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """把秒数格式化为 ffmpeg 接受的小数秒字符串。"""
        return f"{max(0.0, float(seconds)):.3f}"

    @staticmethod
    def _safe_stem(stem: str) -> str:
        """把源视频文件名转换为安全的目录名和文件名前缀。"""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
        return safe or "video"

    @staticmethod
    def _json_path(path: Path) -> str:
        """统一 JSON 中的路径格式，Windows 下也使用正斜杠，便于后续阶段读取。"""
        return path.as_posix()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    started_at = time.perf_counter()

    coarse_filter = VideoCoarseFilter(
        raw_dir=str(project_root / "data" / "raw"),
        interim_dir=str(project_root / "data" / "interim"),
        fps=5,
    )
    coarse_filter.run()

    elapsed = time.perf_counter() - started_at
    logger.info("阶段一独立运行结束 | 总耗时 {:.2f}s", elapsed)
