from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def collect_level3_clips(level3_clips_dir: Path) -> list[Path]:
    """仅收集 level3 目录下的粗筛片段，避免混入其他视频。"""
    if not level3_clips_dir.exists():
        raise FileNotFoundError(f"目录不存在: {level3_clips_dir}")

    # 只匹配 level3 的片段命名，确保不会把 level3 等其他视频拼进去。
    clips = sorted(level3_clips_dir.glob("level3_clip_*.mp4"))
    if not clips:
        raise FileNotFoundError(f"未找到可拼接片段: {level3_clips_dir / 'level3_clip_*.mp4'}")
    return clips


def concat_videos_ffmpeg(clips: list[Path], output_path: Path) -> None:
    """使用 ffmpeg concat demuxer 拼接片段。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        concat_list_path = Path(f.name)
        for clip in clips:
            # ffmpeg concat 列表格式：file 'absolute_path'
            f.write(f"file '{clip.as_posix()}'\n")

    try:
        # 第一优先：直接流拷贝，速度最快。
        fast_cmd = [
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
        result = subprocess.run(
            fast_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            print(f"[OK] 拼接完成（流拷贝）: {output_path}")
            return

        # 若编码参数不完全一致，回退到重编码，保证一定能得到可预览结果。
        print("[WARN] 流拷贝失败，自动回退到重编码模式。")
        slow_cmd = [
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
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
        subprocess.run(slow_cmd, check=True)
        print(f"[OK] 拼接完成（重编码）: {output_path}")
    finally:
        concat_list_path.unlink(missing_ok=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    level3_clips_dir = project_root / "data" / "interim" / "clips" / "level3"
    output_path = level3_clips_dir / "level3_merged_preview.mp4"

    clips = collect_level3_clips(level3_clips_dir)
    print(f"[INFO] 仅拼接 level3 片段，共 {len(clips)} 个。")
    concat_videos_ffmpeg(clips, output_path)


if __name__ == "__main__":
    main()
