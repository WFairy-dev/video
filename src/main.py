# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from loguru import logger

try:
    from stage1_coarse import VideoCoarseFilter
except ModuleNotFoundError as exc:
    if exc.name != "stage1_coarse":
        raise
    from .stage1_coarse import VideoCoarseFilter


def main() -> None:
    """项目全局调度入口，串联水排序高光剪辑流水线的四个阶段。"""
    parser = argparse.ArgumentParser(description="水排序高光剪辑流水线")
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="原始视频目录或单个视频文件路径（默认: data/raw）",
    )
    parser.add_argument(
        "--raw-subdir",
        default=None,
        metavar="REL_PATH",
        help="相对 --raw-dir 的子目录，例如 level2 或 batch_a/clips",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="递归扫描目录下所有子文件夹中的视频",
    )
    parser.add_argument(
        "--interim-dir",
        default="data/interim",
        help="中间产物目录（默认: data/interim）",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="阶段三/四成品输出目录（默认: data/processed）",
    )
    parser.add_argument("--fps", type=int, default=5, help="粗筛分析帧率（默认: 5）")
    parser.add_argument(
        "--stage2-model",
        default=None,
        help="阶段二 OpenRouter 视觉模型（默认读取 OPENROUTER_MODEL）",
    )
    parser.add_argument(
        "--stage4-model",
        default=None,
        help="阶段四音乐模型（默认: google/lyria-3-clip-preview）",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API Key（默认读取 .env / OPENROUTER_API_KEY）",
    )
    parser.add_argument(
        "--openrouter-base-url",
        default=None,
        help="OpenRouter API Base URL（默认读取 .env / OPENROUTER_BASE_URL）",
    )
    parser.add_argument(
        "--only-stage2",
        action="store_true",
        help="跳过阶段一，仅对已有 segments.json 执行阶段二",
    )
    parser.add_argument(
        "--only-stage3",
        action="store_true",
        help="跳过阶段一和阶段二，仅对已有 scored_segments.json 执行阶段三",
    )
    parser.add_argument(
        "--only-stage4",
        action="store_true",
        help="跳过阶段一/二/三，仅对已有 data/processed/{video_name}/v*.mp4 执行阶段四",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="执行阶段一后停止，不调用阶段二、阶段三和阶段四",
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="执行阶段一/二后不生成成品视频，也不执行阶段四",
    )
    parser.add_argument(
        "--skip-stage4",
        action="store_true",
        help="执行到阶段三后停止，不生成 BGM 和 final 混音视频",
    )
    parser.add_argument(
        "--stage3-seed",
        type=int,
        default=None,
        help="阶段三盲盒策略随机种子（默认: 不固定）",
    )
    parser.add_argument(
        "--video-volume",
        type=float,
        default=1.0,
        help="阶段四原视频音轨音量权重（默认: 1.0）",
    )
    parser.add_argument(
        "--bgm-volume",
        type=float,
        default=0.6,
        help="阶段四 BGM 音量权重（默认: 0.6）",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="本次运行输出目录名（默认: 当前时间 YYYYMMDD_HHMMSS）；only-stage4 时可指定要处理的运行目录",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    _load_env_file(project_root / ".env")

    started_at = time.perf_counter()
    logger.info("=== 水排序高光剪辑流水线启动 ===")
    logger.info("项目根目录: {}", project_root)

    if not args.run_id and not args.only_stage4:
        args.run_id = time.strftime("%Y%m%d_%H%M%S")

    target_video_names: list[str] | None = _infer_target_video_names(args)

    if args.only_stage4:
        logger.info(">>> 仅执行阶段四：使用现有阶段三成品生成 BGM 并混音")
        stage4_outputs = _run_stage4(args, target_video_names)
        elapsed = time.perf_counter() - started_at
        logger.info("=== 阶段四执行完毕：输出 {} 个 final 视频，耗时 {:.2f}s ===", len(stage4_outputs), elapsed)
        return

    # 步骤 1: 粗筛 (CV 帧差法)
    if args.only_stage3:
        logger.info(">>> 已跳过阶段一：直接进入阶段三")
    elif args.only_stage2:
        logger.info(">>> 已跳过阶段一：直接使用现有 segments.json")
    else:
        logger.info(">>> 开始阶段一：视频粗筛")
        if args.raw_subdir:
            logger.info(
                "素材路径: {} / {} | 递归扫描: {}",
                args.raw_dir,
                args.raw_subdir,
                args.recursive,
            )
        else:
            logger.info("素材路径: {} | 递归扫描: {}", args.raw_dir, args.recursive)

        coarse_filter = VideoCoarseFilter(
            raw_dir=args.raw_dir,
            interim_dir=args.interim_dir,
            fps=args.fps,
            raw_subdir=args.raw_subdir,
            recursive=args.recursive,
        )
        coarse_records = coarse_filter.run()
        processed_video_names = _source_video_names(coarse_records)
        if processed_video_names:
            target_video_names = processed_video_names
        logger.info("<<< 阶段一完成：结果已写入 data/interim/clips/{video_name}/segments.json")

    # 步骤 2: 精筛 (VLM API)
    if args.only_stage3:
        logger.info(">>> 已跳过阶段二：直接使用现有 scored_segments.json")
    elif args.skip_stage2:
        logger.info(">>> 已跳过阶段二：仅保留阶段一粗筛结果")
    else:
        logger.info(">>> 开始阶段二：VLM 语义过滤")
        VideoFineFilter = _get_video_fine_filter()
        fine_filter = VideoFineFilter(
            interim_dir=args.interim_dir,
            model=args.stage2_model,
            api_key=args.api_key,
            base_url=args.openrouter_base_url,
        )
        scored_segments = []
        for video_name in _iter_target_video_names(target_video_names):
            scored_segments.extend(fine_filter.run(only_video=video_name))
        logger.info("<<< 阶段二完成：共判定 {} 条片段，结果写入 scored_segments.json", len(scored_segments))

    # 步骤 3: 组装 (FFmpeg concat + 变速)
    stage3_outputs = []
    if args.skip_stage3 or args.only_stage2 or args.skip_stage2:
        logger.info(">>> 已跳过阶段三：不生成成品视频")
    else:
        logger.info(">>> 开始阶段三：视频组装")
        VideoHighlightAssembler = _get_video_highlight_assembler()
        assembler = VideoHighlightAssembler(
            interim_dir=args.interim_dir,
            processed_dir=args.processed_dir,
            seed=args.stage3_seed,
            run_id=args.run_id,
        )
        for video_name in _iter_target_video_names(target_video_names):
            stage3_outputs.extend(assembler.run(only_video=video_name))
        logger.info("<<< 阶段三完成：共生成 {} 个无 BGM 高光成品", len(stage3_outputs))

    # 步骤 4: BGM 生成与混音
    if args.skip_stage4 or args.only_stage2 or args.only_stage3 or args.skip_stage2 or args.skip_stage3:
        logger.info(">>> 已跳过阶段四：不生成 BGM 和 final 混音视频")
    else:
        logger.info(">>> 开始阶段四：BGM 生成与混音")
        stage4_outputs = _run_stage4(args, target_video_names)
        logger.info("<<< 阶段四完成：共生成 {} 个 final 视频", len(stage4_outputs))

    elapsed = time.perf_counter() - started_at
    logger.info("=== 全部流程执行完毕，耗时 {:.2f} 秒 ===", elapsed)


def _run_stage4(args: argparse.Namespace, target_video_names: list[str] | None) -> list[dict[str, str]]:
    VideoMusicMixer = _get_video_music_mixer()
    mixer = VideoMusicMixer(
        processed_dir=args.processed_dir,
        video_volume=args.video_volume,
        bgm_volume=args.bgm_volume,
        model=args.stage4_model,
        api_key=args.api_key,
        base_url=args.openrouter_base_url,
        run_id=args.run_id,
    )
    outputs: list[dict[str, str]] = []
    for video_name in _iter_target_video_names(target_video_names):
        outputs.extend(mixer.run(only_video=video_name))
    return outputs


def _infer_target_video_names(args: argparse.Namespace) -> list[str] | None:
    """从命令行参数推断用户明确限定的视频名；None 表示不限定。"""
    if args.raw_subdir:
        return [Path(args.raw_subdir).name]

    raw_path = Path(args.raw_dir)
    if raw_path.is_file():
        return [raw_path.stem]

    return None


def _source_video_names(records: list[dict]) -> list[str]:
    """从阶段一产物提取本次实际处理的视频名。"""
    return sorted(
        {
            str(record.get("source_video", "")).strip()
            for record in records
            if str(record.get("source_video", "")).strip()
        }
    )


def _iter_target_video_names(video_names: list[str] | None) -> list[str | None]:
    """统一遍历目标视频名；None 表示让下游处理全部。"""
    if not video_names:
        return [None]
    return video_names


def _load_env_file(env_path: Path) -> None:
    """轻量读取 .env，避免只跑阶段一时提前导入阶段二/四依赖。"""
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


def _get_video_fine_filter():
    try:
        from stage2_fine import VideoFineFilter
    except ModuleNotFoundError as exc:
        if exc.name != "stage2_fine":
            raise
        from .stage2_fine import VideoFineFilter
    return VideoFineFilter


def _get_video_highlight_assembler():
    try:
        from stage3_edit import VideoHighlightAssembler
    except ModuleNotFoundError as exc:
        if exc.name != "stage3_edit":
            raise
        from .stage3_edit import VideoHighlightAssembler
    return VideoHighlightAssembler


def _get_video_music_mixer():
    try:
        from stage4_music import VideoMusicMixer
    except ModuleNotFoundError as exc:
        if exc.name != "stage4_music":
            raise
        from .stage4_music import VideoMusicMixer
    return VideoMusicMixer


if __name__ == "__main__":
    main()
