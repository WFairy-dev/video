import argparse
import os
import time
from pathlib import Path

from loguru import logger

try:
    from stage1_coarse import VideoCoarseFilter
    from stage3_edit import VideoHighlightAssembler
    from stage2_fine import VideoFineFilter, load_env_file
except ModuleNotFoundError as exc:
    if exc.name not in {"stage1_coarse", "stage2_fine", "stage3_edit"}:
        raise
    from .stage1_coarse import VideoCoarseFilter
    from .stage3_edit import VideoHighlightAssembler
    from .stage2_fine import VideoFineFilter, load_env_file


def main():
    """项目全局调度入口，串联水排序高光剪辑流水线的各个阶段。"""
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
        help="相对于 --raw-dir 的子目录，例如 level2 或 batch_a/clips",
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
        help="阶段三成品输出目录（默认: data/processed）",
    )
    parser.add_argument("--fps", type=int, default=5, help="粗筛分析帧率（默认: 5）")
    parser.add_argument(
        "--stage2-model",
        default=None,
        help="阶段二 OpenRouter 视觉模型（默认读取 OPENROUTER_MODEL 或使用 openai/gpt-4o-mini）",
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
        help="跳过阶段一，仅对已有 data/interim/clips/*/segments.json 执行阶段二",
    )
    parser.add_argument(
        "--only-stage3",
        action="store_true",
        help="跳过阶段一和阶段二，仅对已有 scored_segments.json 执行阶段三",
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="执行阶段一/二后不生成成品视频",
    )
    parser.add_argument(
        "--stage3-seed",
        type=int,
        default=None,
        help="阶段三盲盒策略随机种子（默认: 不固定）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    load_env_file(project_root / ".env")

    started_at = time.perf_counter()
    logger.info("=== 水排序高光剪辑流水线启动 ===")
    logger.info("项目根目录: {}", project_root)

    target_video_names: list[str] | None = _infer_target_video_names(args)

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
        logger.info("<<< 阶段一完成！数据已按视频保存至 data/interim/clips/{video_name}/segments.json")

    # 步骤 2: 精筛 (VLM API)
    if args.only_stage3:
        logger.info(">>> 已跳过阶段二：直接使用现有 scored_segments.json")
    else:
        logger.info(">>> 开始阶段二：VLM 语义过滤")
        fine_filter = VideoFineFilter(
            interim_dir=args.interim_dir,
            model=args.stage2_model,
            api_key=args.api_key,
            base_url=args.openrouter_base_url,
        )
        scored_segments = []
        for video_name in _iter_target_video_names(target_video_names):
            scored_segments.extend(fine_filter.run(only_video=video_name))
        logger.info("<<< 阶段二完成！共判定 {} 条片段，结果写入 scored_segments.json", len(scored_segments))

    # 步骤 3: 剪辑合成 (FFmpeg concat 流拷贝)
    if args.skip_stage3 or args.only_stage2:
        logger.info(">>> 已跳过阶段三：不生成成品视频")
    else:
        logger.info(">>> 开始阶段三：视频组装")
        assembler = VideoHighlightAssembler(
            interim_dir=args.interim_dir,
            processed_dir=args.processed_dir,
            seed=args.stage3_seed,
        )
        outputs = []
        for video_name in _iter_target_video_names(target_video_names):
            outputs.extend(assembler.run(only_video=video_name))
        logger.info("<<< 阶段三完成！共生成 {} 个高光成品", len(outputs))

    elapsed = time.perf_counter() - started_at
    logger.info("=== 全部流程执行完毕，耗时 {:.2f} 秒 ===", elapsed)


def _infer_target_video_names(args: argparse.Namespace) -> list[str] | None:
    """从命令参数推断用户明确限定的视频名；None 表示不限定。"""
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


if __name__ == "__main__":
    main()
