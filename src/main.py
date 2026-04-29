import argparse
import os
import time
from pathlib import Path

from loguru import logger

try:
    from stage1_coarse import VideoCoarseFilter
    from stage2_fine import VideoFineFilter, load_env_file
except ModuleNotFoundError as exc:
    if exc.name not in {"stage1_coarse", "stage2_fine"}:
        raise
    from .stage1_coarse import VideoCoarseFilter
    from .stage2_fine import VideoFineFilter, load_env_file

# from stage3_edit import ...  # 后续补充：视频组装


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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    load_env_file(project_root / ".env")

    started_at = time.perf_counter()
    logger.info("=== 水排序高光剪辑流水线启动 ===")
    logger.info("项目根目录: {}", project_root)

    # 步骤 1: 粗筛 (CV 帧差法)
    if args.only_stage2:
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
        coarse_filter.run()
        logger.info("<<< 阶段一完成！数据已按视频保存至 data/interim/clips/{video_name}/segments.json")

    # 步骤 2: 精筛 (VLM API)
    logger.info(">>> 开始阶段二：VLM 语义过滤")
    fine_filter = VideoFineFilter(
        interim_dir=args.interim_dir,
        model=args.stage2_model,
        api_key=args.api_key,
        base_url=args.openrouter_base_url,
    )
    scored_segments = fine_filter.run()
    logger.info("<<< 阶段二完成！共判定 {} 条片段，结果写入 scored_segments.json", len(scored_segments))

    # 步骤 3: 剪辑合成 - 占位
    # logger.info(">>> 开始阶段三：视频组装")
    # ...

    elapsed = time.perf_counter() - started_at
    logger.info("=== 全部流程执行完毕，耗时 {:.2f} 秒 ===", elapsed)


if __name__ == "__main__":
    main()
