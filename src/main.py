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

# from stage2_fine import ...  # 后续补充：VLM 精筛
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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    started_at = time.perf_counter()
    logger.info("=== 水排序高光剪辑流水线启动 ===")
    logger.info("项目根目录: {}", project_root)

    # 步骤 1: 粗筛 (CV 帧差法)
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

    # 步骤 2: 精筛 (VLM API) - 占位
    # logger.info(">>> 开始阶段二：VLM 精筛")
    # ...

    # 步骤 3: 剪辑合成 - 占位
    # logger.info(">>> 开始阶段三：视频组装")
    # ...

    elapsed = time.perf_counter() - started_at
    logger.info("=== 全部流程执行完毕，耗时 {:.2f} 秒 ===", elapsed)


if __name__ == "__main__":
    main()
