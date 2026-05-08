# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

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

BLOCK_MIN_DURATION = 8.0
BLOCK_MAX_DURATION = 12.0
VIDEO_MIN_DURATION = 28.0
VIDEO_MAX_DURATION = 32.0
BLOCK_MIN_CLIPS = 2
BLOCK_MAX_CLIPS = 5
BLOCKS_PER_VIDEO = 3
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30
DEFAULT_MAX_VIDEOS = 200
DEFAULT_SPEED = 1.0
SPEED_MAP = {
    "trial_error": 1.5,   # Fast-forward failed attempts.
    "success_step": 1.2,  # Slightly accelerate normal progress.
    "victory": 1.3,       # Keep victory highlights at source speed.
}

SUCCESS_ACTIONS = {"success_step"}
FAIL_ACTIONS = {"trial_error"}
VICTORY_ACTIONS = {"victory", "level_success", "final_success", "game_success", "win", "clear"}
NORMAL_ACTIONS = SUCCESS_ACTIONS | FAIL_ACTIONS


@dataclass(frozen=True)
class Clip:
    id: str
    level: str
    action_type: str
    original_video: str
    path: Path
    duration: float
    timestamp: float
    order: int
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class Block:
    id: str
    level: str
    kind: str
    clips: tuple[Clip, ...]
    start_index: int
    end_index: int
    start_timestamp: float
    end_timestamp: float
    source_duration: float
    playback_duration: float

    @property
    def short_level(self) -> str:
        return level_label(self.level)

    def clip_ids(self) -> list[str]:
        return [clip.id for clip in self.clips]


@dataclass(frozen=True)
class Recipe:
    name: str
    levels: tuple[str, str, str]


@dataclass(frozen=True)
class RenderSegment:
    clip: Clip
    speed: float

    @property
    def playback_duration(self) -> float:
        return self.clip.duration / self.speed


def load_assets(assets_json: str | Path = "data/global_assets.json") -> list[Clip]:
    path = resolve_path(assets_json)
    if not path.exists():
        raise FileNotFoundError(f"assets json not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("assets"), list):
        records = data["assets"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"assets json must be a list or contain an assets list: {path}")

    clips: list[Clip] = []
    for order, item in enumerate(records):
        if not isinstance(item, dict):
            continue

        raw_path = str(item.get("final_clip_path") or item.get("path") or "").strip()
        if not raw_path:
            continue

        clips.append(
            Clip(
                id=str(item.get("id") or f"asset_{order:05d}"),
                level=normalize_level(item.get("level")),
                action_type=normalize_action(item.get("action_type")),
                original_video=str(item.get("original_video") or item.get("source_video") or "unknown"),
                path=resolve_path(raw_path),
                duration=parse_duration(item.get("duration")),
                timestamp=parse_asset_timestamp(item, order),
                order=order,
                raw=dict(item),
            )
        )

    clips.sort(key=lambda clip: (natural_level_key(clip.level), clip.timestamp, clip.order, clip.id))
    logger.info("Loaded {} assets from {}", len(clips), path)
    return clips


def mine_blocks_for_level(clips: list[Clip], level: str) -> dict[str, list[Block]]:
    timeline = sorted(
        [clip for clip in clips if clip.level == level],
        key=lambda clip: (clip.timestamp, clip.order, clip.id),
    )
    pools: dict[str, list[Block]] = {"normal": [], "victory": []}

    for start in range(len(timeline)):
        source_duration = timeline[start].duration
        playback_duration = clip_playback_duration(timeline[start])
        for count in range(BLOCK_MIN_CLIPS, BLOCK_MAX_CLIPS + 1):
            end = start + count
            if end > len(timeline):
                break

            block_clips = timeline[start:end]
            new_clip = block_clips[-1]
            source_duration += new_clip.duration
            playback_duration += clip_playback_duration(new_clip)

            if playback_duration < BLOCK_MIN_DURATION:
                continue
            if playback_duration > BLOCK_MAX_DURATION:
                break

            kind = classify_block(block_clips)
            if kind is None:
                continue

            block = Block(
                id=f"{level_label(level)}_{kind}_{start:04d}_{end - 1:04d}",
                level=level,
                kind=kind,
                clips=tuple(block_clips),
                start_index=start,
                end_index=end - 1,
                start_timestamp=block_clips[0].timestamp,
                end_timestamp=block_clips[-1].timestamp,
                source_duration=source_duration,
                playback_duration=playback_duration,
            )
            pools[kind].append(block)

    for pool in pools.values():
        pool.sort(key=lambda block: (abs(block.playback_duration - 10.0), block.start_timestamp, block.start_index))

    logger.info(
        "Mined blocks | level={} | normal={} | victory={}",
        level,
        len(pools["normal"]),
        len(pools["victory"]),
    )
    return pools


def classify_block(clips: list[Clip]) -> str | None:
    if not clips:
        return None

    actions = [clip.action_type for clip in clips]
    last_action = actions[-1]

    if last_action in FAIL_ACTIONS:
        return None

    if last_action in VICTORY_ACTIONS:
        allowed_prefix = NORMAL_ACTIONS | VICTORY_ACTIONS
        return "victory" if all(action in allowed_prefix for action in actions) else None

    if all(action in NORMAL_ACTIONS for action in actions) and not any(action in VICTORY_ACTIONS for action in actions):
        return "normal"

    return None


def mine_all_blocks(clips: list[Clip], levels: Iterable[str]) -> dict[str, dict[str, list[Block]]]:
    return {level: mine_blocks_for_level(clips, level) for level in levels}


def build_recipes(levels: list[str]) -> list[Recipe]:
    if not levels:
        raise ValueError("No levels have enough valid blocks to build recipes.")

    recipes: list[Recipe] = []
    active_levels = levels[:3]

    for level in active_levels:
        recipes.append(Recipe(name=f"single_{level_label(level)}", levels=(level, level, level)))

    for first, second in itertools.combinations(active_levels, 2):
        recipes.append(Recipe(name=f"two_{level_label(first)}_{level_label(second)}_2_1", levels=(first, first, second)))
        recipes.append(Recipe(name=f"two_{level_label(first)}_{level_label(second)}_1_2", levels=(first, second, second)))

    if len(active_levels) >= 3:
        l1, l2, l3 = active_levels
        recipes.append(Recipe(name=f"three_{level_label(l1)}_{level_label(l2)}_{level_label(l3)}", levels=(l1, l2, l3)))

    return recipes


def iter_block_sequences(
    recipe: Recipe,
    ending_kind: str,
    block_pools: dict[str, dict[str, list[Block]]],
) -> Iterable[tuple[Block, Block, Block]]:
    pools_by_slot: list[list[Block]] = []
    for index, level in enumerate(recipe.levels):
        kind = ending_kind if index == BLOCKS_PER_VIDEO - 1 else "normal"
        pool = block_pools.get(level, {}).get(kind, [])
        if not pool:
            return
        pools_by_slot.append(pool)

    for blocks in iter_shuffled_block_product(pools_by_slot, f"{recipe.name}:{ending_kind}"):
        if blocks_are_temporally_valid(blocks) and blocks_match_target_duration(blocks):
            yield blocks


def blocks_match_target_duration(blocks: tuple[Block, ...]) -> bool:
    playback_duration = sum(block.playback_duration for block in blocks)
    return VIDEO_MIN_DURATION <= playback_duration <= VIDEO_MAX_DURATION


def iter_shuffled_block_product(pools_by_slot: list[list[Block]], seed_text: str) -> Iterable[tuple[Block, ...]]:
    lengths = [len(pool) for pool in pools_by_slot]
    total = math.prod(lengths)
    if total <= 0:
        return

    seed = stable_seed(seed_text)
    cursor = seed % total
    stride = coprime_stride(total, seed)

    for _ in range(total):
        indices = unravel_product_index(cursor, lengths)
        yield tuple(pool[index] for pool, index in zip(pools_by_slot, indices))
        cursor = (cursor + stride) % total


def unravel_product_index(flat_index: int, lengths: list[int]) -> list[int]:
    indices = [0] * len(lengths)
    for index in range(len(lengths) - 1, -1, -1):
        length = lengths[index]
        indices[index] = flat_index % length
        flat_index //= length
    return indices


def coprime_stride(total: int, seed: int) -> int:
    stride = max(1, total // 3 + seed % max(1, total // 5))
    while math.gcd(stride, total) != 1:
        stride += 1
    return stride


def stable_seed(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def blocks_are_temporally_valid(blocks: tuple[Block, ...]) -> bool:
    """Keep A/B/C ordered only within the same level timeline."""
    by_level: dict[str, list[Block]] = defaultdict(list)
    for block in blocks:
        by_level[block.level].append(block)

    for level_blocks in by_level.values():
        for previous, current in zip(level_blocks, level_blocks[1:]):
            if previous.start_timestamp >= current.start_timestamp:
                return False
            if previous.end_timestamp >= current.start_timestamp:
                return False
            if previous.end_index >= current.start_index:
                return False

    return True


def flatten_blocks(blocks: tuple[Block, ...]) -> list[RenderSegment]:
    return [RenderSegment(clip=clip, speed=clip_speed(clip)) for block in blocks for clip in block.clips]


def concat_segments_with_ffmpeg(segments: list[RenderSegment], output_path: Path) -> None:
    if not segments:
        raise ValueError("cannot render an empty segment list")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y"]
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    next_input_index = 0

    for segment_index, segment in enumerate(segments):
        clip = segment.clip
        if not clip.path.exists():
            raise FileNotFoundError(f"clip path does not exist: {clip.path}")

        video_input = next_input_index
        command.extend(["-i", str(clip.path)])
        next_input_index += 1

        setpts_factor = 1.0 / segment.speed
        filter_parts.append(
            f"[{video_input}:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={TARGET_FPS},format=yuv420p,setpts={setpts_factor:.8f}*PTS"
            f"[v{segment_index}]"
        )

        if probe_has_audio(clip.path):
            audio_label = f"{video_input}:a"
        else:
            audio_input = next_input_index
            command.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    f"{max(0.1, clip.duration):.3f}",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                ]
            )
            next_input_index += 1
            audio_label = f"{audio_input}:a"

        filter_parts.append(
            f"[{audio_label}]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"{atempo_filter(segment.speed)},aresample=async=1:first_pts=0"
            f"[a{segment_index}]"
        )
        concat_inputs.append(f"[v{segment_index}][a{segment_index}]")

    filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=1[vcat][acat]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vcat]",
            "-map",
            "[acat]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    run_ffmpeg(command, f"ffmpeg concat failed: {output_path}")


def generate_all_combinations(
    assets_json: str | Path = "data/global_assets.json",
    output_dir: str | Path = "data/processed",
    levels: list[str] | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
    max_videos: int = DEFAULT_MAX_VIDEOS,
) -> list[dict[str, Any]]:
    clips = load_assets(assets_json)
    usable_clips = [clip for clip in clips if clip.path.exists()]
    missing_count = len(clips) - len(usable_clips)
    if missing_count:
        logger.warning("Skipped {} clips because local video paths do not exist.", missing_count)
    if not usable_clips:
        raise RuntimeError("No usable clips found in global_assets.json.")

    candidate_levels = normalize_requested_levels(levels) if levels else all_asset_levels(usable_clips)
    block_pools = mine_all_blocks(usable_clips, candidate_levels)
    available_levels = available_levels_from_blocks(block_pools)
    selected_levels = available_levels[:3]
    recipes = build_recipes(selected_levels)
    log_data_level_mapping(available_levels, selected_levels)
    log_adaptive_recipe_mode(available_levels, selected_levels, recipes)

    batch_dir = resolve_output_root(output_dir) / "batch_block_combinations" / safe_name(run_id or timestamp())
    batch_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(batch_dir / "stage3.log"), rotation="10 MB", level="INFO", encoding="utf-8")

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    max_videos = max(0, int(max_videos))
    video_index = 0

    logger.info(
        "Stage3 deterministic block engine started | levels={} | recipes={} | output={} | dry_run={}",
        selected_levels,
        len(recipes),
        batch_dir,
        dry_run,
    )

    active_tasks: list[dict[str, Any]] = []
    for recipe in recipes:
        for ending_kind, ending_label in (("victory", "Victory"), ("normal", "Normal")):
            active_tasks.append(
                {
                    "recipe": recipe,
                    "ending_label": ending_label,
                    "iterator": iter(iter_block_sequences(recipe, ending_kind, block_pools)),
                    "local_index": 0,
                }
            )

    while active_tasks:
        next_tasks: list[dict[str, Any]] = []
        for task in active_tasks:
            if max_videos and video_index >= max_videos:
                return write_manifest_and_return(
                    batch_dir=batch_dir,
                    results=results,
                    failures=failures,
                    clips=usable_clips,
                    selected_levels=selected_levels,
                    block_pools=block_pools,
                    recipes=recipes,
                    dry_run=dry_run,
                    max_videos=max_videos,
                )

            try:
                blocks = next(task["iterator"])
            except StopIteration:
                continue

            next_tasks.append(task)
            recipe = task["recipe"]
            ending_label = task["ending_label"]
            video_index += 1
            task["local_index"] += 1
            local_index = task["local_index"]
            combo_name = "_".join(level_label(level) for level in recipe.levels)
            output_path = batch_dir / f"combo_{combo_name}_Ending_{ending_label}_{local_index:03d}.mp4"
            segments = flatten_blocks(blocks)
            record = build_manifest_record(
                index=video_index,
                recipe=recipe,
                ending=ending_label,
                local_index=local_index,
                blocks=blocks,
                segments=segments,
                output_path=output_path,
                dry_run=dry_run,
            )

            try:
                logger.info(
                    "[{}] Render plan | recipe={} | ending={} | clips={} | duration={:.3f}s | output={}",
                    video_index,
                    combo_name,
                    ending_label,
                    len(segments),
                    record["playback_duration"],
                    output_path,
                )
                if not dry_run:
                    concat_segments_with_ffmpeg(segments, output_path)
                results.append(record)
            except Exception as exc:
                logger.exception("Combination render failed: {}", exc)
                failures.append(f"{output_path.name}: {exc}")

        active_tasks = next_tasks

    return write_manifest_and_return(
        batch_dir=batch_dir,
        results=results,
        failures=failures,
        clips=usable_clips,
        selected_levels=selected_levels,
        block_pools=block_pools,
        recipes=recipes,
        dry_run=dry_run,
        max_videos=max_videos,
    )


def write_manifest_and_return(
    *,
    batch_dir: Path,
    results: list[dict[str, Any]],
    failures: list[str],
    clips: list[Clip],
    selected_levels: list[str],
    block_pools: dict[str, dict[str, list[Block]]],
    recipes: list[Recipe],
    dry_run: bool,
    max_videos: int,
) -> list[dict[str, Any]]:
    manifest = {
        "run_id": batch_dir.name,
        "engine": "deterministic_10s_block_combinations",
        "dry_run": dry_run,
        "speed_map": SPEED_MAP,
        "max_videos": max_videos,
        "source_clip_count": len(clips),
        "levels": selected_levels,
        "block_rules": {
            "clip_count": [BLOCK_MIN_CLIPS, BLOCK_MAX_CLIPS],
            "duration": [BLOCK_MIN_DURATION, BLOCK_MAX_DURATION],
            "blocks_per_video": BLOCKS_PER_VIDEO,
        },
        "block_counts": {
            level: {
                "normal": len(pools["normal"]),
                "victory": len(pools["victory"]),
            }
            for level, pools in block_pools.items()
        },
        "recipes": [
            {
                "name": recipe.name,
                "levels": list(recipe.levels),
            }
            for recipe in recipes
        ],
        "failures": failures,
        "count_generated": len(results),
        "videos": results,
    }
    manifest_path = batch_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (batch_dir / "content_index.txt").write_text(build_content_index(results), encoding="utf-8")
    logger.info("Stage3 deterministic engine finished | videos={} | manifest={}", len(results), manifest_path)
    return results


def build_manifest_record(
    *,
    index: int,
    recipe: Recipe,
    ending: str,
    local_index: int,
    blocks: tuple[Block, ...],
    segments: list[RenderSegment],
    output_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "index": index,
        "engine": "deterministic_10s_block_combinations",
        "recipe": recipe.name,
        "recipe_levels": list(recipe.levels),
        "ending": ending,
        "local_index": local_index,
        "output_path": json_path(output_path),
        "rendered": not dry_run,
        "source_duration": round(sum(segment.clip.duration for segment in segments), 3),
        "playback_duration": round(sum(segment.playback_duration for segment in segments), 3),
        "blocks": [
            {
                "id": block.id,
                "level": block.level,
                "kind": block.kind,
                "source_index_range": [block.start_index, block.end_index],
                "timestamp_range": [round(block.start_timestamp, 3), round(block.end_timestamp, 3)],
                "source_duration": round(block.source_duration, 3),
                "playback_duration": round(block.playback_duration, 3),
                "clip_ids": block.clip_ids(),
            }
            for block in blocks
        ],
        "clip_sequence": [
            {
                "id": segment.clip.id,
                "path": json_path(segment.clip.path),
                "level": segment.clip.level,
                "action_type": segment.clip.action_type,
                "original_duration": round(segment.clip.duration, 3),
                "speed": round(segment.speed, 3),
                "playback_duration": round(segment.playback_duration, 3),
                "timestamp": round(segment.clip.timestamp, 3),
            }
            for segment in segments
        ],
    }


def build_content_index(records: list[dict[str, Any]]) -> str:
    lines = ["Stage 3 Deterministic Block Combinations", ""]
    for item in records:
        lines.append(
            f"{item['index']:03d}. {item['recipe']} | Ending={item['ending']} | "
            f"{item['playback_duration']:.1f}s | {item['output_path']}"
        )
        for block in item["blocks"]:
            lines.append(
                f"    {block['id']} | {block['level']} {block['kind']} "
                f"{block['playback_duration']:.1f}s clips={len(block['clip_ids'])} "
                f"source_index={block['source_index_range'][0]}-{block['source_index_range'][1]} "
                f"timestamp={block['timestamp_range'][0]:.3f}-{block['timestamp_range'][1]:.3f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate_batch(
    count: int = 10,
    assets_json: str | Path = "data/global_assets.json",
    output_dir: str | Path = "data/processed",
    seed: int | None = None,
    dry_run: bool = False,
    max_gap_seconds: float = 30.0,
    run_id: str | None = None,
    templates: list[str] | None = None,
    levels: list[str] | None = None,
    speed: float = DEFAULT_SPEED,
    **_: Any,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for older callers.

    ``count`` now limits deterministic combinations. Set count <= 0 to render
    every valid combination.
    """
    if seed is not None:
        logger.info("Stage3 deterministic engine ignores random seed: {}", seed)
    if max_gap_seconds != 30.0:
        logger.info("Stage3 deterministic engine ignores legacy max_gap_seconds: {}", max_gap_seconds)
    if templates:
        logger.info("Stage3 deterministic engine ignores legacy template filters: {}", templates)
    if speed != DEFAULT_SPEED:
        logger.info("Stage3 deterministic engine ignores legacy fixed speed {}; using SPEED_MAP.", speed)

    return generate_all_combinations(
        assets_json=assets_json,
        output_dir=output_dir,
        levels=levels,
        dry_run=dry_run,
        run_id=run_id,
        max_videos=max(0, int(count)),
    )


class VideoAssembler:
    """Compatibility wrapper used by src/main.py."""

    def __init__(
        self,
        interim_dir: str = "data/interim",
        processed_dir: str = "data/processed",
        global_assets_path: str = "data/global_assets.json",
        batch_size: int = 200,
        target_duration: float = 30.0,
        max_gap_seconds: float = 30.0,
        recipe_names: list[str] | None = None,
        run_id: str | None = None,
        seed: int | None = None,
        levels: list[str] | None = None,
        speed: float = DEFAULT_SPEED,
        **_: Any,
    ) -> None:
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir
        self.global_assets_path = global_assets_path
        self.batch_size = batch_size
        self.target_duration = target_duration
        self.max_gap_seconds = max_gap_seconds
        self.recipe_names = recipe_names
        self.run_id = run_id
        self.seed = seed
        self.levels = levels
        self.speed = speed

    def run(self, only_video: str | None = None) -> list[dict[str, Any]]:
        if only_video:
            logger.info("Stage3 deterministic engine uses global_assets.json; only_video={} is ignored.", only_video)
        return generate_all_combinations(
            assets_json=self.global_assets_path,
            output_dir=self.processed_dir,
            levels=self.levels,
            dry_run=False,
            run_id=self.run_id,
            max_videos=max(0, int(self.batch_size)),
        )


VideoHighlightAssembler = VideoAssembler


def all_asset_levels(clips: list[Clip]) -> list[str]:
    levels = sorted({clip.level for clip in clips}, key=natural_level_key)
    if not levels:
        raise ValueError("No levels found in usable assets.")
    return levels


def normalize_requested_levels(levels: list[str]) -> list[str]:
    normalized = [normalize_level(level) for level in levels if str(level).strip()]
    if not normalized:
        raise ValueError("--levels must contain at least 1 level, for example: L1 or L1,L2")
    return sorted(dict.fromkeys(normalized), key=natural_level_key)


def available_levels_from_blocks(block_pools: dict[str, dict[str, list[Block]]]) -> list[str]:
    available: list[str] = []
    for level in sorted(block_pools, key=natural_level_key):
        pools = block_pools[level]
        if pools.get("normal") and pools.get("victory"):
            available.append(level)
    if not available:
        detail = {
            level: {
                "normal": len(pools.get("normal", [])),
                "victory": len(pools.get("victory", [])),
            }
            for level, pools in block_pools.items()
        }
        raise RuntimeError(f"No level has enough valid blocks to build recipes. Block counts: {detail}")
    return available


def log_data_level_mapping(available_levels: list[str], selected_levels: list[str]) -> None:
    logger.info(
        "[数据解析] 从 JSON 中动态扫描到 {} 个有效关卡：{}，已映射至配方引擎。参与配方关卡：{}",
        len(available_levels),
        available_levels,
        selected_levels,
    )


def log_adaptive_recipe_mode(available_levels: list[str], selected_levels: list[str], recipes: list[Recipe]) -> None:
    available_names = ", ".join(available_levels)
    selected_names = ", ".join(selected_levels)
    count = len(available_levels)
    if count == 1:
        suffix = "已自适应降级为单关卡配方生成模式"
    elif count == 2:
        suffix = "已自适应降级为单/双关卡混合配方生成模式"
    else:
        suffix = f"已启用完整单/双/三关卡配方生成模式，参与配方关卡为 {selected_names}"
    logger.info(
        "[配方引擎] 当前素材库可用关卡数量为 {} ({})，{}。配方数量={}",
        count,
        available_names,
        suffix,
        len(recipes),
    )


def clip_speed(clip: Clip) -> float:
    if clip.action_type in VICTORY_ACTIONS:
        return SPEED_MAP["victory"]
    return float(SPEED_MAP.get(clip.action_type, DEFAULT_SPEED))


def clip_playback_duration(clip: Clip) -> float:
    return clip.duration / clip_speed(clip)


def probe_has_audio(path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        logger.warning("ffprobe was not found; assuming clip has audio: {}", path)
        return True
    return "audio" in (result.stdout or "").lower()


def atempo_filter(speed: float) -> str:
    factors: list[float] = []
    remaining = max(0.01, float(speed))
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.8f}" for factor in factors)


def run_ffmpeg(command: list[str], error_prefix: str) -> None:
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
        raise RuntimeError("ffmpeg was not found. Please install it and add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() or "ffmpeg did not return stderr."
        quoted = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(f"{error_prefix}\nCOMMAND:\n{quoted}\nSTDERR:\n{stderr}") from exc


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return PROJECT_ROOT / path


def resolve_output_root(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_duration(value: Any) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = 0.0
    return max(0.1, duration)


def parse_asset_timestamp(item: dict[str, Any], fallback_order: int) -> float:
    for key in ("source_start_time", "original_start_time", "start_time", "timeline_start", "timestamp"):
        if key in item:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                pass

    parsed = parse_datetime(str(item.get("created_at") or ""))
    if parsed is not None:
        return parsed.timestamp()
    return float(fallback_order)


def parse_datetime(raw: str) -> datetime | None:
    if not raw:
        return None

    candidates = [raw, raw.replace("Z", "+00:00")]
    if re.search(r"[+-]\d{4}$", raw):
        candidates.append(f"{raw[:-5]}{raw[-5:-2]}:{raw[-2:]}")

    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def normalize_action(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_level(value: Any) -> str:
    return str(value or "Unknown_Level").strip() or "Unknown_Level"


def level_number(level: str) -> int:
    match = re.search(r"(\d+)", str(level))
    return int(match.group(1)) if match else 999999


def natural_level_key(level: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(level))
    key: list[tuple[int, Any]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key) or ((1, ""),)


def level_label(level: str) -> str:
    return safe_name(level)


def safe_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw)).strip("_") or "run"


def json_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_levels_arg(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3 deterministic 10-second block combination engine")
    parser.add_argument("--assets-json", "--global-assets-path", dest="assets_json", default="data/global_assets.json")
    parser.add_argument("--output-dir", "--processed-dir", dest="output_dir", default="data/processed")
    parser.add_argument("--levels", default=None, help="Comma-separated levels. Example: L1,L2,L3")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Deprecated; SPEED_MAP controls clip speed.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest without rendering videos.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--max-videos",
        "--count",
        "--batch-size",
        dest="max_videos",
        type=int,
        default=DEFAULT_MAX_VIDEOS,
        help="Limit rendered combinations. Use 0 for every valid combination.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    load_env_file(PROJECT_ROOT / ".env")
    if args.speed != DEFAULT_SPEED:
        logger.info("--speed={} is deprecated and ignored; SPEED_MAP controls clip speed.", args.speed)
    started = time.perf_counter()
    logger.info("=== Stage3 deterministic block engine started ===")
    outputs = generate_all_combinations(
        assets_json=args.assets_json,
        output_dir=args.output_dir,
        levels=parse_levels_arg(args.levels),
        dry_run=args.dry_run,
        run_id=args.run_id,
        max_videos=args.max_videos,
    )
    logger.info("=== Stage3 finished | outputs={} | elapsed={:.2f}s ===", len(outputs), time.perf_counter() - started)


if __name__ == "__main__":
    main()
