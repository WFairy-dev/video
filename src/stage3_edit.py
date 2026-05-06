# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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

SUCCESS_ACTIONS = {"success_step"}
FAIL_ACTIONS = {"trial_error"}
VICTORY_ACTIONS = {"victory", "level_success", "final_success", "game_success", "win", "clear"}

TARGET_DURATION = 30.0
MIN_DURATION = 26.0
MAX_DURATION = 32.0
DEFAULT_BATCH_COUNT = 30
DEFAULT_MAX_GAP_SECONDS = 30.0


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
class Unit:
    unit_type: str
    clips: tuple[Clip, ...]
    level: str
    original_video: str
    duration: float
    timestamp: float
    id: str

    def clip_ids(self) -> list[str]:
        return [clip.id for clip in self.clips]

    def label(self) -> str:
        return f"{self.level}({self.unit_type})"


@dataclass
class GeneratedPlan:
    template: str
    units: list[Unit]
    total_duration: float
    score: float
    slug: str
    reason: str = ""

    def clip_ids(self) -> list[str]:
        return [clip.id for unit in self.units for clip in unit.clips]

    def unit_types(self) -> list[str]:
        return [unit.unit_type for unit in self.units]


@dataclass
class AssetIndex:
    clips: list[Clip]
    success_units: list[Unit]
    victory_units: list[Unit]
    fail_recovery_units: list[Unit]
    by_level: dict[str, list[Unit]]
    victories_by_level: dict[str, list[Unit]]
    recoveries_by_level: dict[str, list[Unit]]
    missing_paths: list[str]


def load_assets(assets_json: str | Path = "data/global_assets.json") -> list[Clip]:
    path = Path(assets_json)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
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
        raw_path = str(item.get("final_clip_path") or "").strip()
        if not raw_path:
            continue
        clip_path = resolve_path(raw_path)
        action_type = normalize_action(item.get("action_type"))
        clips.append(
            Clip(
                id=str(item.get("id") or f"asset_{order:05d}"),
                level=normalize_level(item.get("level")),
                action_type=action_type,
                original_video=str(item.get("original_video") or item.get("source_video") or "unknown"),
                path=clip_path,
                duration=parse_duration(item.get("duration")),
                timestamp=parse_asset_timestamp(item, order),
                order=order,
                raw=dict(item),
            )
        )
    return clips


def build_success_units(clips: list[Clip]) -> list[Unit]:
    return [make_unit("S", (clip,)) for clip in clips if clip.action_type in SUCCESS_ACTIONS]


def build_victory_units(clips: list[Clip]) -> list[Unit]:
    return [make_unit("V", (clip,)) for clip in clips if clip.action_type in VICTORY_ACTIONS]


def build_fail_recovery_units(
    clips: list[Clip],
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> list[Unit]:
    grouped: dict[tuple[str, str], list[Clip]] = defaultdict(list)
    for clip in clips:
        grouped[(clip.level, clip.original_video)].append(clip)

    units: list[Unit] = []
    for (_level, _video), timeline in grouped.items():
        timeline.sort(key=lambda clip: (clip.timestamp, clip.order, clip.id))
        for current, next_clip in zip(timeline, timeline[1:]):
            if current.action_type not in FAIL_ACTIONS:
                continue
            if next_clip.action_type not in SUCCESS_ACTIONS:
                continue
            gap = max(0.0, next_clip.timestamp - (current.timestamp + current.duration))
            if gap <= max_gap_seconds:
                units.append(make_unit("F_R", (current, next_clip)))
    return units


def build_asset_index(
    assets_json: str | Path = "data/global_assets.json",
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> AssetIndex:
    clips = load_assets(assets_json)
    missing_paths = [clip.path.as_posix() for clip in clips if not clip.path.exists()]
    usable_clips = [clip for clip in clips if clip.path.exists()]

    success_units = build_success_units(usable_clips)
    victory_units = build_victory_units(usable_clips)
    fail_recovery_units = build_fail_recovery_units(usable_clips, max_gap_seconds=max_gap_seconds)

    by_level: dict[str, list[Unit]] = defaultdict(list)
    victories_by_level: dict[str, list[Unit]] = defaultdict(list)
    recoveries_by_level: dict[str, list[Unit]] = defaultdict(list)
    for unit in success_units:
        by_level[unit.level].append(unit)
    for unit in victory_units:
        victories_by_level[unit.level].append(unit)
    for unit in fail_recovery_units:
        recoveries_by_level[unit.level].append(unit)

    for bucket in [by_level, victories_by_level, recoveries_by_level]:
        for units in bucket.values():
            units.sort(key=lambda unit: (unit.timestamp, unit.id))

    logger.info(
        "Stage3 assets loaded | clips={} usable={} S={} F_R={} V={} missing={}",
        len(clips),
        len(usable_clips),
        len(success_units),
        len(fail_recovery_units),
        len(victory_units),
        len(missing_paths),
    )
    return AssetIndex(
        clips=usable_clips,
        success_units=success_units,
        victory_units=victory_units,
        fail_recovery_units=fail_recovery_units,
        by_level=dict(by_level),
        victories_by_level=dict(victories_by_level),
        recoveries_by_level=dict(recoveries_by_level),
        missing_paths=missing_paths,
    )


def generate_template_a(index: AssetIndex, rng: random.Random) -> GeneratedPlan | None:
    levels = candidate_levels(index, need_same_level=True)
    rng.shuffle(levels)
    candidates: list[GeneratedPlan] = []
    for level in levels:
        s_pool = list(index.by_level.get(level, []))
        fr_pool = list(index.recoveries_by_level.get(level, []))
        v_pool = list(index.victories_by_level.get(level, []))
        if len(s_pool) < 2 or not fr_pool or not v_pool:
            continue
        for _ in range(12):
            fr = rng.choice(fr_pool)
            victory = choose_late_victory(v_pool, rng)
            units = pick_s_units(s_pool, 3, rng, exclude=clip_ids_of([fr, victory]))
            units = natural_sort_units(units) + [fr]
            units.extend(pick_s_units(s_pool, 1, rng, exclude=clip_ids_of(units + [victory])))
            units.append(victory)
            adjusted = adjust_duration(units, s_pool, rng, required_types={"F_R", "V"}, same_level=level)
            plan = make_plan("TemplateA", adjusted, slug=level, reason="Safe & Smooth")
            if score_plan(plan) is not None:
                candidates.append(plan)
    return best_plan(candidates)


def generate_template_b(index: AssetIndex, rng: random.Random) -> GeneratedPlan | None:
    levels = candidate_levels(index, need_same_level=True)
    rng.shuffle(levels)
    candidates: list[GeneratedPlan] = []
    for level in levels:
        s_pool = list(index.by_level.get(level, []))
        fr_pool = list(index.recoveries_by_level.get(level, []))
        v_pool = list(index.victories_by_level.get(level, []))
        if not s_pool or not fr_pool or not v_pool:
            continue
        for fr_count in (2, 1):
            if len(fr_pool) < fr_count:
                continue
            for _ in range(12):
                picked_fr = pick_units(fr_pool, fr_count, rng)
                victory = choose_late_victory(v_pool, rng)
                first_s = pick_s_units(s_pool, 1, rng, exclude=clip_ids_of(picked_fr + [victory]))
                middle_s = pick_s_units(s_pool, 2, rng, exclude=clip_ids_of(first_s + picked_fr + [victory]))
                if not first_s:
                    continue
                if fr_count == 2:
                    units = first_s + [picked_fr[0]] + middle_s + [picked_fr[1], victory]
                else:
                    units = first_s + [picked_fr[0]] + middle_s + [victory]
                adjusted = adjust_duration(units, s_pool, rng, required_types={"F_R", "V"}, same_level=level)
                plan = make_plan("TemplateB", adjusted, slug=level, reason="High Contrast")
                if score_plan(plan) is not None:
                    candidates.append(plan)
            if candidates:
                break
    return best_plan(candidates)


def generate_template_c(index: AssetIndex, rng: random.Random) -> GeneratedPlan | None:
    if not index.fail_recovery_units or not index.victory_units or not index.success_units:
        return None

    levels = sorted({unit.level for unit in index.success_units + index.fail_recovery_units + index.victory_units}, key=level_number)
    candidates: list[GeneratedPlan] = []
    for _ in range(40):
        fr = weighted_higher_level(index.fail_recovery_units, rng)
        victory_pool = sorted(index.victory_units, key=lambda unit: level_number(unit.level), reverse=True)
        victory = rng.choice(victory_pool[: max(1, min(3, len(victory_pool)))])

        early_levels = [level for level in levels if level_number(level) <= level_number(fr.level)]
        late_levels = [level for level in levels if level_number(level) >= level_number(fr.level)]
        rng.shuffle(early_levels)
        rng.shuffle(late_levels)

        used = clip_ids_of([fr, victory])
        before: list[Unit] = []
        for level in early_levels:
            choices = [unit for unit in index.by_level.get(level, []) if not clips_overlap(unit, used)]
            if choices:
                picked = rng.choice(choices)
                before.append(picked)
                used.update(picked.clip_ids())
            if len(before) >= 2:
                break

        after: list[Unit] = []
        for level in late_levels:
            choices = [unit for unit in index.by_level.get(level, []) if not clips_overlap(unit, used)]
            if choices:
                picked = rng.choice(choices)
                after.append(picked)
                used.update(picked.clip_ids())
                break

        units = natural_sort_units(before) + [fr] + natural_sort_units(after) + [victory]
        adjusted = adjust_duration(units, index.success_units, rng, required_types={"F_R", "V"})
        plan = make_plan("TemplateC", adjusted, slug="Mixed", reason="Multi-Level Mix")
        if score_plan(plan) is not None:
            candidates.append(plan)
    return best_plan(candidates)


def adjust_duration(
    units: list[Unit],
    success_pool: list[Unit],
    rng: random.Random,
    required_types: set[str],
    same_level: str | None = None,
) -> list[Unit]:
    adjusted = list(units)

    while total_duration(adjusted) > MAX_DURATION:
        removable = [
            idx
            for idx, unit in enumerate(adjusted)
            if unit.unit_type == "S" and unit.unit_type not in required_types
        ]
        if not removable:
            break
        adjusted.pop(removable[-1])

    attempts = 0
    while total_duration(adjusted) < MIN_DURATION and attempts < 30:
        attempts += 1
        used = clip_ids_of(adjusted)
        pool = [
            unit
            for unit in success_pool
            if not clips_overlap(unit, used)
            and (same_level is None or unit.level == same_level)
            and total_duration(adjusted) + unit.duration <= MAX_DURATION
        ]
        if not pool:
            break
        insert_at = max(0, len(adjusted) - 1)
        adjusted.insert(insert_at, rng.choice(pool))

    if adjusted and adjusted[-1].unit_type != "V":
        victories = [idx for idx, unit in enumerate(adjusted) if unit.unit_type == "V"]
        if victories:
            adjusted.append(adjusted.pop(victories[-1]))
    return adjusted


def score_plan(plan: GeneratedPlan) -> float | None:
    if not plan.units:
        return None
    if has_duplicate_clips(plan.units):
        return None
    if plan.total_duration > MAX_DURATION:
        return None
    if not any(unit.unit_type == "F_R" for unit in plan.units):
        return None
    if plan.units[-1].unit_type != "V":
        return None

    score = 100.0
    score -= abs(TARGET_DURATION - plan.total_duration) * 4.0
    if MIN_DURATION <= plan.total_duration <= MAX_DURATION:
        score += 20.0
    score += 25.0
    score += 25.0
    if plan.template == "TemplateC":
        levels = [level_number(unit.level) for unit in plan.units]
        score += sum(1 for left, right in zip(levels, levels[1:]) if right >= left) * 3.0
        score += len(set(unit.level for unit in plan.units)) * 2.5
    score -= repeated_source_penalty(plan.units)
    plan.score = score
    return score


def concat_with_ffmpeg(clips: list[Clip], output_path: Path) -> None:
    if not clips:
        raise ValueError("cannot concat an empty clip list")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y"]
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    next_input_index = 0

    for clip_index, clip in enumerate(clips):
        if not clip.path.exists():
            raise FileNotFoundError(f"clip path does not exist: {clip.path}")

        video_input = next_input_index
        command.extend(["-i", str(clip.path)])
        next_input_index += 1

        filter_parts.append(
            f"[{video_input}:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
            "setsar=1,fps=30,format=yuv420p"
            f"[v{clip_index}]"
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
            "aresample=async=1:first_pts=0"
            f"[a{clip_index}]"
        )
        concat_inputs.append(f"[v{clip_index}][a{clip_index}]")

    filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(clips)}:v=1:a=1[v][a]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
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
            "160k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )

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
        raise RuntimeError(f"ffmpeg concat failed: {output_path}\nCOMMAND:\n{quoted}\nSTDERR:\n{stderr}") from exc


def generate_batch(
    count: int = DEFAULT_BATCH_COUNT,
    assets_json: str | Path = "data/global_assets.json",
    output_dir: str | Path = "data/processed",
    seed: int | None = None,
    dry_run: bool = False,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    run_id: str | None = None,
    templates: list[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    index = build_asset_index(assets_json, max_gap_seconds=max_gap_seconds)
    diagnose_index(index)

    generator_map: dict[str, Callable[[AssetIndex, random.Random], GeneratedPlan | None]] = {
        "A": generate_template_a,
        "B": generate_template_b,
        "C": generate_template_c,
    }
    enabled = normalize_templates(templates)
    generators = [(name, generator_map[name]) for name in enabled if name in generator_map]

    root = Path(output_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    batch_dir = root / "batch_story_blocks" / safe_name(run_id or timestamp())
    batch_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for batch_index in range(1, int(count) + 1):
        names = list(generators)
        rng.shuffle(names)
        plan: GeneratedPlan | None = None
        tried: list[str] = []
        for template_name, generator in names:
            tried.append(template_name)
            plan = generator(index, rng)
            if plan is not None and score_plan(plan) is not None:
                break
            plan = None

        if plan is None:
            reason = f"video {batch_index:02d}: no valid plan after templates {tried}"
            logger.warning(reason)
            failures.append(reason)
            continue

        output_path = batch_dir / make_output_name(batch_index, plan)
        record = plan_to_manifest_record(batch_index, plan, output_path)
        logger.info(
            "[{}/{}] {} | {:.1f}s | {} | {}",
            batch_index,
            count,
            plan.template,
            plan.total_duration,
            " -> ".join(plan.unit_types()),
            output_path,
        )

        if not dry_run:
            try:
                concat_with_ffmpeg(flatten_clips(plan.units), output_path)
            except Exception as exc:
                logger.exception("video {} render failed: {}", batch_index, exc)
                record["render_error"] = str(exc)
                failures.append(f"video {batch_index:02d}: {exc}")
                results.append(record)
                continue

        results.append(record)

    manifest = {
        "run_id": batch_dir.name,
        "dry_run": dry_run,
        "count_requested": count,
        "count_planned": len(results),
        "failures": failures,
        "unit_summary": {
            "S": len(index.success_units),
            "F_R": len(index.fail_recovery_units),
            "V": len(index.victory_units),
            "missing_paths": len(index.missing_paths),
        },
        "videos": results,
    }
    manifest_path = batch_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (batch_dir / "content_index.txt").write_text(build_content_index(results), encoding="utf-8")
    logger.info("Stage3 batch finished | output_dir={} | videos={} | manifest={}", batch_dir, len(results), manifest_path)
    if not results:
        shortage = manifest["unit_summary"]
        raise RuntimeError(
            "Stage3 could not generate any valid video plan. "
            f"Available units: S={shortage['S']}, F_R={shortage['F_R']}, V={shortage['V']}. "
            "Check that global_assets.json contains success_step clips, victory clips, "
            "and trial_error clips immediately followed by success_step in the same level/original_video."
        )
    return results


class VideoAssembler:
    """Compatibility wrapper used by src/main.py."""

    def __init__(
        self,
        interim_dir: str = "data/interim",
        processed_dir: str = "data/processed",
        global_assets_path: str = "data/global_assets.json",
        batch_size: int = DEFAULT_BATCH_COUNT,
        target_duration: float = TARGET_DURATION,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
        recipe_names: list[str] | None = None,
        run_id: str | None = None,
        seed: int | None = None,
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

    def run(self, only_video: str | None = None) -> list[dict[str, Any]]:
        if only_video:
            logger.info("Stage3 uses global_assets.json; only_video={} is ignored by the template engine.", only_video)
        return generate_batch(
            count=self.batch_size,
            assets_json=self.global_assets_path,
            output_dir=self.processed_dir,
            seed=self.seed,
            dry_run=False,
            max_gap_seconds=self.max_gap_seconds,
            run_id=self.run_id,
            templates=self.recipe_names,
        )


VideoHighlightAssembler = VideoAssembler


def make_unit(unit_type: str, clips: tuple[Clip, ...]) -> Unit:
    first = clips[0]
    duration = sum(clip.duration for clip in clips)
    unit_id = f"{unit_type}_{'_'.join(clip.id[:8] for clip in clips)}"
    return Unit(
        unit_type=unit_type,
        clips=clips,
        level=first.level,
        original_video=first.original_video,
        duration=duration,
        timestamp=first.timestamp,
        id=unit_id,
    )


def make_plan(template: str, units: list[Unit], slug: str, reason: str) -> GeneratedPlan:
    plan = GeneratedPlan(
        template=template,
        units=list(units),
        total_duration=total_duration(units),
        score=0.0,
        slug=slug,
        reason=reason,
    )
    score_plan(plan)
    return plan


def best_plan(plans: list[GeneratedPlan]) -> GeneratedPlan | None:
    valid = [plan for plan in plans if score_plan(plan) is not None]
    if not valid:
        return None
    return max(valid, key=lambda plan: plan.score)


def candidate_levels(index: AssetIndex, need_same_level: bool) -> list[str]:
    if not need_same_level:
        return sorted(index.by_level.keys(), key=level_number)
    levels = []
    for level in index.by_level:
        if index.by_level.get(level) and index.recoveries_by_level.get(level) and index.victories_by_level.get(level):
            levels.append(level)
    return sorted(levels, key=level_number)


def pick_units(pool: list[Unit], count: int, rng: random.Random, exclude: set[str] | None = None) -> list[Unit]:
    exclude = set(exclude or set())
    candidates = [unit for unit in pool if not clips_overlap(unit, exclude)]
    rng.shuffle(candidates)
    picked: list[Unit] = []
    used = set(exclude)
    for unit in candidates:
        if clips_overlap(unit, used):
            continue
        picked.append(unit)
        used.update(unit.clip_ids())
        if len(picked) >= count:
            break
    return picked


def pick_s_units(pool: list[Unit], count: int, rng: random.Random, exclude: set[str] | None = None) -> list[Unit]:
    picked = pick_units(pool, count, rng, exclude=exclude)
    return natural_sort_units(picked)


def choose_late_victory(pool: list[Unit], rng: random.Random) -> Unit:
    sorted_pool = sorted(pool, key=lambda unit: (unit.timestamp, unit.id), reverse=True)
    return rng.choice(sorted_pool[: max(1, min(3, len(sorted_pool)))])


def weighted_higher_level(pool: list[Unit], rng: random.Random) -> Unit:
    sorted_pool = sorted(pool, key=lambda unit: level_number(unit.level), reverse=True)
    return rng.choice(sorted_pool[: max(1, min(5, len(sorted_pool)))])


def natural_sort_units(units: list[Unit]) -> list[Unit]:
    return sorted(units, key=lambda unit: (level_number(unit.level), unit.timestamp, unit.id))


def flatten_clips(units: list[Unit]) -> list[Clip]:
    return [clip for unit in units for clip in unit.clips]


def clip_ids_of(items: list[Unit]) -> set[str]:
    return {clip.id for unit in items for clip in unit.clips}


def clips_overlap(unit: Unit, used: set[str]) -> bool:
    return any(clip.id in used for clip in unit.clips)


def has_duplicate_clips(units: list[Unit]) -> bool:
    ids = [clip.id for unit in units for clip in unit.clips]
    return len(ids) != len(set(ids))


def total_duration(units: list[Unit]) -> float:
    return sum(unit.duration for unit in units)


def repeated_source_penalty(units: list[Unit]) -> float:
    sources = [unit.original_video for unit in units]
    return max(0, len(sources) - len(set(sources))) * 2.0


def plan_to_manifest_record(batch_index: int, plan: GeneratedPlan, output_path: Path) -> dict[str, Any]:
    return {
        "index": batch_index,
        "template": plan.template,
        "reason": plan.reason,
        "output_path": json_path(output_path),
        "total_duration": round(plan.total_duration, 3),
        "score": round(plan.score, 3),
        "unit_sequence": plan.unit_types(),
        "units": [
            {
                "unit_type": unit.unit_type,
                "unit_id": unit.id,
                "level": unit.level,
                "original_video": unit.original_video,
                "duration": round(unit.duration, 3),
                "clips": [clip_to_record(clip) for clip in unit.clips],
            }
            for unit in plan.units
        ],
    }


def clip_to_record(clip: Clip) -> dict[str, Any]:
    return {
        "id": clip.id,
        "path": json_path(clip.path),
        "level": clip.level,
        "action_type": clip.action_type,
        "original_video": clip.original_video,
        "duration": round(clip.duration, 3),
    }


def build_content_index(records: list[dict[str, Any]]) -> str:
    lines = ["Stage 3 Emotional Rhythm Batch", ""]
    for item in records:
        units = " -> ".join(item["unit_sequence"])
        lines.append(
            f"{item['index']:02d}. {item['template']} | {item['total_duration']:.1f}s | {units} | {item['output_path']}"
        )
        for unit in item["units"]:
            clip_line = ", ".join(f"{clip['action_type']}:{Path(clip['path']).name}" for clip in unit["clips"])
            lines.append(f"    {unit['unit_type']} {unit['level']} {unit['original_video']} | {clip_line}")
        if item.get("render_error"):
            lines.append(f"    ERROR: {item['render_error']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def make_output_name(index: int, plan: GeneratedPlan) -> str:
    level_slug = plan.slug
    if plan.template in {"TemplateA", "TemplateB"}:
        level_slug = plan.units[-1].level if plan.units else plan.slug
    return f"video_{index:02d}_{plan.template}_{safe_name(level_slug)}.mp4"


def diagnose_index(index: AssetIndex) -> None:
    missing = []
    if not index.success_units:
        missing.append("S(success_step)")
    if not index.fail_recovery_units:
        missing.append("F_R(trial_error followed by adjacent success_step in same level/original_video)")
    if not index.victory_units:
        missing.append("V(victory/level_success/final_success/game_success)")
    if missing:
        logger.warning("Stage3 unit shortage: missing {}", ", ".join(missing))
    if index.missing_paths:
        logger.warning("Stage3 skipped {} assets because final_clip_path does not exist.", len(index.missing_paths))


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


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
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
    text = str(value or "Unknown_Level").strip() or "Unknown_Level"
    match = re.search(r"(\d+)", text)
    if match:
        return f"Level_{int(match.group(1))}"
    return safe_name(text)


def level_number(level: str) -> int:
    match = re.search(r"(\d+)", level)
    return int(match.group(1)) if match else 999999


def normalize_templates(templates: list[str] | None) -> list[str]:
    if not templates:
        return ["A", "B", "C"]
    aliases = {
        "A": "A",
        "B": "B",
        "C": "C",
        "TEMPLATEA": "A",
        "TEMPLATEB": "B",
        "TEMPLATEC": "C",
    }
    normalized: list[str] = []
    for item in templates:
        key = str(item).strip().upper()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unknown template {item!r}; expected A, B, or C")
        normalized.append(aliases[key])
    return normalized or ["A", "B", "C"]


def safe_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw)).strip("_") or "run"


def json_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_template_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3 emotional rhythm template editing engine")
    parser.add_argument("--assets-json", "--global-assets-path", dest="assets_json", default="data/global_assets.json")
    parser.add_argument("--output-dir", "--processed-dir", dest="output_dir", default="data/processed")
    parser.add_argument("--count", "--batch-size", dest="count", type=int, default=DEFAULT_BATCH_COUNT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-gap-seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--templates", "--recipes", dest="templates", default=None)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    started = time.perf_counter()
    logger.info("=== Stage3 emotional rhythm engine started ===")
    outputs = generate_batch(
        count=args.count,
        assets_json=args.assets_json,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run=args.dry_run,
        max_gap_seconds=args.max_gap_seconds,
        run_id=args.run_id,
        templates=parse_template_arg(args.templates),
    )
    logger.info("=== Stage3 finished | outputs={} | elapsed={:.2f}s ===", len(outputs), time.perf_counter() - started)


if __name__ == "__main__":
    main()
