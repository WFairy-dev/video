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
from typing import Any

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

TARGET_DURATION = 30.0
MIN_DIRECTOR_DURATION = 28.0
MAX_DIRECTOR_DURATION = 32.0
DEFAULT_BATCH_COUNT = 10
DEFAULT_CANDIDATE_MIN = 30
DEFAULT_CANDIDATE_MAX = 50
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_BASE_URL = "https://apirouter.zhiqiteai.cn/ApiRouterServ/v1"
SUCCESS_ACTIONS = {"success_step"}
FAIL_ACTIONS = {"trial_error"}
VICTORY_ACTIONS = {"victory", "level_success", "final_success", "game_success", "win", "clear"}

# DIRECTOR_SYSTEM_PROMPT = """你是一个顶级的短视频游戏剪辑导演。你的任务是从我提供的【素材库 JSON】中，挑选素材并编排一个总时长在 **28 到 32 秒**之间的剪辑剧本。

# 【剪辑流派规则（请随机选择以下一种风格进行编排）】：
# 1. **多关卡平分秋色**：
# - 如果有 3 个关卡，前 10s 纯放 Level 1，中间 10s 放 Level 2，最后 10s 放 Level 3。
# - 如果有2个关卡，前15秒放Level 2，后面15s 放 Level 3。
# - 每个关卡内部必须按照 `timestamp` 升序连贯拼接，且结尾以最高关卡的 `victory` 压轴。

# 2. **单关卡深度解剖**：
# - 如果素材多为一个关卡，可采用“成功+试错+成功”交替的倒水展示，每个关卡内部必须按照 `timestamp` 升序连贯拼接，可以采用以下方法：
#     (1)按照时间顺序进进行拼接，视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
#     (2)按照时间顺序进进行拼接，且结尾以该关卡 `victory` 的结尾，视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
#     (3)按照时间倒序从后往前选取片段，且结尾以该关卡 `victory` 的结尾，拼接视频是从选取的片段正序拼接。视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
#     (4) 选取中间片段进行拼接，且结尾以该关卡 `victory` 的结尾，拼接视频是从选取的片段正序拼接。视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
#     (5) 选取中间片段进行拼接，拼接视频是从选取的片段正序拼接。视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
#     (6)随机选取片段，至少要有三个片段连接，保证一定的连贯性。不要只选择一个片段就跳跃到另一个片段里面。
# - 以上方法，前五个方法都要实现，第六个方法随机生成视频。

# 【微观连贯性规则】：
# - `trial_error` (试错) 后面必须紧跟**同一关卡、时间戳相近**的 `success_step` (成功)。

# 【时长与倍速规则 (Speed Control)】：
# - 实际播放时长 = `duration / speed`。
# - 你可以为每个片段指定 `speed` (建议范围 0.8 到 2.0)。
# - 遇到连续的多个 `success_step`，可以将倍速调高至 1.5x 或 2.0x 制造爽感。
# - 比如：victory画面1.0x ， 倒水片段1.0到1.5x之间，思考等待片段2.0 ，然后trial_error片段是1.2x
# - 你必须计算总时长，确保所有选出片段的 `(duration / speed)` 之和极其接近 30 秒。

# 【强制输出格式】：
# 只输出一个严格的 JSON 数组，包含你选中的片段 ID、排序和设定的倍速：
# [
#   {"id": "clip_012", "speed": 1.0, "reason": "开场试错制造悬念"},
#   {"id": "clip_013", "speed": 1.5, "reason": "紧跟正确操作，加速制造爽感"}
# ]"""

DIRECTOR_SYSTEM_PROMPT = """你是一个顶级的短视频游戏剪辑总导演。你的任务是从我提供的【素材库 JSON】中，挑选素材并编排一个总时长在 **28 到 32 秒**之间的剪辑剧本。

【核心架构】
你选出的所有素材，在物理时间上必须严格划分为 3 个时间区块：
- 区块 A：连续的一组操作，时长凑够约 10 秒（加上倍速后）。
- 区块 B：连续的一组操作，时长凑够约 10 秒（加上倍速后）。
- 区块 C：连续的一组操作，时长凑够约 10 秒（加上倍速后）。

【微观法则：区块内绝对连贯】
在任何一个区块内部挑选的多个片段，必须满足：
1. **同场景**：属于同一个 Level。
2. **时间连续**：它们在原视频中的 `timestamp` 必须是紧紧挨着的。
3. **试错闭环**：如果选了 `trial_error` (试错) 片段，紧跟着的下一个必须是该场景的 `success_step` (纠正成功)。

【宏观法则：时间单向流逝】
- 通一个level中，区块 A -> 区块 B -> 区块 C 的全局时间轴（`timestamp`）必须严格从早到晚推进，绝对禁止时间倒流或穿插！

【关卡剪辑】
1. **多关卡平分秋色**：
- 如果有 3 个关卡，前 10s 纯放 Level 1，中间 10s 放 Level 2，最后 10s 放 Level 3。
- 如果有2个关卡，前15秒放Level 2，后面15s 放 Level 3。
- 3 个区块必须跨越至少 2 个以上的不同 Level（例如：区块 A 是 Level_1，区块 B 是 Level_2，区块 C 是 Level_3）。
- 每个关卡内部必须按照 `timestamp` 升序连贯拼接，且结尾以最高关卡的 `victory` 压轴。

2. **单关卡深度解剖**：
- 如果素材多为一个关卡，可采用“成功+试错+成功”交替的倒水展示，每个关卡内部必须按照 `timestamp` 升序连贯拼接，可以采用以下方法：
    (1)按照时间顺序进进行拼接，视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
    (2)按照时间顺序进进行拼接，且结尾以该关卡 `victory` 的结尾，视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
    (3)按照时间倒序从后往前选取片段，且结尾以该关卡 `victory` 的结尾，拼接视频是从选取的片段正序拼接。视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
    (4) 选取中间片段进行拼接，且结尾以该关卡 `victory` 的结尾，拼接视频是从选取的片段正序拼接。视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
    (5) 选取中间片段进行拼接，拼接视频是从选取的片段正序拼接。视频时间累计30s左右，超过则放弃最后一个视频，或者将最后一个视频压缩开倍速。
    (6)随机选取片段，至少要有三个片段连接，保证一定的连贯性。不要只选择一个片段就跳跃到另一个片段里面。
- 以上方法，前五个方法都要实现，第六个方法随机生成视频。区块之间允许有时间跳跃（比如跳过无聊部分），但必须保持 A < B < C 的时间递进。

【时长与倍速规则 (Speed Control)】：
- 实际播放时长 = `duration / speed`。
- 你可以为每个片段指定 `speed` (建议范围 0.8 到 2.0，以填满对应的 10 秒区块)。
- 推荐倍速节奏：`victory` 画面 1.0x；连续倒水的顺畅片段可开 1.0x ；停顿思考的无聊画面可开 2.0x 快速跳过；`trial_error` 试错片段可设 1.0x。
- 确保所有选出片段的 `(duration / speed)` 之和极其接近 30 秒！

【强制输出格式】：
只输出一个严格的 JSON 数组，包含你选中的片段 ID、排序和设定的倍速：
[
  {"id": "clip_012", "speed": 1.0, "reason": "开场试错制造悬念"},
  {"id": "clip_013", "speed": 1.5, "reason": "紧跟正确操作，加速制造爽感"}
]

"""

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

    def menu_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "action_type": self.action_type,
            "duration": round(self.duration, 3),
            "timestamp": round(self.timestamp, 3),
        }


@dataclass(frozen=True)
class DirectedSegment:
    clip: Clip
    speed: float
    reason: str

    @property
    def playback_duration(self) -> float:
        return self.clip.duration / self.speed


@dataclass
class DirectedPlan:
    index: int
    style_hint: str
    segments: list[DirectedSegment]
    raw_response: str
    candidate_count: int
    output_path: Path

    @property
    def total_duration(self) -> float:
        return sum(segment.playback_duration for segment in self.segments)

    def clip_ids(self) -> list[str]:
        return [segment.clip.id for segment in self.segments]


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

    clips.sort(key=lambda clip: (level_number(clip.level), clip.timestamp, clip.order, clip.id))
    logger.info("Loaded {} assets from {}", len(clips), path)
    return clips


def prepare_asset_menu(
    clips: list[Clip],
    rng: random.Random,
    min_candidates: int = DEFAULT_CANDIDATE_MIN,
    max_candidates: int = DEFAULT_CANDIDATE_MAX,
) -> tuple[list[dict[str, Any]], list[Clip]]:
    usable = [clip for clip in clips if clip.path.exists()]
    missing = len(clips) - len(usable)
    if missing:
        logger.warning("Skipped {} assets because local video paths do not exist.", missing)
    if not usable:
        raise RuntimeError("No usable clips found in global_assets.json.")

    max_candidates = max(1, int(max_candidates))
    min_candidates = max(1, min(int(min_candidates), max_candidates))
    if len(usable) <= max_candidates:
        picked = usable
    else:
        picked = select_candidate_clips(usable, rng, min_candidates=min_candidates, max_candidates=max_candidates)

    picked = sorted(picked, key=lambda clip: (level_number(clip.level), clip.timestamp, clip.order, clip.id))
    return [clip.menu_record() for clip in picked], picked


def select_candidate_clips(
    clips: list[Clip],
    rng: random.Random,
    min_candidates: int,
    max_candidates: int,
) -> list[Clip]:
    selected: dict[str, Clip] = {}

    for clip in clips:
        if clip.action_type in VICTORY_ACTIONS:
            selected[clip.id] = clip

    for trial, success in find_trial_success_pairs(clips):
        selected[trial.id] = trial
        selected[success.id] = success
        if len(selected) >= max_candidates:
            break

    grouped: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        grouped[clip.level].append(clip)
    levels = sorted(grouped, key=level_number)
    per_level_quota = max(1, max_candidates // max(1, len(levels)))
    for level in levels:
        level_clips = grouped[level]
        success_clips = [clip for clip in level_clips if clip.action_type in SUCCESS_ACTIONS]
        rng.shuffle(success_clips)
        for clip in success_clips[:per_level_quota]:
            selected[clip.id] = clip
            if len(selected) >= max_candidates:
                break
        if len(selected) >= max_candidates:
            break

    shuffled = list(clips)
    rng.shuffle(shuffled)
    for clip in shuffled:
        selected.setdefault(clip.id, clip)
        if len(selected) >= min_candidates:
            break
    while len(selected) > max_candidates:
        removable = [clip_id for clip_id, clip in selected.items() if clip.action_type not in VICTORY_ACTIONS]
        if not removable:
            break
        selected.pop(rng.choice(removable), None)

    return list(selected.values())


def find_trial_success_pairs(clips: list[Clip], max_gap_seconds: float = 30.0) -> list[tuple[Clip, Clip]]:
    grouped: dict[tuple[str, str], list[Clip]] = defaultdict(list)
    for clip in clips:
        grouped[(clip.level, clip.original_video)].append(clip)

    pairs: list[tuple[Clip, Clip]] = []
    for timeline in grouped.values():
        timeline.sort(key=lambda clip: (clip.timestamp, clip.order, clip.id))
        for current, next_clip in zip(timeline, timeline[1:]):
            if current.action_type not in FAIL_ACTIONS or next_clip.action_type not in SUCCESS_ACTIONS:
                continue
            gap = max(0.0, next_clip.timestamp - (current.timestamp + current.duration))
            if gap <= max_gap_seconds:
                pairs.append((current, next_clip))
    return sorted(pairs, key=lambda pair: (level_number(pair[0].level), pair[0].timestamp))


class LLMDirector:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.9,
        max_tokens: int = 2048,
    ) -> None:
        self.model = (
            model
            or os.getenv("STAGE3_DIRECTOR_MODEL", "").strip()
            or os.getenv("OPENROUTER_MODEL", "").strip()
            or DEFAULT_MODEL
        )
        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY", "sk-or-v1-5794a8b038307965ef5bcdfea40fcfc18").strip()
        if not resolved_api_key:
            raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.")
        resolved_base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "").strip() or DEFAULT_BASE_URL
        self.client = make_openai_client(api_key=resolved_api_key, base_url=resolved_base_url)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def create_script(
        self,
        asset_menu: list[dict[str, Any]],
        run_index: int,
        temperature: float | None = None,
        extra_hint: str | None = None,
    ) -> str:
        user_prompt = build_director_user_prompt(asset_menu, run_index=run_index, extra_hint=extra_hint)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature if temperature is None else float(temperature),
            max_tokens=self.max_tokens,
        )
        raw_text = (response.choices[0].message.content or "").strip()
        if not raw_text:
            raise RuntimeError("Director LLM returned an empty response.")
        return raw_text


def build_director_user_prompt(
    asset_menu: list[dict[str, Any]],
    run_index: int,
    extra_hint: str | None = None,
) -> str:
    levels = sorted({str(item.get("level")) for item in asset_menu}, key=level_number)
    action_summary: dict[str, int] = defaultdict(int)
    for item in asset_menu:
        action_summary[str(item.get("action_type"))] += 1
    hint = extra_hint or random_style_hint(run_index)
    return (
        f"这是第 {run_index} 条批量成片，请生成一个和其他批次有差异的剪辑剧本。\n"
        f"可用关卡: {', '.join(levels)}\n"
        f"动作统计: {dict(sorted(action_summary.items()))}\n"
        f"差异化提示: {hint}\n\n"
        "【素材库 JSON】如下。请只使用其中存在的 id，不要编造 id。\n"
        f"{json.dumps(asset_menu, ensure_ascii=False, indent=2)}"
    )


def parse_director_script(raw_text: str, clip_by_id: dict[str, Clip]) -> list[DirectedSegment]:
    parsed = parse_json_array(raw_text)
    segments: list[DirectedSegment] = []
    seen: set[str] = set()
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            logger.warning("Director item #{} is not an object; skipped.", index)
            continue
        clip_id = str(item.get("id") or "").strip()
        if not clip_id:
            logger.warning("Director item #{} has no id; skipped.", index)
            continue
        clip = clip_by_id.get(clip_id)
        if clip is None:
            logger.warning("Director selected unknown clip id {}; skipped.", clip_id)
            continue
        if clip_id in seen:
            logger.warning("Director selected duplicate clip id {}; skipped.", clip_id)
            continue
        speed = clamp_speed(item.get("speed"))
        reason = str(item.get("reason") or "").strip() or "LLM director selection"
        segments.append(DirectedSegment(clip=clip, speed=speed, reason=reason))
        seen.add(clip_id)

    if not segments:
        raise RuntimeError("Director script did not contain any usable clip ids.")
    validate_micro_continuity(segments)
    return segments


def parse_json_array(raw_text: str) -> list[Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            raise ValueError(f"LLM response does not contain a JSON array: {raw_text[:500]}")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Director response must be a JSON array.")
    return parsed


def validate_micro_continuity(segments: list[DirectedSegment], max_gap_seconds: float = 30.0) -> None:
    for index, segment in enumerate(segments[:-1]):
        if segment.clip.action_type not in FAIL_ACTIONS:
            continue
        next_segment = segments[index + 1]
        same_level = next_segment.clip.level == segment.clip.level
        near_time = abs(next_segment.clip.timestamp - segment.clip.timestamp) <= max_gap_seconds
        success = next_segment.clip.action_type in SUCCESS_ACTIONS
        if not (same_level and near_time and success):
            logger.warning(
                "Director continuity warning: trial_error {} is followed by {} instead of nearby same-level success_step.",
                segment.clip.id,
                next_segment.clip.id,
            )


def concat_directed_plan_with_ffmpeg(segments: list[DirectedSegment], output_path: Path, hard_limit: float = TARGET_DURATION) -> None:
    if not segments:
        raise ValueError("cannot render an empty director plan")

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
            f"[{video_input}:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
            "setsar=1,fps=30,format=yuv420p,"
            f"setpts={setpts_factor:.8f}*PTS"
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
            "-t",
            f"{hard_limit:.3f}",
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
    run_ffmpeg(command, f"ffmpeg director concat failed: {output_path}")


def generate_ai_directed_videos(
    batch_size: int = DEFAULT_BATCH_COUNT,
    assets_json: str | Path = "data/global_assets.json",
    output_dir: str | Path = "data/processed",
    seed: int | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    candidate_min: int = DEFAULT_CANDIDATE_MIN,
    candidate_max: int = DEFAULT_CANDIDATE_MAX,
    temperature: float = 0.9,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    clips = load_assets(assets_json)
    root = resolve_output_root(output_dir)
    batch_dir = root / "batch_story_blocks" / safe_name(run_id or timestamp())
    batch_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(batch_dir / "stage3.log"), rotation="10 MB", level="INFO", encoding="utf-8")

    director = LLMDirector(model=model, api_key=api_key, base_url=base_url, temperature=temperature)
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    logger.info(
        "Stage3 AI director started | batch_size={} | model={} | output={}",
        batch_size,
        director.model,
        batch_dir,
    )

    for batch_index in range(1, int(batch_size) + 1):
        asset_menu, candidate_clips = prepare_asset_menu(
            clips,
            rng=rng,
            min_candidates=candidate_min,
            max_candidates=candidate_max,
        )
        clip_by_id = {clip.id: clip for clip in candidate_clips}
        output_path = batch_dir / f"video_{batch_index:02d}_AI_Director.mp4"
        run_temperature = min(1.4, max(0.1, temperature + (batch_index - 1) * 0.03))
        style_hint = random_style_hint(batch_index)

        try:
            raw_response = director.create_script(
                asset_menu,
                run_index=batch_index,
                temperature=run_temperature,
                extra_hint=style_hint,
            )
            segments = parse_director_script(raw_response, clip_by_id)
            plan = DirectedPlan(
                index=batch_index,
                style_hint=style_hint,
                segments=segments,
                raw_response=raw_response,
                candidate_count=len(candidate_clips),
                output_path=output_path,
            )
            record = plan_to_manifest_record(plan)
            logger.info(
                "[{}/{}] Director plan | clips={} | calculated={:.3f}s | output={}",
                batch_index,
                batch_size,
                len(segments),
                plan.total_duration,
                output_path,
            )

            if not dry_run:
                concat_directed_plan_with_ffmpeg(segments, output_path, hard_limit=TARGET_DURATION)
            results.append(record)
        except Exception as exc:
            logger.exception("AI directed video {} failed: {}", batch_index, exc)
            failures.append(f"video {batch_index:02d}: {exc}")

    manifest = {
        "run_id": batch_dir.name,
        "engine": "llm_as_director",
        "dry_run": dry_run,
        "model": director.model,
        "count_requested": batch_size,
        "count_planned": len(results),
        "target_duration": TARGET_DURATION,
        "director_duration_range": [MIN_DIRECTOR_DURATION, MAX_DIRECTOR_DURATION],
        "failures": failures,
        "videos": results,
    }
    manifest_path = batch_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (batch_dir / "content_index.txt").write_text(build_content_index(results), encoding="utf-8")
    logger.info("Stage3 AI director finished | videos={} | manifest={}", len(results), manifest_path)

    if not results:
        raise RuntimeError(f"Stage3 AI director could not generate any videos. Failures: {failures}")
    return results


def generate_batch(
    count: int = DEFAULT_BATCH_COUNT,
    assets_json: str | Path = "data/global_assets.json",
    output_dir: str | Path = "data/processed",
    seed: int | None = None,
    dry_run: bool = False,
    max_gap_seconds: float = 30.0,
    run_id: str | None = None,
    templates: list[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for the former template engine entrypoint."""
    if templates:
        logger.info("Stage3 AI director ignores legacy template filters: {}", templates)
    return generate_ai_directed_videos(
        batch_size=count,
        assets_json=assets_json,
        output_dir=output_dir,
        seed=seed,
        dry_run=dry_run,
        run_id=run_id,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


class VideoAssembler:
    """Compatibility wrapper used by src/main.py."""

    def __init__(
        self,
        interim_dir: str = "data/interim",
        processed_dir: str = "data/processed",
        global_assets_path: str = "data/global_assets.json",
        batch_size: int = DEFAULT_BATCH_COUNT,
        target_duration: float = TARGET_DURATION,
        max_gap_seconds: float = 30.0,
        recipe_names: list[str] | None = None,
        run_id: str | None = None,
        seed: int | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
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
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def run(self, only_video: str | None = None) -> list[dict[str, Any]]:
        if only_video:
            logger.info("Stage3 AI director uses global_assets.json; only_video={} is ignored.", only_video)
        return generate_ai_directed_videos(
            batch_size=self.batch_size,
            assets_json=self.global_assets_path,
            output_dir=self.processed_dir,
            seed=self.seed,
            dry_run=False,
            run_id=self.run_id,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )


VideoHighlightAssembler = VideoAssembler


def plan_to_manifest_record(plan: DirectedPlan) -> dict[str, Any]:
    return {
        "index": plan.index,
        "engine": "llm_as_director",
        "style_hint": plan.style_hint,
        "output_path": json_path(plan.output_path),
        "candidate_count": plan.candidate_count,
        "total_duration": round(plan.total_duration, 3),
        "will_be_hard_clipped_to": TARGET_DURATION,
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
                "reason": segment.reason,
            }
            for segment in plan.segments
        ],
        "director_raw_response": plan.raw_response,
    }


def build_content_index(records: list[dict[str, Any]]) -> str:
    lines = ["Stage 3 AI Director Batch", ""]
    for item in records:
        lines.append(
            f"{item['index']:02d}. AI_Director | {item['total_duration']:.1f}s -> 30.0s | {item['output_path']}"
        )
        for clip in item["clip_sequence"]:
            lines.append(
                "    "
                f"{clip['level']} {clip['action_type']} "
                f"{Path(clip['path']).name} | speed={clip['speed']:.2f} | "
                f"play={clip['playback_duration']:.2f}s | {clip['reason']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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
    remaining = float(speed)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.8f}" for factor in factors)


def make_openai_client(api_key: str, base_url: str):
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The stage3 AI director requires the openai Python package. "
            "Install project dependencies or run: pip install openai"
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url)


def clamp_speed(value: Any) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        speed = 1.0
    if speed < 0.8 or speed > 2.0:
        logger.warning("Director speed {} outside recommended range; clamped to 0.8-2.0.", speed)
    return min(2.0, max(0.8, speed))


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
    text = str(value or "Unknown_Level").strip() or "Unknown_Level"
    match = re.search(r"(\d+)", text)
    if match:
        return f"Level_{int(match.group(1))}"
    return safe_name(text)


def level_number(level: str) -> int:
    match = re.search(r"(\d+)", str(level))
    return int(match.group(1)) if match else 999999


def random_style_hint(run_index: int) -> str:
    hints = [
        "优先选择多关卡平分秋色，最后用最高关卡 victory 收束。",
        "优先选择单关卡深度解剖，强调 trial_error 到 success_step 的因果关系。",
        "让前半段更慢更清楚，后半段连续 success_step 提速制造爽感。",
        "允许少量倒叙高光，但 trial_error 后必须立刻接同关卡成功片段。",
        "尽量选择不同原视频来源，避免画面重复感。",
    ]
    return hints[(run_index - 1) % len(hints)]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3 LLM-as-a-Director video editing engine")
    parser.add_argument("--assets-json", "--global-assets-path", dest="assets_json", default="data/global_assets.json")
    parser.add_argument("--output-dir", "--processed-dir", dest="output_dir", default="data/processed")
    parser.add_argument("--count", "--batch-size", dest="count", type=int, default=DEFAULT_BATCH_COUNT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default=None, help="OpenRouter model name for the stage3 director.")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.")
    parser.add_argument("--base-url", default=None, help="OpenRouter-compatible base URL.")
    parser.add_argument("--candidate-min", type=int, default=DEFAULT_CANDIDATE_MIN)
    parser.add_argument("--candidate-max", type=int, default=DEFAULT_CANDIDATE_MAX)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    load_env_file(PROJECT_ROOT / ".env")
    started = time.perf_counter()
    logger.info("=== Stage3 AI director engine started ===")
    outputs = generate_ai_directed_videos(
        batch_size=args.count,
        assets_json=args.assets_json,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run=args.dry_run,
        run_id=args.run_id,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        candidate_min=args.candidate_min,
        candidate_max=args.candidate_max,
        temperature=args.temperature,
    )
    logger.info("=== Stage3 finished | outputs={} | elapsed={:.2f}s ===", len(outputs), time.perf_counter() - started)


if __name__ == "__main__":
    main()
