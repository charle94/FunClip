#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
"""Hierarchical (Map-Reduce) LLM analysis for long SRT subtitles.

Long videos produce SRT files that easily exceed an LLM's effective
attention window: even when the raw token count fits, models tend to lose
track of overall plot, pacing, and the location of the climax. This module
implements a small two-pass architecture on top of the existing
``llm_inference`` building blocks:

Step 1 (Map). The SRT is split into overlapping chunks of roughly two
minutes (50 cues, 8-cue overlap). Each chunk is sent to the LLM with
:data:`MAP_SYSTEM_PROMPT` / :data:`MAP_USER_PROMPT`, which constrains the
output to a compact JSON "narrative card" with the chunk's time range,
narrative role (hook / setup / rising / turn / climax / closing),
intensity score 1-5 and a one-sentence summary. Each chunk is independent
so calls run in parallel.

Step 2 (Reduce). All cards are concatenated (a few hundred tokens per
chunk after compression) and fed back to the LLM with
:data:`REDUCE_USER_PROMPT`, which asks for 3-8 final clip ranges. The
reduce-stage output uses the standard ``N. [HH:MM:SS,mmm-HH:MM:SS,mmm]``
format so the existing ``extract_timestamps`` pipeline picks it up
unchanged.

The orchestrator falls back to a single-shot call when the input is short
enough (``len(chunks) <= 1``) so users do not pay the cost of map+reduce
on small clips.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from utils.subtitle_utils import parse_srt, time_convert

# Type alias: any callable matching ``llm_inference(system, user, srt, model, apikey)``.
LLMCaller = Callable[[str, str, str, str, str], str]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
MAP_SYSTEM_PROMPT = (
    "你是视频叙事分析师。下面给你一个视频的某一段 SRT 字幕（仅是整段视频的一个片段）。\n"
    "请仅基于这段字幕，输出一条 JSON 格式的“叙事摘要卡”，描述这段内容的时间范围、"
    "在整体短视频叙事结构中的角色、情绪强度，以及 1-2 句的内容摘要。\n"
    "严格按照如下 JSON 输出，禁止输出任何额外文字、Markdown 代码围栏或解释：\n"
    "{\n"
    "  \"range\": \"[HH:MM:SS,mmm-HH:MM:SS,mmm]\",\n"
    "  \"role\": \"hook|setup|rising|turn|climax|closing\",\n"
    "  \"intensity\": 1-5 之间的整数,\n"
    "  \"summary\": \"用 1-2 句话概括本段核心内容\"\n"
    "}\n"
    "其中 range 必须使用本段字幕的真实起止时间戳，不得编造；intensity 越大表示戏剧张力越强；"
    "role 必须是给定六个枚举之一，无法判断时使用 \"setup\"。"
)

MAP_USER_PROMPT = "下面是该视频片段的 SRT 字幕，请输出叙事摘要卡 JSON："

REDUCE_SYSTEM_PROMPT = (
    "你是一名资深短视频剪辑师。下面给你一份按时间顺序排列的“叙事摘要卡”列表，"
    "每张卡描述了完整视频中一个时间段的叙事角色（开场钩子 / 起 / 承 / 转 / 高潮 / 收尾）、"
    "情绪强度（1-5）以及内容摘要。请基于这些卡片整体把握剧情和节奏，"
    "挑选出 3 到 8 个最具戏剧张力、最能体现高潮与转折的片段，组成一条逻辑连贯的短视频。\n"
    "选片要求：\n"
    "1. 优先挑选 role 为 climax 或 turn 且 intensity>=4 的卡片，并辅以必要的 hook / setup / closing 形成叙事闭环；\n"
    "2. 每个片段时长建议 15-60 秒，时间戳必须直接来自所给卡片的 range 字段，不得编造；\n"
    "3. 时间段所在行必须严格使用如下格式，独立成行：\n"
    "   N. [HH:MM:SS,mmm-HH:MM:SS,mmm] 简介文案\n"
    "   其中 N 从 1 开始，连接符为半角\"-\"，方括号外紧跟一句简介；\n"
    "4. 在所有时间段行之后，可另起一段附上整体剪辑思路（情节脉络、高潮位置、转折说明），"
    "禁止在时间段行内插入换行或额外标点，禁止使用 Markdown 表格。"
)

REDUCE_USER_PROMPT = (
    "以下是按时间顺序排列的叙事摘要卡列表，请据此完成全局选片，"
    "并严格按系统指令的格式输出最终的剪辑片段："
)


# Fold-stage prompts. When a movie produces too many Map-stage cards, groups
# of consecutive cards are summarised into a single coarser "act card" so the
# Reduce stage never has to reason over hundreds of cards at once.
FOLD_SYSTEM_PROMPT = (
    "你是视频叙事分析师。下面给你若干张按时间顺序排列的“叙事摘要卡”，"
    "它们共同覆盖完整视频中一个较大的时间段（一个场景或一幕）。\n"
    "请把它们归纳为一条更高层次的 JSON 摘要卡，描述这一整幕的时间范围、"
    "在整体叙事中的角色、情绪强度峰值，以及 1-2 句的剧情概括。\n"
    "严格按照如下 JSON 输出，禁止输出任何额外文字、Markdown 代码围栏或解释：\n"
    "{\n"
    "  \"range\": \"[HH:MM:SS,mmm-HH:MM:SS,mmm]\",\n"
    "  \"role\": \"hook|setup|rising|turn|climax|closing\",\n"
    "  \"intensity\": 1-5 之间的整数,\n"
    "  \"summary\": \"用 1-2 句话概括这一整幕的核心剧情与转折\"\n"
    "}\n"
    "range 必须覆盖这些卡片的整体起止时间，取第一张卡的起点与最后一张卡的终点；"
    "intensity 取这些卡片中的最高戏剧张力；role 若其中包含 climax 或 turn，则优先取该角色。"
)

FOLD_USER_PROMPT = "下面是同一幕内的若干叙事摘要卡，请归纳为一条更高层次的叙事摘要卡 JSON："


# Movie-oriented Reduce prompts. Unlike the short-video reduce prompt, these
# ask the model to first lay out the overall plot structure and turning points
# (识别主要情节与转折) and only then pick the clip ranges, which works better
# for feature-length content.
MOVIE_REDUCE_SYSTEM_PROMPT = (
    "你是一名资深电影解说与混剪剪辑师。下面给你一份按时间顺序排列的“叙事摘要卡”列表，"
    "每张卡描述了完整影片中一个时间段的叙事角色（开场钩子 / 起 / 承 / 转 / 高潮 / 收尾）、"
    "情绪强度（1-5）以及剧情概括；列表开头可能附有“关键转折提示”，标注了程序初步识别出的"
    "主要情节与转折所在的时间段。\n"
    "请完成两件事：\n"
    "第一步：通读所有卡片，理清影片的主线剧情，识别出主要情节节点与关键转折"
    "（例如开端、激励事件、中点反转、高潮、结局），把握整体节奏。\n"
    "第二步：基于上述理解，挑选出 3 到 8 个最能讲清主线剧情、最具戏剧张力的片段，"
    "组成一条逻辑连贯、能独立看懂的短视频。\n"
    "选片要求：\n"
    "1. 必须覆盖主要转折与高潮，并辅以必要的开端/收尾形成完整叙事闭环；\n"
    "2. 每个片段时长建议 15-90 秒，时间戳必须直接来自所给卡片的 range 字段，不得编造；\n"
    "3. 时间段所在行必须严格使用如下格式，独立成行：\n"
    "   N. [HH:MM:SS,mmm-HH:MM:SS,mmm] 简介文案\n"
    "   其中 N 从 1 开始，连接符为半角\"-\"，方括号外紧跟一句简介；\n"
    "4. 请在所有时间段行【之后】，另起一段以“剧情脉络：”开头，简述主线情节、"
    "关键转折位置与高潮所在，禁止在时间段行内插入换行或额外标点，禁止使用 Markdown 表格。"
)

MOVIE_REDUCE_USER_PROMPT = (
    "以下是按时间顺序排列的叙事摘要卡列表（含关键转折提示），请先梳理主线剧情与转折，"
    "再据此完成全局选片，并严格按系统指令的格式输出最终的剪辑片段："
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_sentences(
    sentences: List[Dict],
    window: int = 50,
    overlap: int = 8,
) -> List[List[Dict]]:
    """Split ``sentences`` into overlapping chunks.

    ``window`` is the number of cues per chunk and ``overlap`` is the number
    of trailing cues from the previous chunk that are repeated at the start
    of the next chunk. A non-positive ``window`` falls back to ``50`` and
    ``overlap`` is clamped to ``[0, window-1]``.

    Returns a list of (non-empty) cue sub-lists. When ``sentences`` is empty
    the result is ``[]``; when it fits in a single window the result is
    ``[sentences]``.
    """
    if not sentences:
        return []
    if window <= 0:
        window = 50
    if overlap < 0:
        overlap = 0
    if overlap >= window:
        overlap = window - 1
    step = window - overlap
    chunks: List[List[Dict]] = []
    n = len(sentences)
    i = 0
    while i < n:
        chunks.append(sentences[i:i + window])
        if i + window >= n:
            break
        i += step
    return chunks


def chunk_to_srt(chunk: List[Dict]) -> str:
    """Render a chunk of parsed sentences back to a numbered SRT block.

    The block uses the same format produced by ``parse_srt``'s
    ``normalized_srt`` so the LLM sees identical timestamp formatting in
    both passes.
    """
    lines: List[str] = []
    for idx, sent in enumerate(chunk):
        text = sent.get('raw_text')
        if text is None:
            t = sent.get('text', [])
            text = ''.join(t) if isinstance(t, list) else str(t)
        lines.append(str(idx))
        lines.append("{} --> {}".format(
            time_convert(sent['start']), time_convert(sent['end'])))
        lines.append(text)
        lines.append('')
    return '\n'.join(lines)


def chunk_time_range(chunk: List[Dict]) -> Tuple[int, int]:
    """Return ``(start_ms, end_ms)`` of a chunk."""
    return chunk[0]['start'], chunk[-1]['end']


# ---------------------------------------------------------------------------
# Map / Reduce stages
# ---------------------------------------------------------------------------
_JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)
_VALID_ROLES = {"hook", "setup", "rising", "turn", "climax", "closing"}


def parse_card(raw: str, fallback_range: str) -> Dict:
    """Parse a Map-stage LLM response into a structured card.

    Tolerates LLMs that wrap JSON in Markdown code fences, prepend prose,
    or omit / mistype individual fields. Returns a card dict with keys
    ``range``, ``role``, ``intensity``, ``summary``; missing fields use
    sensible defaults so the Reduce stage never sees malformed input.
    """
    card = {
        'range': fallback_range,
        'role': 'setup',
        'intensity': 3,
        'summary': '',
    }
    if not raw:
        return card
    text = raw.strip()
    # Strip Markdown code fences if present.
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        card['summary'] = text[:120]
        return card
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        card['summary'] = text[:120]
        return card
    if not isinstance(obj, dict):
        return card
    rng = obj.get('range')
    if isinstance(rng, str) and rng.strip():
        card['range'] = rng.strip()
    role = obj.get('role')
    if isinstance(role, str) and role.strip().lower() in _VALID_ROLES:
        card['role'] = role.strip().lower()
    intensity = obj.get('intensity')
    try:
        iv = int(intensity)
        if 1 <= iv <= 5:
            card['intensity'] = iv
    except (TypeError, ValueError):
        pass
    summary = obj.get('summary')
    if isinstance(summary, str):
        card['summary'] = summary.strip()
    return card


def format_cards_for_reduce(cards: List[Dict]) -> str:
    """Render a list of cards into the text payload for the Reduce stage."""
    lines: List[str] = []
    for i, c in enumerate(cards, 1):
        lines.append(
            "卡片{idx}: range={rng} role={role} intensity={inten}\n摘要: {summary}".format(
                idx=i,
                rng=c.get('range', ''),
                role=c.get('role', 'setup'),
                inten=c.get('intensity', 3),
                summary=c.get('summary', ''),
            )
        )
    return '\n\n'.join(lines)


# ---------------------------------------------------------------------------
# Plot / turning-point detection (deterministic)
# ---------------------------------------------------------------------------
_RANGE_RE = re.compile(
    r'(\d{1,2}:\d{2}:\d{2}[,\.:]\d{1,3})\s*-\s*(\d{1,2}:\d{2}:\d{2}[,\.:]\d{1,3})'
)


def _parse_range_ms(range_str: str) -> Optional[Tuple[int, int]]:
    """Parse a card ``range`` like ``[HH:MM:SS,mmm-HH:MM:SS,mmm]`` to ms.

    Returns ``None`` when the string cannot be parsed so callers can fall
    back gracefully.
    """
    if not range_str:
        return None
    m = _RANGE_RE.search(range_str)
    if not m:
        return None

    def _to_ms(t: str) -> int:
        t = t.replace('.', ',')
        parts = re.split(r'[:,]', t)
        h, mi, s, ms = (parts + ['0', '0', '0', '0'])[:4]
        ms = ms.zfill(3)
        return ((int(h) * 60 + int(mi)) * 60 + int(s)) * 1000 + int(ms)

    try:
        return _to_ms(m.group(1)), _to_ms(m.group(2))
    except (ValueError, IndexError):
        return None


def detect_turning_points(cards: List[Dict]) -> List[int]:
    """Identify the indices of the main plot turning points in ``cards``.

    A card is flagged as a turning point when it is dramatically more intense
    than the surrounding context. The heuristic is deliberately simple and
    deterministic so it can run without an LLM and be unit-tested:

    * Any card whose ``role`` is ``turn`` or ``climax`` is always a turning
      point (these roles encode an explicit narrative shift).
    * Any card whose ``intensity`` rises by ``>= 2`` versus the previous card
      marks a turning point (a sharp escalation).
    * Local intensity maxima with ``intensity >= 4`` are kept as turning
      points so the global climax is never missed.

    Returns a sorted list of unique card indices (0-based).
    """
    points = set()
    n = len(cards)

    def _inten(card) -> int:
        try:
            return int(card.get('intensity', 3))
        except (TypeError, ValueError):
            return 3

    for i, c in enumerate(cards):
        role = str(c.get('role', '')).lower()
        inten = _inten(c)
        if role in ('turn', 'climax'):
            points.add(i)
            continue
        if i > 0 and inten - _inten(cards[i - 1]) >= 2:
            points.add(i)
        prev_i = _inten(cards[i - 1]) if i > 0 else -1
        next_i = _inten(cards[i + 1]) if i + 1 < n else -1
        if inten >= 4 and inten >= prev_i and inten >= next_i:
            points.add(i)
    return sorted(points)


def format_plot_map(cards: List[Dict], turning_points: List[int]) -> str:
    """Render cards plus a turning-point header for the movie Reduce stage.

    The header lists the program-detected main turning points (1-based card
    numbers, their time range and role) so the LLM is explicitly told where
    the plot shifts before it sees the full card list.
    """
    header_lines = ["关键转折提示（程序初步识别，供参考）："]
    if turning_points:
        for idx in turning_points:
            c = cards[idx]
            header_lines.append(
                "- 卡片{n}: range={rng} role={role} intensity={inten}".format(
                    n=idx + 1,
                    rng=c.get('range', ''),
                    role=c.get('role', 'setup'),
                    inten=c.get('intensity', 3),
                )
            )
    else:
        header_lines.append("- （未检测到明显转折，请根据摘要自行判断）")
    header = '\n'.join(header_lines)
    return header + '\n\n' + format_cards_for_reduce(cards)


# ---------------------------------------------------------------------------
# Fold stage (collapse many cards into fewer act-level cards)
# ---------------------------------------------------------------------------
def _merge_cards_fallback(group: List[Dict]) -> Dict:
    """Deterministically merge a group of cards into one act-level card.

    Used when no ``llm_caller`` is available or a fold call fails. The merged
    range spans the group, intensity takes the group maximum, role prefers an
    explicit ``climax``/``turn`` if present, and the summary concatenates the
    member summaries.
    """
    first_range = _parse_range_ms(group[0].get('range', ''))
    last_range = _parse_range_ms(group[-1].get('range', ''))
    if first_range and last_range:
        rng = "[{}-{}]".format(time_convert(first_range[0]), time_convert(last_range[1]))
    else:
        rng = group[0].get('range', '') or group[-1].get('range', '')
    intensity = 1
    role = 'setup'
    for c in group:
        try:
            intensity = max(intensity, int(c.get('intensity', 3)))
        except (TypeError, ValueError):
            pass
    for preferred in ('climax', 'turn', 'rising', 'hook', 'closing'):
        if any(str(c.get('role', '')).lower() == preferred for c in group):
            role = preferred
            break
    summaries = [str(c.get('summary', '')).strip() for c in group if c.get('summary')]
    summary = ' '.join(summaries)[:160]
    return {'range': rng, 'role': role, 'intensity': intensity, 'summary': summary}


def fold_cards(
    cards: List[Dict],
    llm_caller: Optional[LLMCaller] = None,
    model: str = '',
    apikey: str = '',
    max_cards: int = 40,
    group_size: int = 5,
    max_workers: int = 4,
) -> List[Dict]:
    """Recursively collapse ``cards`` until at most ``max_cards`` remain.

    Feature-length movies can yield far more Map-stage cards than a single
    Reduce call can reason over without losing the plot. This groups
    consecutive cards (``group_size`` per group) and summarises each group
    into one act-level card, repeating until ``len(cards) <= max_cards``.

    When ``llm_caller`` is provided each group is summarised by the LLM
    (:data:`FOLD_SYSTEM_PROMPT`); otherwise — or if a call fails — a
    deterministic merge (:func:`_merge_cards_fallback`) is used. Card order
    (and therefore chronological order) is always preserved.
    """
    if max_cards < 1:
        max_cards = 1
    if group_size < 2:
        group_size = 2
    guard = 0
    while len(cards) > max_cards and guard < 20:
        guard += 1
        groups = [cards[i:i + group_size] for i in range(0, len(cards), group_size)]

        def _fold_one(group: List[Dict]) -> Dict:
            if len(group) == 1:
                return group[0]
            fallback = _merge_cards_fallback(group)
            if llm_caller is None:
                return fallback
            payload = format_cards_for_reduce(group)
            try:
                raw = llm_caller(FOLD_SYSTEM_PROMPT, FOLD_USER_PROMPT, payload, model, apikey)
            except Exception:  # noqa: BLE001 - degrade to deterministic merge.
                logging.exception("Fold-stage LLM call failed; using deterministic merge")
                return fallback
            card = parse_card(raw or '', fallback['range'])
            # Keep the deterministic span/intensity if the LLM omitted them.
            if not _parse_range_ms(card.get('range', '')):
                card['range'] = fallback['range']
            if not card.get('summary'):
                card['summary'] = fallback['summary']
            return card

        new_cards: List[Optional[Dict]] = [None] * len(groups)
        workers = max(1, min(max_workers, len(groups)))
        if llm_caller is None or workers == 1:
            for idx, group in enumerate(groups):
                new_cards[idx] = _fold_one(group)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_idx = {pool.submit(_fold_one, g): i for i, g in enumerate(groups)}
                for fut in as_completed(fut_to_idx):
                    new_cards[fut_to_idx[fut]] = fut.result()
        folded = [c for c in new_cards if c is not None]
        if len(folded) >= len(cards):
            # No progress (e.g. group_size too large for the data); stop.
            break
        cards = folded
    return cards


def hierarchical_llm_inference(
    srt_text: str,
    model: str,
    apikey: str,
    llm_caller: LLMCaller,
    reduce_system_prompt: Optional[str] = None,
    reduce_user_prompt: Optional[str] = None,
    window: int = 50,
    overlap: int = 8,
    max_workers: int = 4,
    max_reduce_cards: Optional[int] = None,
    fold_group_size: int = 5,
    plot_aware: bool = False,
) -> str:
    """Run a Map-Reduce LLM analysis over a long SRT.

    Parameters
    ----------
    srt_text:
        Raw SRT subtitle text.
    model, apikey:
        Forwarded to ``llm_caller`` unchanged.
    llm_caller:
        Callable matching the existing ``llm_inference(system_content,
        user_content, srt_text, model, apikey)`` signature in
        ``funclip.launch``. Injected so this module never imports
        Gradio/launch-side state and stays unit-testable.
    reduce_system_prompt, reduce_user_prompt:
        Optional overrides for the Reduce-stage prompts. Defaults to the
        short-video narrative prompts defined above.
    window, overlap:
        Chunking knobs; see :func:`chunk_sentences`.
    max_workers:
        Upper bound on concurrent Map-stage LLM calls. Set to 1 to force
        sequential execution (e.g. when the upstream LLM has tight rate
        limits).
    max_reduce_cards:
        When set, an intermediate **Fold** stage collapses the Map-stage
        cards down to at most this many act-level cards (see
        :func:`fold_cards`) before the Reduce stage. This keeps feature-length
        movies, which produce dozens-to-hundreds of cards, within the LLM's
        effective attention window. ``None`` (default) disables folding so
        existing callers behave exactly as before.
    fold_group_size:
        Number of consecutive cards summarised per group during folding.
    plot_aware:
        When ``True`` the Reduce payload is rendered with
        :func:`format_plot_map`, prepending the deterministically detected
        main turning points so the LLM is told where the plot shifts.

    Returns
    -------
    str
        Final text emitted by the Reduce stage. Time-range lines follow
        the ``N. [HH:MM:SS,mmm-HH:MM:SS,mmm] caption`` format expected by
        :func:`utils.trans_utils.extract_timestamps`.
    """
    sentences, normalized_srt = parse_srt(srt_text)
    if not sentences:
        # Nothing to analyse; defer to the single-shot path.
        return llm_caller(
            reduce_system_prompt or REDUCE_SYSTEM_PROMPT,
            reduce_user_prompt or REDUCE_USER_PROMPT,
            srt_text or '',
            model,
            apikey,
        )

    chunks = chunk_sentences(sentences, window=window, overlap=overlap)
    # Auto-fallback for short clips: a single chunk does not benefit from
    # map-reduce, so just use the reduce prompt on the full SRT directly
    # (still better than the legacy general prompt for narrative pacing).
    if len(chunks) <= 1:
        return llm_caller(
            reduce_system_prompt or REDUCE_SYSTEM_PROMPT,
            reduce_user_prompt or REDUCE_USER_PROMPT,
            normalized_srt,
            model,
            apikey,
        )

    # ---- Map stage ----------------------------------------------------
    def _map_one(chunk: List[Dict]) -> Dict:
        start_ms, end_ms = chunk_time_range(chunk)
        fallback_range = "[{}-{}]".format(time_convert(start_ms), time_convert(end_ms))
        chunk_srt = chunk_to_srt(chunk)
        try:
            raw = llm_caller(MAP_SYSTEM_PROMPT, MAP_USER_PROMPT, chunk_srt, model, apikey)
        except Exception:  # noqa: BLE001 - propagate as a fallback card.
            logging.exception("Map-stage LLM call failed for chunk %s", fallback_range)
            return {
                'range': fallback_range, 'role': 'setup', 'intensity': 3, 'summary': ''
            }
        return parse_card(raw or '', fallback_range)

    cards: List[Optional[Dict]] = [None] * len(chunks)
    workers = max(1, min(max_workers, len(chunks)))
    if workers == 1:
        for idx, chunk in enumerate(chunks):
            cards[idx] = _map_one(chunk)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(_map_one, chunk): idx
                             for idx, chunk in enumerate(chunks)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                cards[idx] = fut.result()

    # Drop any (unexpected) None entries before reduce.
    cards_clean: List[Dict] = [c for c in cards if c is not None]

    # ---- Fold stage (optional) ----------------------------------------
    # Movies produce many cards; collapse them into act-level cards so the
    # Reduce stage keeps the whole plot in view.
    if max_reduce_cards is not None and len(cards_clean) > max_reduce_cards:
        cards_clean = fold_cards(
            cards_clean,
            llm_caller=llm_caller,
            model=model,
            apikey=apikey,
            max_cards=max_reduce_cards,
            group_size=fold_group_size,
            max_workers=max_workers,
        )

    if plot_aware:
        turning_points = detect_turning_points(cards_clean)
        payload = format_plot_map(cards_clean, turning_points)
    else:
        payload = format_cards_for_reduce(cards_clean)

    # ---- Reduce stage -------------------------------------------------
    return llm_caller(
        reduce_system_prompt or REDUCE_SYSTEM_PROMPT,
        reduce_user_prompt or REDUCE_USER_PROMPT,
        payload,
        model,
        apikey,
    )


def movie_llm_inference(
    srt_text: str,
    model: str,
    apikey: str,
    llm_caller: LLMCaller,
    reduce_system_prompt: Optional[str] = None,
    reduce_user_prompt: Optional[str] = None,
    window: int = 50,
    overlap: int = 8,
    max_workers: int = 4,
    max_reduce_cards: int = 40,
    fold_group_size: int = 5,
) -> str:
    """Plot-aware Map-Fold-Reduce tailored for feature-length movies.

    Thin wrapper over :func:`hierarchical_llm_inference` that enables the
    Fold stage and turning-point-aware Reduce payload, and defaults to the
    movie-oriented Reduce prompts which ask the model to first lay out the
    main plot and turning points before selecting clip ranges.
    """
    return hierarchical_llm_inference(
        srt_text=srt_text,
        model=model,
        apikey=apikey,
        llm_caller=llm_caller,
        reduce_system_prompt=reduce_system_prompt or MOVIE_REDUCE_SYSTEM_PROMPT,
        reduce_user_prompt=reduce_user_prompt or MOVIE_REDUCE_USER_PROMPT,
        window=window,
        overlap=overlap,
        max_workers=max_workers,
        max_reduce_cards=max_reduce_cards,
        fold_group_size=fold_group_size,
        plot_aware=True,
    )
