#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""Tests for the subtitle-driven AI clipping pipeline.

Covers the SRT parser and the LLM-output parsing used by AI Clip,
without requiring funasr/moviepy or a real LLM call.
"""

import os
import re
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FUNCLIP_DIR = os.path.dirname(THIS_DIR)
if FUNCLIP_DIR not in sys.path:
    sys.path.insert(0, FUNCLIP_DIR)

from utils.subtitle_utils import parse_srt, generate_srt_clip  # noqa: E402
from utils.trans_utils import extract_timestamps  # noqa: E402
from llm.hierarchical import (  # noqa: E402
    chunk_sentences,
    chunk_to_srt,
    parse_card,
    format_cards_for_reduce,
    hierarchical_llm_inference,
    movie_llm_inference,
    detect_turning_points,
    fold_cards,
    format_plot_map,
    _parse_range_ms,
    MAP_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    FOLD_SYSTEM_PROMPT,
    MOVIE_REDUCE_SYSTEM_PROMPT,
)


SRT_SAMPLE = (
    "\ufeff1\r\n"
    "00:00:00,500 --> 00:00:02,100\r\n"
    "读万卷书行万里路，\r\n"
    "\r\n"
    "2\r\n"
    "00:00:02.310 --> 00:00:03,990\r\n"
    "这里是读书三六九，\r\n"
    "\r\n"
    "00:00:04,670 --> 00:00:07,990\r\n"
    "今天要和您分享的这篇文章是人民日报，\r\n"
)


class ParseSrtTests(unittest.TestCase):

    def test_parses_three_cues_with_tolerated_quirks(self):
        sentences, normalized = parse_srt(SRT_SAMPLE)
        # BOM, CRLF, '.' separator, and a missing sequence number are tolerated.
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0]['start'], 500)
        self.assertEqual(sentences[0]['end'], 2100)
        self.assertEqual(sentences[1]['start'], 2310)
        self.assertEqual(sentences[2]['start'], 4670)
        # Each cue is shaped for downstream consumption.
        for s in sentences:
            self.assertIsInstance(s['text'], list)
            self.assertEqual(len(s['timestamp']), 1)
            self.assertEqual(len(s['timestamp'][0]), 2)
        # Normalized SRT is renderable and matches the prompt format
        # (zero-based indices, HH:MM:SS,mmm timestamps).
        self.assertIn("00:00:00,500 --> 00:00:02,100", normalized)
        self.assertTrue(normalized.lstrip().startswith("0\n"))

    def test_milliseconds_left_padded(self):
        # ",5" means 5ms (left-padded), ",50" means 50ms, ",500" means 500ms.
        # This matches the convention produced by time_convert and the SRT
        # spec, which zero-pads milliseconds on the left.
        sample = (
            "1\n00:00:00,5 --> 00:00:00,50\nshort\n\n"
            "2\n00:00:00,500 --> 00:00:01,000\nlong\n"
        )
        sentences, _ = parse_srt(sample)
        self.assertEqual(sentences[0]['start'], 5)
        self.assertEqual(sentences[0]['end'], 50)
        self.assertEqual(sentences[1]['start'], 500)
        self.assertEqual(sentences[1]['end'], 1000)

    def test_handles_empty_input(self):
        # parse_srt itself is forgiving and returns ([], '') for empty/None
        # inputs; VideoClipper.video_recog_from_srt is the public entry that
        # raises ValueError, but it depends on moviepy/librosa which are
        # heavyweight optional deps for the test runner.
        self.assertEqual(parse_srt("")[0], [])
        self.assertEqual(parse_srt(None), ([], ''))

    def test_clip_uses_parsed_sentences(self):
        sentences, _ = parse_srt(SRT_SAMPLE)
        clip_srt, subs, cc = generate_srt_clip(sentences, 0.5, 4.0)
        # Both the first cue (fully inside) and the second cue (fully inside)
        # should appear in the clipped SRT.
        self.assertEqual(len(subs), 2)
        self.assertIn("读万卷书行万里路", clip_srt)
        self.assertIn("这里是读书三六九", clip_srt)


class ExtractTimestampsTests(unittest.TestCase):

    def test_short_video_prompt_format_is_parsed(self):
        # This is the exact format the short_video_prompt_system constrains
        # the LLM to emit, with optional trailing narrative description.
        llm_output = (
            "1. [00:00:00,500-00:00:15,200] 开场钩子：抛出问题\n"
            "2. [00:00:30,000-00:01:05,990] 高潮段落\n"
            "3. [00:02:10,500-00:02:55,750] 收尾点题\n"
            "\n"
            "整体思路：开场抛出疑问，第二段是高潮，第三段升华。"
        )
        ts = extract_timestamps(llm_output)
        self.assertEqual(ts, [
            [500, 15200],
            [30000, 65990],
            [130500, 175750],
        ])


def _fmt(ms):
    """Format milliseconds as an SRT ``HH:MM:SS,mmm`` timestamp."""
    h = ms // 3600000
    mi = (ms // 60000) % 60
    sec = (ms // 1000) % 60
    mm = ms % 1000
    return "{:02d}:{:02d}:{:02d},{:03d}".format(h, mi, sec, mm)


def _make_srt(n_cues, start_ms=0, dur=2000, gap=200):
    """Build a synthetic SRT string with ``n_cues`` cues for chunking tests."""
    out = []
    t = start_ms
    for i in range(n_cues):
        s, e = t, t + dur
        out.append("{}\n{} --> {}\nline {}\n".format(i, _fmt(s), _fmt(e), i))
        t = e + gap
    return "\n".join(out)


class HierarchicalChunkingTests(unittest.TestCase):

    def test_chunk_sentences_overlap_and_step(self):
        sentences, _ = parse_srt(_make_srt(120))
        chunks = chunk_sentences(sentences, window=50, overlap=8)
        # Step is window-overlap=42; expected chunks: ceil((120-50)/42)+1 = 3.
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 50)
        self.assertEqual(len(chunks[1]), 50)
        self.assertLessEqual(len(chunks[2]), 50)
        # Overlap: last 8 cues of chunk 0 should reappear at start of chunk 1.
        self.assertEqual(
            [s['start'] for s in chunks[0][-8:]],
            [s['start'] for s in chunks[1][:8]],
        )
        # Coverage: every sentence appears in at least one chunk.
        covered = set()
        for c in chunks:
            for s in c:
                covered.add(s['start'])
        self.assertEqual(covered, {s['start'] for s in sentences})

    def test_chunk_sentences_short_input_returns_single_chunk(self):
        sentences, _ = parse_srt(_make_srt(10))
        chunks = chunk_sentences(sentences, window=50, overlap=8)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 10)

    def test_chunk_sentences_empty(self):
        self.assertEqual(chunk_sentences([], window=50, overlap=8), [])

    def test_chunk_sentences_overlap_clamped(self):
        # overlap >= window must not produce a zero / negative step.
        sentences, _ = parse_srt(_make_srt(20))
        chunks = chunk_sentences(sentences, window=5, overlap=10)
        # overlap is clamped to window-1=4, so step = window-overlap = 5-4 = 1;
        # we still make progress and cover all cues.
        self.assertGreater(len(chunks), 1)
        covered = set()
        for c in chunks:
            for s in c:
                covered.add(s['start'])
        self.assertEqual(covered, {s['start'] for s in sentences})

    def test_chunk_to_srt_round_trip_via_parse(self):
        sentences, _ = parse_srt(_make_srt(5))
        rendered = chunk_to_srt(sentences)
        # Re-parsing the chunk yields the same start/end timestamps.
        reparsed, _ = parse_srt(rendered)
        self.assertEqual(len(reparsed), 5)
        self.assertEqual(
            [(s['start'], s['end']) for s in reparsed],
            [(s['start'], s['end']) for s in sentences],
        )


class HierarchicalParseCardTests(unittest.TestCase):

    def test_parses_clean_json(self):
        raw = ('{"range":"[00:00:00,000-00:01:40,000]",'
               '"role":"climax","intensity":5,"summary":"主角揭示真相"}')
        c = parse_card(raw, "[00:00:00,000-00:01:40,000]")
        self.assertEqual(c['range'], "[00:00:00,000-00:01:40,000]")
        self.assertEqual(c['role'], 'climax')
        self.assertEqual(c['intensity'], 5)
        self.assertIn('真相', c['summary'])

    def test_strips_markdown_fence_and_prose(self):
        raw = ("分析如下：\n"
               "```json\n"
               "{\"range\": \"[00:00:00,000-00:00:30,000]\", "
               "\"role\": \"hook\", \"intensity\": 4, \"summary\": \"开场\"}\n"
               "```\n")
        c = parse_card(raw, "[00:00:00,000-00:00:30,000]")
        self.assertEqual(c['role'], 'hook')
        self.assertEqual(c['intensity'], 4)

    def test_falls_back_for_invalid_role_and_intensity(self):
        raw = '{"range":"[00:00:00,000-00:01:00,000]","role":"BOSS","intensity":99,"summary":"x"}'
        c = parse_card(raw, "[00:00:00,000-00:01:00,000]")
        # invalid role -> default 'setup', invalid intensity -> default 3.
        self.assertEqual(c['role'], 'setup')
        self.assertEqual(c['intensity'], 3)
        self.assertEqual(c['summary'], 'x')

    def test_falls_back_for_unparseable_output(self):
        c = parse_card("not json at all", "[00:00:00,000-00:01:00,000]")
        self.assertEqual(c['range'], "[00:00:00,000-00:01:00,000]")
        self.assertEqual(c['role'], 'setup')
        self.assertEqual(c['intensity'], 3)

    def test_handles_empty_input(self):
        c = parse_card("", "[00:00:00,000-00:00:01,000]")
        self.assertEqual(c['range'], "[00:00:00,000-00:00:01,000]")


class HierarchicalOrchestratorTests(unittest.TestCase):

    def _fake_caller(self, log):
        """Return an llm_caller stub that records calls and replies based on prompt."""
        def caller(system, user, srt, model, apikey):
            log.append({'system': system, 'user': user, 'srt': srt,
                        'model': model, 'apikey': apikey})
            if system == MAP_SYSTEM_PROMPT:
                # Find the first cue's start in the chunk and emit a card.
                m = re.search(r'(\d{2}:\d{2}:\d{2},\d{3}) -->', srt)
                start = m.group(1) if m else "00:00:00,000"
                m2 = re.findall(r'--> (\d{2}:\d{2}:\d{2},\d{3})', srt)
                end = m2[-1] if m2 else "00:00:00,000"
                return ('{"range":"[%s-%s]","role":"climax",'
                        '"intensity":5,"summary":"chunk"}' % (start, end))
            # Reduce stage: emit two correctly-formatted ranges so
            # extract_timestamps works downstream.
            return ("1. [00:00:00,000-00:00:10,000] 高潮一\n"
                    "2. [00:00:20,000-00:00:30,000] 高潮二")
        return caller

    def test_short_input_uses_single_shot(self):
        log = []
        srt = _make_srt(5)
        out = hierarchical_llm_inference(
            srt_text=srt, model='m', apikey='k',
            llm_caller=self._fake_caller(log), max_workers=1,
        )
        # Only one LLM call (single chunk -> single reduce call, no map).
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]['system'], REDUCE_SYSTEM_PROMPT)
        self.assertIn("高潮", out)

    def test_long_input_runs_map_then_reduce(self):
        log = []
        srt = _make_srt(120)
        out = hierarchical_llm_inference(
            srt_text=srt, model='m', apikey='k',
            llm_caller=self._fake_caller(log),
            window=50, overlap=8, max_workers=1,
        )
        # Map calls = number of chunks (3); plus 1 reduce call = 4.
        map_calls = [c for c in log if c['system'] == MAP_SYSTEM_PROMPT]
        reduce_calls = [c for c in log if c['system'] == REDUCE_SYSTEM_PROMPT]
        self.assertEqual(len(map_calls), 3)
        self.assertEqual(len(reduce_calls), 1)
        # Reduce-stage payload should contain the summarised cards, NOT
        # the original SRT (verifying compression actually happened).
        self.assertIn("卡片1", reduce_calls[0]['srt'])
        self.assertIn("intensity=5", reduce_calls[0]['srt'])
        self.assertNotIn("--> ", reduce_calls[0]['srt'])
        # Final timestamps are extractable by the existing pipeline.
        ts = extract_timestamps(out)
        self.assertEqual(ts, [[0, 10000], [20000, 30000]])

    def test_failed_map_call_falls_back_to_default_card(self):
        sentences, _ = parse_srt(_make_srt(120))
        chunks = chunk_sentences(sentences, window=50, overlap=8)
        self.assertEqual(len(chunks), 3)

        attempts = {'n': 0}

        def caller(system, user, srt, model, apikey):
            if system == MAP_SYSTEM_PROMPT:
                attempts['n'] += 1
                if attempts['n'] == 2:
                    raise RuntimeError("upstream timeout")
                return '{"range":"[00:00:00,000-00:00:10,000]","role":"setup","intensity":3,"summary":"ok"}'
            # Reduce: just echo so we can inspect the payload.
            return srt

        payload = hierarchical_llm_inference(
            srt_text=_make_srt(120), model='m', apikey='k',
            llm_caller=caller, window=50, overlap=8, max_workers=1,
        )
        # Three cards still produced even though one map call raised.
        self.assertEqual(payload.count("卡片"), 3)


class FormatCardsForReduceTests(unittest.TestCase):

    def test_includes_index_role_intensity_and_summary(self):
        cards = [
            {'range': '[00:00:00,000-00:00:30,000]', 'role': 'hook',
             'intensity': 4, 'summary': 'A'},
            {'range': '[00:00:30,000-00:01:00,000]', 'role': 'climax',
             'intensity': 5, 'summary': 'B'},
        ]
        text = format_cards_for_reduce(cards)
        self.assertIn('卡片1', text)
        self.assertIn('卡片2', text)
        self.assertIn('role=hook', text)
        self.assertIn('intensity=5', text)
        self.assertIn('A', text)
        self.assertIn('B', text)


class TurningPointTests(unittest.TestCase):

    def _cards(self, specs):
        cards = []
        for i, (role, inten) in enumerate(specs):
            s = i * 10000
            e = s + 9000
            cards.append({
                'range': '[{}-{}]'.format(_fmt(s), _fmt(e)),
                'role': role, 'intensity': inten, 'summary': 's{}'.format(i),
            })
        return cards

    def test_role_turn_and_climax_always_flagged(self):
        cards = self._cards([('setup', 2), ('turn', 3), ('setup', 2),
                             ('climax', 3)])
        self.assertEqual(detect_turning_points(cards), [1, 3])

    def test_sharp_intensity_rise_flagged(self):
        # 2 -> 4 is a +2 escalation; should be a turning point.
        cards = self._cards([('setup', 2), ('rising', 4), ('setup', 2)])
        self.assertIn(1, detect_turning_points(cards))

    def test_local_max_high_intensity_kept(self):
        # A local maximum at intensity 5 must never be missed (global climax).
        cards = self._cards([('rising', 4), ('rising', 5), ('rising', 4)])
        self.assertIn(1, detect_turning_points(cards))

    def test_flat_low_intensity_has_no_turning_points(self):
        cards = self._cards([('setup', 2), ('setup', 2), ('setup', 2)])
        self.assertEqual(detect_turning_points(cards), [])

    def test_handles_bad_intensity_values(self):
        cards = [
            {'range': '[00:00:00,000-00:00:09,000]', 'role': 'setup',
             'intensity': 'bad', 'summary': ''},
            {'range': '[00:00:10,000-00:00:19,000]', 'role': 'turn',
             'intensity': None, 'summary': ''},
        ]
        # Should not raise; the explicit 'turn' role is still flagged.
        self.assertIn(1, detect_turning_points(cards))


class ParseRangeTests(unittest.TestCase):

    def test_parses_bracketed_range(self):
        self.assertEqual(
            _parse_range_ms('[00:01:02,500-00:01:10,000]'), (62500, 70000))

    def test_tolerates_dot_separator(self):
        self.assertEqual(
            _parse_range_ms('00:00:01.250 - 00:00:02.000'), (1250, 2000))

    def test_returns_none_for_garbage(self):
        self.assertIsNone(_parse_range_ms('not a range'))
        self.assertIsNone(_parse_range_ms(''))


class FoldCardsTests(unittest.TestCase):

    def _many(self, n):
        cards = []
        for i in range(n):
            s = i * 10000
            e = s + 9000
            cards.append({
                'range': '[{}-{}]'.format(_fmt(s), _fmt(e)),
                'role': 'setup', 'intensity': 3, 'summary': 's{}'.format(i),
            })
        return cards

    def test_no_fold_when_within_limit(self):
        cards = self._many(5)
        self.assertEqual(fold_cards(cards, max_cards=10), cards)

    def test_deterministic_fold_preserves_span_and_order(self):
        cards = self._many(30)
        out = fold_cards(cards, llm_caller=None, max_cards=6, group_size=5)
        self.assertLessEqual(len(out), 6)
        # The folded list spans the same overall time range.
        self.assertEqual(_parse_range_ms(out[0]['range'])[0], 0)
        self.assertEqual(_parse_range_ms(out[-1]['range'])[1],
                         _parse_range_ms(cards[-1]['range'])[1])

    def test_llm_fold_invoked_per_group(self):
        cards = self._many(20)
        calls = {'n': 0}

        def caller(system, user, srt, model, apikey):
            self.assertEqual(system, FOLD_SYSTEM_PROMPT)
            calls['n'] += 1
            # Echo each group's real span so folding preserves chronology.
            starts = re.findall(r'range=\[(\d{2}:\d{2}:\d{2},\d{3})-', srt)
            ends = re.findall(r'-(\d{2}:\d{2}:\d{2},\d{3})\]', srt)
            s = starts[0] if starts else "00:00:00,000"
            e = ends[-1] if ends else "00:00:00,000"
            return ('{"range":"[%s-%s]","role":"climax",'
                    '"intensity":5,"summary":"act"}' % (s, e))

        out = fold_cards(cards, llm_caller=caller, model='m', apikey='k',
                         max_cards=4, group_size=5, max_workers=1)
        self.assertLessEqual(len(out), 4)
        self.assertGreater(calls['n'], 0)
        # Folded cards stay in chronological order and span the original range.
        self.assertEqual(_parse_range_ms(out[0]['range'])[0], 0)
        self.assertEqual(_parse_range_ms(out[-1]['range'])[1],
                         _parse_range_ms(cards[-1]['range'])[1])

    def test_fold_failure_falls_back_to_merge(self):
        cards = self._many(12)

        def caller(system, user, srt, model, apikey):
            raise RuntimeError("fold timeout")

        out = fold_cards(cards, llm_caller=caller, model='m', apikey='k',
                         max_cards=3, group_size=5, max_workers=1)
        # Deterministic merge keeps it progressing despite every call failing.
        self.assertLessEqual(len(out), 3)
        self.assertEqual(_parse_range_ms(out[0]['range'])[0], 0)


class FormatPlotMapTests(unittest.TestCase):

    def test_includes_turning_point_header_and_cards(self):
        cards = [
            {'range': '[00:00:00,000-00:00:30,000]', 'role': 'setup',
             'intensity': 2, 'summary': 'A'},
            {'range': '[00:00:30,000-00:01:00,000]', 'role': 'climax',
             'intensity': 5, 'summary': 'B'},
        ]
        text = format_plot_map(cards, detect_turning_points(cards))
        self.assertIn('关键转折提示', text)
        self.assertIn('卡片2', text)
        self.assertIn('A', text)
        self.assertIn('B', text)

    def test_header_present_even_without_turning_points(self):
        cards = [
            {'range': '[00:00:00,000-00:00:30,000]', 'role': 'setup',
             'intensity': 2, 'summary': 'A'},
        ]
        text = format_plot_map(cards, [])
        self.assertIn('关键转折提示', text)
        self.assertIn('未检测到明显转折', text)


class MovieInferenceTests(unittest.TestCase):

    def _caller(self, log):
        def caller(system, user, srt, model, apikey):
            log.append({'system': system, 'srt': srt})
            if system == MAP_SYSTEM_PROMPT:
                m = re.search(r'(\d{2}:\d{2}:\d{2},\d{3}) -->', srt)
                start = m.group(1) if m else "00:00:00,000"
                m2 = re.findall(r'--> (\d{2}:\d{2}:\d{2},\d{3})', srt)
                end = m2[-1] if m2 else "00:00:00,000"
                return ('{"range":"[%s-%s]","role":"climax",'
                        '"intensity":5,"summary":"c"}' % (start, end))
            if system == FOLD_SYSTEM_PROMPT:
                starts = re.findall(r'range=\[(\d{2}:\d{2}:\d{2},\d{3})-', srt)
                ends = re.findall(r'-(\d{2}:\d{2}:\d{2},\d{3})\]', srt)
                s = starts[0] if starts else "00:00:00,000"
                e = ends[-1] if ends else "00:00:00,000"
                return ('{"range":"[%s-%s]","role":"climax",'
                        '"intensity":5,"summary":"act"}' % (s, e))
            return ("1. [00:00:00,000-00:00:10,000] 开端\n"
                    "2. [00:00:20,000-00:00:30,000] 高潮")
        return caller

    def test_movie_uses_plot_aware_reduce(self):
        log = []
        out = movie_llm_inference(
            srt_text=_make_srt(120), model='m', apikey='k',
            llm_caller=self._caller(log), max_workers=1,
        )
        reduce_calls = [c for c in log
                        if c['system'] == MOVIE_REDUCE_SYSTEM_PROMPT]
        self.assertEqual(len(reduce_calls), 1)
        # Reduce payload must carry the turning-point header.
        self.assertIn('关键转折提示', reduce_calls[0]['srt'])
        self.assertEqual(extract_timestamps(out), [[0, 10000], [20000, 30000]])

    def test_movie_folds_when_many_cards(self):
        log = []
        # 2000 cues -> dozens of chunks -> more than the 40-card fold limit.
        movie_llm_inference(
            srt_text=_make_srt(2000), model='m', apikey='k',
            llm_caller=self._caller(log), max_workers=1,
        )
        fold_calls = [c for c in log if c['system'] == FOLD_SYSTEM_PROMPT]
        reduce_calls = [c for c in log
                        if c['system'] == MOVIE_REDUCE_SYSTEM_PROMPT]
        self.assertGreater(len(fold_calls), 0)
        # The reduce stage sees at most the fold limit worth of cards.
        self.assertLessEqual(reduce_calls[0]['srt'].count('摘要:'), 40)


if __name__ == '__main__':
    unittest.main()
