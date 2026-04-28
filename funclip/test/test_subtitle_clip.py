#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""Tests for the subtitle-driven AI clipping pipeline.

Covers the SRT parser and the LLM-output parsing used by AI Clip,
without requiring funasr/moviepy or a real LLM call.
"""

import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FUNCLIP_DIR = os.path.dirname(THIS_DIR)
if FUNCLIP_DIR not in sys.path:
    sys.path.insert(0, FUNCLIP_DIR)

from utils.subtitle_utils import parse_srt, generate_srt_clip  # noqa: E402
from utils.trans_utils import extract_timestamps  # noqa: E402


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


if __name__ == '__main__':
    unittest.main()
