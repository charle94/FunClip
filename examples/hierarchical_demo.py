#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""Hierarchical (Map-Reduce) LLM Clipping — Demo Script
=======================================================

This script walks through the full **hierarchical LLM clipping** pipeline
introduced in FunClip's "分层叙事 | Hierarchical" prompt mode. It runs
entirely without a real video file or ASR model: a bundled sample SRT is
used as input, and you can choose between a *mock* LLM (no API key needed)
and a *real* LLM via a DeepSeek / OpenAI-compatible endpoint.

Usage
-----

# Run with the built-in mock LLM (no API key required)
python examples/hierarchical_demo.py

# Run with a real LLM (DeepSeek Chat example)
python examples/hierarchical_demo.py --model deepseek-chat --apikey <YOUR_KEY>

# Inspect the map-stage cards before the reduce step
python examples/hierarchical_demo.py --show-cards

What this script demonstrates
------------------------------
1. Parsing an SRT string with ``parse_srt`` (tolerant of common quirks).
2. Splitting the cue list into overlapping chunks with ``chunk_sentences``.
3. Map stage: each chunk → compact JSON narrative card.
4. Reduce stage: all cards → 3-8 final clip ranges.
5. Extracting machine-readable timestamps from the reduce output with
   ``extract_timestamps``, ready to feed into
   ``VideoClipper.video_clip(timestamp_list=...)``.

For a CLI-only end-to-end workflow (real video + SRT + LLM → clipped mp4)
see the README section "分层叙事模式 | Hierarchical Mode" or run:

    python funclip/videoclipper.py --stage 3 \\
        --file path/to/video.mp4 \\
        --srt_input path/to/subtitles.srt \\
        --llm_model deepseek-chat --apikey <YOUR_KEY> \\
        --prompt_mode hierarchical \\
        --output_dir ./output
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Make sure the funclip package is importable when running from the repo root
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
FUNCLIP_DIR = os.path.join(REPO_ROOT, "funclip")
if FUNCLIP_DIR not in sys.path:
    sys.path.insert(0, FUNCLIP_DIR)

from utils.subtitle_utils import parse_srt  # noqa: E402
from utils.trans_utils import extract_timestamps  # noqa: E402
from llm.hierarchical import (  # noqa: E402
    chunk_sentences,
    chunk_to_srt,
    chunk_time_range,
    parse_card,
    format_cards_for_reduce,
    hierarchical_llm_inference,
    MAP_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Sample SRT (60 cues, ~2 minutes; suitable for a 2-chunk demo run)
# ---------------------------------------------------------------------------
SAMPLE_SRT = """\
1
00:00:00,500 --> 00:00:02,100
大家好，欢迎来到今天的分享。

2
00:00:02,310 --> 00:00:04,200
今天我们要聊一个非常有趣的话题——

3
00:00:04,500 --> 00:00:07,000
如何用人工智能帮助我们更高效地剪辑视频。

4
00:00:07,400 --> 00:00:10,100
视频剪辑是一项耗时耗力的工作，

5
00:00:10,500 --> 00:00:13,200
但借助大语言模型，我们可以大幅降低这个门槛。

6
00:00:13,800 --> 00:00:16,400
首先让我们了解一下 FunClip 的基本架构。

7
00:00:16,900 --> 00:00:19,600
FunClip 集成了阿里巴巴开源的 Paraformer 模型

8
00:00:20,000 --> 00:00:22,500
用于高精度的语音识别与时间戳预测。

9
00:00:23,000 --> 00:00:26,100
识别完成后，用户可以直接输入想要保留的文字片段，

10
00:00:26,500 --> 00:00:29,300
系统会自动定位并裁剪出对应的视频段落。

11
00:00:30,000 --> 00:00:33,200
接下来，我们看一下与大语言模型结合的智能剪辑流程。

12
00:00:33,700 --> 00:00:37,100
在 LLM 智能裁剪模式中，系统将 SRT 字幕发送给大模型，

13
00:00:37,500 --> 00:00:41,000
请它按照短视频叙事结构识别出最有价值的片段。

14
00:00:41,500 --> 00:00:44,800
模型输出带有起止时间戳的片段列表，

15
00:00:45,200 --> 00:00:48,400
FunClip 解析时间戳后自动完成剪辑，全程无需人工干预。

16
00:00:49,000 --> 00:00:52,300
今天演示的核心场景是"分层叙事剪辑"——

17
00:00:52,800 --> 00:00:56,100
也就是本脚本展示的 Hierarchical 模式。

18
00:00:56,600 --> 00:01:00,200
对于超长字幕，单次 LLM 调用往往无法把握整体剧情。

19
00:01:00,700 --> 00:01:04,500
分层模式将字幕切分为若干重叠的块，

20
00:01:05,000 --> 00:01:09,000
每块单独发送给 LLM，获取一张"叙事摘要卡"。

21
00:01:09,500 --> 00:01:13,300
叙事摘要卡记录了该段字幕的时间范围、叙事角色、

22
00:01:13,800 --> 00:01:17,200
情绪强度，以及 1-2 句内容摘要。

23
00:01:17,700 --> 00:01:21,500
所有摘要卡被汇聚成一份简洁的"情节地图"，

24
00:01:22,000 --> 00:01:26,100
再由 LLM 基于全局视角挑选出最具戏剧张力的片段。

25
00:01:26,600 --> 00:01:30,300
这种 Map-Reduce 结构让模型能够同时关注局部细节

26
00:01:30,800 --> 00:01:34,500
与整体叙事脉络，大幅提升长视频的选片质量。

27
00:01:35,200 --> 00:01:39,100
下面进入实际演示环节。

28
00:01:39,600 --> 00:01:43,400
我们准备了一段约 60 条字幕的示例 SRT，

29
00:01:43,900 --> 00:01:47,700
涵盖开场介绍、技术讲解、案例演示和总结收尾四个段落。

30
00:01:48,200 --> 00:01:52,400
首先是第一步：解析 SRT，拆分为重叠的字幕块。

31
00:01:53,000 --> 00:01:57,200
Map 阶段将每个块独立发送给 LLM 做叙事分析。

32
00:01:57,800 --> 00:02:02,100
每个 LLM 调用都是独立的，因此可以并行执行，

33
00:02:02,600 --> 00:02:06,500
大幅缩短总等待时间，即使字幕非常长也不怕。

34
00:02:07,100 --> 00:02:11,400
Reduce 阶段将所有摘要卡拼接后发给 LLM，

35
00:02:11,900 --> 00:02:15,700
请它从全局视角选出 3 到 8 个最佳剪辑片段。

36
00:02:16,300 --> 00:02:20,600
输出格式严格遵循 N. [HH:MM:SS,mmm-HH:MM:SS,mmm] 简介，

37
00:02:21,100 --> 00:02:25,400
确保 FunClip 的时间戳提取器能够正确解析。

38
00:02:26,000 --> 00:02:30,300
接下来我们看一下真实的运行效果。

39
00:02:30,900 --> 00:02:35,200
在使用 Mock LLM 运行时，所有 LLM 调用都由本地函数模拟，

40
00:02:35,700 --> 00:02:39,900
无需任何 API Key 或网络连接。

41
00:02:40,500 --> 00:02:44,800
Mock LLM 会返回格式正确的占位输出，

42
00:02:45,300 --> 00:02:49,600
让你可以验证整个流程的正确性而无需消耗 Token。

43
00:02:50,200 --> 00:02:54,500
若要使用真实 LLM，只需传入 --model 和 --apikey 参数。

44
00:02:55,100 --> 00:02:59,400
脚本会自动调用对应的模型接口完成真正的语义分析。

45
00:03:00,000 --> 00:03:04,300
现在进入本次分享最核心的案例——

46
00:03:04,800 --> 00:03:09,100
一段来自某技术峰会的演讲，时长约 2 小时。

47
00:03:09,700 --> 00:03:14,000
如果直接将全部字幕发给 LLM，

48
00:03:14,500 --> 00:03:18,800
Token 消耗巨大，且模型往往遗漏中间的高潮片段。

49
00:03:19,400 --> 00:03:23,700
使用 Hierarchical 模式后，我们将字幕分成 12 个块，

50
00:03:24,300 --> 00:03:28,600
并行发起 12 次 Map 调用，每次仅消耗少量 Token。

51
00:03:29,200 --> 00:03:33,500
最终的 Reduce 调用只接收 12 张摘要卡，

52
00:03:34,000 --> 00:03:38,300
远比原始字幕短，成本与延迟都大幅降低。

53
00:03:38,900 --> 00:03:43,200
剪辑质量方面，分层模式精准识别出了演讲的三个高潮：

54
00:03:43,800 --> 00:03:48,100
产品发布、现场 Demo 和 Q&A 精华片段。

55
00:03:48,700 --> 00:03:53,000
而单次调用仅选出了开头和结尾两段，完全遗漏了中间高潮。

56
00:03:53,600 --> 00:03:57,900
这正是 Hierarchical 模式的核心价值所在。

57
00:03:58,500 --> 00:04:02,800
好，以上就是今天的全部内容。

58
00:04:03,400 --> 00:04:07,700
如果你觉得有帮助，欢迎给 FunClip 仓库点个 Star！

59
00:04:08,300 --> 00:04:12,600
我们会持续迭代，提供更多好用的 AI 视频剪辑功能。

60
00:04:13,200 --> 00:04:17,000
感谢收看，下次见！
"""


# ---------------------------------------------------------------------------
# Mock LLM caller (no network / API key needed)
# ---------------------------------------------------------------------------
def _mock_llm_caller(system_content, user_content, srt_text, model, apikey):
    """Simulate LLM responses for demo purposes.

    * Map stage: returns a plausible narrative card JSON derived from the
      first and last timestamp found in the chunk SRT.
    * Reduce stage: returns two hardcoded clip ranges whose timestamps are
      guaranteed to fall within the sample SRT, so ``extract_timestamps``
      always succeeds.
    """
    import re as _re
    if system_content == MAP_SYSTEM_PROMPT:
        # Extract first start and last end timestamps from the chunk SRT.
        starts = _re.findall(r'(\d{2}:\d{2}:\d{2},\d{3}) -->', srt_text)
        ends = _re.findall(r'--> (\d{2}:\d{2}:\d{2},\d{3})', srt_text)
        s = starts[0] if starts else "00:00:00,000"
        e = ends[-1] if ends else "00:00:01,000"
        card = {
            "range": "[{}-{}]".format(s, e),
            "role": "setup",
            "intensity": 3,
            "summary": "（Mock）本段覆盖 {} → {}".format(s, e),
        }
        return json.dumps(card, ensure_ascii=False)
    # Reduce stage: return two clip lines using real timestamps from the SRT.
    return (
        "1. [00:00:13,800-00:00:48,400] 技术介绍：FunClip 架构与 LLM 智能裁剪流程\n"
        "2. [00:01:35,200-00:02:25,400] 核心原理：分层 Map-Reduce 选片机制详解\n"
        "3. [00:03:38,900-00:04:17,000] 效果对比与总结"
    )


# ---------------------------------------------------------------------------
# Real LLM caller (wraps existing FunClip llm helpers)
# ---------------------------------------------------------------------------
def _real_llm_caller(model, apikey):
    """Return a real LLM caller bound to the given model/apikey."""
    from llm.openai_api import openai_call
    from llm.qwen_api import call_qwen_model
    from llm.g4f_openai_api import g4f_openai_call

    def caller(system_content, user_content, srt_text, _model, _apikey):
        payload = user_content + "\n" + srt_text
        if _model.startswith("qwen"):
            return call_qwen_model(_apikey, _model, payload, system_content)
        if _model.startswith(("gpt", "moonshot", "deepseek")):
            return openai_call(_apikey, _model, system_content, payload)
        if _model.startswith("g4f"):
            return g4f_openai_call("-".join(_model.split("-")[1:]), system_content, payload)
        raise ValueError("Unsupported model prefix: {}".format(_model))

    # Bind model/apikey so the caller matches the expected signature.
    def bound_caller(system_content, user_content, srt_text, model, apikey):
        return caller(system_content, user_content, srt_text, model, apikey)

    return bound_caller


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def run_demo(model=None, apikey=None, show_cards=False, window=25, overlap=4):
    """Execute the full hierarchical clipping demo.

    Parameters
    ----------
    model, apikey : str or None
        When provided, use a real LLM. Otherwise use the built-in mock.
    show_cards : bool
        Print each map-stage narrative card after the Map phase.
    window, overlap : int
        Chunk size knobs. Defaults are small so the demo runs quickly on
        the 60-cue sample SRT.
    """
    sep = "-" * 68
    use_mock = model is None

    print(sep)
    print("FunClip — Hierarchical LLM Clipping Demo")
    print("LLM: {}".format("Mock (no API key)" if use_mock else model))
    print("Chunk window={}, overlap={}".format(window, overlap))
    print(sep)

    # ------------------------------------------------------------------
    # Step 1: Parse SRT
    # ------------------------------------------------------------------
    print("\n[1/4] Parsing sample SRT…")
    sentences, normalized_srt = parse_srt(SAMPLE_SRT)
    print("      Parsed {} cues.".format(len(sentences)))

    # ------------------------------------------------------------------
    # Step 2: Split into chunks
    # ------------------------------------------------------------------
    print("\n[2/4] Splitting into overlapping chunks…")
    chunks = chunk_sentences(sentences, window=window, overlap=overlap)
    print("      {} chunk(s) created (window={}, overlap={}).".format(
        len(chunks), window, overlap))
    for i, chunk in enumerate(chunks):
        s_ms, e_ms = chunk_time_range(chunk)
        print("      Chunk {:2d}: {} cues  [{:.1f}s – {:.1f}s]".format(
            i + 1, len(chunk), s_ms / 1000, e_ms / 1000))

    # ------------------------------------------------------------------
    # Step 3: Run hierarchical_llm_inference
    # ------------------------------------------------------------------
    print("\n[3/4] Running hierarchical LLM inference…")
    if use_mock:
        llm_caller = _mock_llm_caller
    else:
        llm_caller = _real_llm_caller(model, apikey)

    llm_result = hierarchical_llm_inference(
        srt_text=normalized_srt,
        model=model or "mock",
        apikey=apikey or "",
        llm_caller=llm_caller,
        window=window,
        overlap=overlap,
        max_workers=1,  # Sequential for readable demo output.
    )

    # ------------------------------------------------------------------
    # Optional: show narrative cards
    # ------------------------------------------------------------------
    if show_cards and len(chunks) > 1:
        print("\n      ── Map-stage narrative cards ──")
        # Re-run Map locally just to display cards (results already
        # consumed by hierarchical_llm_inference above, so we replay).
        for i, chunk in enumerate(chunks):
            s_ms, e_ms = chunk_time_range(chunk)
            fallback_range = "[{}-{}]".format(
                "{:02d}:{:02d}:{:02d},{:03d}".format(
                    s_ms // 3600000, (s_ms // 60000) % 60,
                    (s_ms // 1000) % 60, s_ms % 1000),
                "{:02d}:{:02d}:{:02d},{:03d}".format(
                    e_ms // 3600000, (e_ms // 60000) % 60,
                    (e_ms // 1000) % 60, e_ms % 1000),
            )
            raw = llm_caller(
                MAP_SYSTEM_PROMPT, "", chunk_to_srt(chunk),
                model or "mock", apikey or "")
            card = parse_card(raw, fallback_range)
            print("      Card {:2d}: role={:<8s} intensity={} range={}".format(
                i + 1, card["role"], card["intensity"], card["range"]))
            if card["summary"]:
                print("              {}".format(card["summary"]))

    print("\n      LLM result:\n")
    for line in llm_result.splitlines():
        print("      " + line)

    # ------------------------------------------------------------------
    # Step 4: Extract timestamps
    # ------------------------------------------------------------------
    print("\n[4/4] Extracting timestamps from LLM result…")
    ts_list = extract_timestamps(llm_result)
    if not ts_list:
        print("      ⚠  No timestamps extracted — LLM output may not match "
              "the expected format.")
    else:
        print("      {} clip segment(s) ready for VideoClipper:".format(len(ts_list)))
        for i, (start_ms, end_ms) in enumerate(ts_list, 1):
            print("      Segment {:2d}: {:.1f}s – {:.1f}s  (duration {:.1f}s)".format(
                i, start_ms / 1000, end_ms / 1000, (end_ms - start_ms) / 1000))

    print("\n" + sep)
    print("Demo complete.")
    print()
    print("Next step — clip an actual video:")
    print()
    print("  python funclip/videoclipper.py --stage 3 \\")
    print("      --file path/to/video.mp4 \\")
    print("      --srt_input path/to/subtitles.srt \\")
    print("      --llm_model deepseek-chat --apikey <YOUR_KEY> \\")
    print("      --prompt_mode hierarchical \\")
    print("      --output_dir ./output")
    print(sep)

    return ts_list


def main():
    parser = argparse.ArgumentParser(
        description="FunClip Hierarchical LLM Clipping — Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="LLM model name (e.g. deepseek-chat, qwen-plus, gpt-4-turbo). "
             "Omit to use the built-in mock (no API key required).",
    )
    parser.add_argument(
        "--apikey", "-k", default=None,
        help="API key for the chosen LLM. Not required when using the mock.",
    )
    parser.add_argument(
        "--show-cards", action="store_true",
        help="Print each map-stage narrative card after the Map phase.",
    )
    parser.add_argument(
        "--window", type=int, default=25,
        help="Chunk size in SRT cues (default: 25). "
             "Use a larger value for real long-video subtitles.",
    )
    parser.add_argument(
        "--overlap", type=int, default=4,
        help="Overlap between consecutive chunks in cues (default: 4).",
    )
    args = parser.parse_args()

    if args.model and not args.apikey and not args.model.startswith("g4f"):
        parser.error("--apikey is required when --model is specified "
                     "(g4f-* models do not need an API key).")

    run_demo(
        model=args.model,
        apikey=args.apikey,
        show_cards=args.show_cards,
        window=args.window,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
