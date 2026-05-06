#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
import re

def time_convert(ms):
    ms = int(ms)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    h = "00" if h == 0 else str(h)
    mi = "00" if mi == 0 else str(mi)
    s = "00" if s == 0 else str(s)
    tail = str(tail).zfill(3)
    if len(h) == 1: h = '0' + h
    if len(mi) == 1: mi = '0' + mi
    if len(s) == 1: s = '0' + s
    return "{}:{}:{},{}".format(h, mi, s, tail)

def str2list(text):
    pattern = re.compile(r'[\u4e00-\u9fff]|[\w-]+', re.UNICODE)
    elements = pattern.findall(text)
    return elements

class Text2SRT():
    def __init__(self, text, timestamp, offset=0):
        self.token_list = text
        self.timestamp = timestamp
        start, end = timestamp[0][0] - offset, timestamp[-1][1] - offset
        self.start_sec, self.end_sec = start, end
        self.start_time = time_convert(start)
        self.end_time = time_convert(end)
    def text(self):
        if isinstance(self.token_list, str):
            return self.token_list.rstrip("、。，")
        else:
            res = ""
            for word in self.token_list:
                if '\u4e00' <= word <= '\u9fff':
                    res += word
                else:
                    res += " " + word
            return res.lstrip().rstrip("、。，")
    def srt(self, acc_ost=0.0):
        return "{} --> {}\n{}\n".format(
            time_convert(self.start_sec+acc_ost*1000),
            time_convert(self.end_sec+acc_ost*1000), 
            self.text())
    def time(self, acc_ost=0.0):
        return (self.start_sec/1000+acc_ost, self.end_sec/1000+acc_ost)


def generate_srt(sentence_list):
    srt_total = ''
    for i, sent in enumerate(sentence_list):
        t2s = Text2SRT(sent['text'], sent['timestamp'])
        if 'spk' in sent:
            srt_total += "{}  spk{}\n{}".format(i + 1, sent['spk'], t2s.srt())
        else:
            srt_total += "{}\n{}\n".format(i + 1, t2s.srt())
    return srt_total

def generate_srt_clip(sentence_list, start, end, begin_index=0, time_acc_ost=0.0):
    start, end = int(start * 1000), int(end * 1000)
    srt_total = ''
    cc = 1 + begin_index
    subs = []
    for _, sent in enumerate(sentence_list):
        if isinstance(sent['text'], str):
            sent['text'] = str2list(sent['text'])
        if sent['timestamp'][-1][1] <= start:
            # print("CASE0")
            continue
        if sent['timestamp'][0][0] >= end:
            # print("CASE4")
            break
        # parts in between
        if (sent['timestamp'][-1][1] <= end and sent['timestamp'][0][0] > start) or (sent['timestamp'][-1][1] == end and sent['timestamp'][0][0] == start):
            # print("CASE1"); import pdb; pdb.set_trace()
            t2s = Text2SRT(sent['text'], sent['timestamp'], offset=start)
            srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
            subs.append((t2s.time(time_acc_ost), t2s.text()))
            cc += 1
            continue
        if sent['timestamp'][0][0] <= start:
            # print("CASE2"); import pdb; pdb.set_trace()
            if not sent['timestamp'][-1][1] > end:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        break
                _text = sent['text'][j:]
                _ts = sent['timestamp'][j:]
            else:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        _start = j
                        break
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > end:
                        _end = j
                        break
                # _text = " ".join(sent['text'][_start:_end])
                _text = sent['text'][_start:_end]
                _ts = sent['timestamp'][_start:_end]
            if len(ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append((t2s.time(time_acc_ost), t2s.text()))
                cc += 1
            continue
        if sent['timestamp'][-1][1] > end:
            # print("CASE3"); import pdb; pdb.set_trace()
            for j, ts in enumerate(sent['timestamp']):
                if ts[1] > end:
                    break
            _text = sent['text'][:j]
            _ts = sent['timestamp'][:j]
            if len(_ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append(
                    (t2s.time(time_acc_ost), t2s.text())
                    )
                cc += 1
            continue
    return srt_total, subs, cc


def _srt_time_to_ms(t):
    """Convert an SRT timestamp string (HH:MM:SS,mmm or HH:MM:SS.mmm) to milliseconds.

    Tolerant of variants seen in the wild:
      * ``,`` or ``.`` as the millisecond separator.
      * ``H:MM:SS,mmm`` (single digit hour).
      * 1-3 digit milliseconds (left padded to 3).
    """
    t = t.strip().replace('.', ',')
    m = re.match(r'^(\d{1,2}):(\d{2}):(\d{2})[,:](\d{1,3})$', t)
    if not m:
        raise ValueError("Invalid SRT timestamp: {}".format(t))
    h, mi, s, ms = m.groups()
    # SRT milliseconds are zero-padded on the LEFT (",5" means 5ms, ",50"
    # means 50ms, ",500" means 500ms). The same convention is produced by
    # ``time_convert`` (``zfill(3)``) above.
    ms = ms.zfill(3)
    return ((int(h) * 60 + int(mi)) * 60 + int(s)) * 1000 + int(ms)


def parse_srt(srt_text):
    """Parse an SRT subtitle string.

    Returns a tuple ``(sentences, normalized_srt)`` where:

    * ``sentences`` is a list of dicts compatible with ``generate_srt_clip``,
      each having keys ``text`` (list with a single string token), ``timestamp``
      (``[[start_ms, end_ms]]``), ``start`` (ms), ``end`` (ms) and
      ``raw_text`` (joined original text for the cue).
    * ``normalized_srt`` is a re-rendered SRT string with sequential indices
      starting from 0 and ``HH:MM:SS,mmm`` timestamps, matching the format
      expected by the LLM prompts in :mod:`funclip.llm.demo_prompt`.

    The parser is tolerant of common variants: UTF-8 BOM, ``\\r\\n`` line
    endings, blank lines between cues, missing or duplicated sequence
    numbers, and ``,``/``.`` millisecond separators.
    """
    if srt_text is None:
        return [], ''
    # Strip BOM and normalise newlines.
    if srt_text.startswith('\ufeff'):
        srt_text = srt_text[1:]
    srt_text = srt_text.replace('\r\n', '\n').replace('\r', '\n')

    time_re = re.compile(
        r'(\d{1,2}:\d{2}:\d{2}[,\.:]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.:]\d{1,3})'
    )

    lines = srt_text.split('\n')
    sentences = []
    i = 0
    n = len(lines)
    while i < n:
        # Skip empty lines and stray sequence-number lines.
        while i < n and not time_re.search(lines[i]):
            i += 1
        if i >= n:
            break
        m = time_re.search(lines[i])
        try:
            start_ms = _srt_time_to_ms(m.group(1))
            end_ms = _srt_time_to_ms(m.group(2))
        except ValueError:
            i += 1
            continue
        i += 1
        text_parts = []
        while i < n and lines[i].strip() != '' and not time_re.search(lines[i]):
            text_parts.append(lines[i].rstrip())
            i += 1
        text = '\n'.join(text_parts).strip()
        if end_ms <= start_ms:
            # Skip degenerate cues but do not abort.
            continue
        sentences.append({
            'text': [text] if text else [''],
            'timestamp': [[start_ms, end_ms]],
            'start': start_ms,
            'end': end_ms,
            'raw_text': text,
        })

    normalized_lines = []
    for idx, sent in enumerate(sentences):
        normalized_lines.append(str(idx))
        normalized_lines.append("{} --> {}".format(
            time_convert(sent['start']), time_convert(sent['end'])))
        normalized_lines.append(sent['raw_text'])
        normalized_lines.append('')
    normalized_srt = '\n'.join(normalized_lines)
    return sentences, normalized_srt

