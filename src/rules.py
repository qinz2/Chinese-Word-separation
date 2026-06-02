"""Layer3：数字、日期、英文、中文数字（句级 + 词级）。"""
from __future__ import annotations

import re

RE_ARABIC_NUM = re.compile(r"\d+(?:\.\d+)?")

# 日期规则（放在最前面，优先级最高）
RE_DATE = re.compile(
    r"\d{4}年\d{1,2}月\d{1,2}日|"
    r"\d{4}年\d{1,2}月|"
    r"\d{4}年|"
    r"\d{4}-\d{1,2}-\d{1,2}|"
    r"\d{4}/\d{1,2}/\d{1,2}"
)

# 英文规则（其次）
RE_ENGLISH = re.compile(r"[A-Za-z][A-Za-z0-9+#_@.]*")

# 中文数字规则（扩展支持更多表达）
# 支持：零一二三四五六七八九十百千万亿〇两壹贰叁肆伍陆柒捌玖拾佰仟萬亿
# 支持：成（如五成）、倍、分之（如三分之一）、点（如三点五）
RE_CN_NUM = re.compile(
    r"[零一二三四五六七八九十百千万亿〇两壹贰叁肆伍陆柒捌玖拾佰仟萬亿]+"
    r"(?:(?:[成倍分])|(?:分之[零一二三四五六七八九十百千万亿〇两壹贰叁肆伍陆柒捌玖拾佰仟萬亿]+))?"
    r"(?:点[零一二三四五六七八九十]+)?"
)

# 混合实体规则（最后匹配，避免与日期冲突）
RE_MIXED = re.compile(r"[A-Za-z0-9]+(?:[年月日时分秒])?")


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s >= merged[-1][1]:
            merged.append((s, e))
        elif e > merged[-1][1]:
            merged[-1] = (merged[-1][0], e)
    return merged


def _collect_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    # 按优先级顺序匹配：日期 > 英文 > 阿拉伯数字 > 中文数字 > 混合实体
    # 日期优先，避免被混合实体规则错误匹配
    for pat in (RE_DATE, RE_ENGLISH, RE_ARABIC_NUM, RE_CN_NUM, RE_MIXED):
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return _merge_spans(spans)


def _is_pure_chinese(text: str) -> bool:
    return bool(text) and all("\u4e00" <= c <= "\u9fff" for c in text)


def _split_token(text: str) -> list[str]:
    if not text:
        return []
    if _is_pure_chinese(text) and not RE_CN_NUM.fullmatch(text):
        return [text]
    spans = _collect_spans(text)
    if not spans:
        return [text]
    result: list[str] = []
    pos = 0
    for s, e in spans:
        if s > pos:
            mid = text[pos:s]
            if mid:
                result.append(mid)
        result.append(text[s:e])
        pos = e
    if pos < len(text):
        tail = text[pos:]
        if tail:
            result.append(tail)
    return result if result else [text]


def merge_entities(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        out.extend(_split_token(tok))
    return out


def apply_sentence_entities(sentence: str, tokens: list[str]) -> list[str]:
    """句级实体切分：优先提取日期等，其余区间用 tokens 顺序填充。"""
    if "".join(tokens) != sentence:
        return merge_entities(tokens)
    spans = _collect_spans(sentence)
    if not spans:
        return tokens
    result: list[str] = []
    pos = 0
    ti = 0
    buf = ""
    for s, e in spans:
        while pos < s and ti < len(tokens):
            piece = tokens[ti]
            if pos + len(piece) <= s:
                result.append(piece)
                pos += len(piece)
                ti += 1
            else:
                break
        result.append(sentence[s:e])
        pos = e
    while ti < len(tokens):
        result.append(tokens[ti])
        ti += 1
    return result
