"""双向最大匹配 (BiMM)。"""
from __future__ import annotations

from .bmm import bmm
from .fmm import fmm


def _single_char_count(words: list[str]) -> int:
    return sum(1 for w in words if len(w) == 1)


def bimm(sentence: str, dictionary: set[str], max_len: int) -> list[str]:
    f_result = fmm(sentence, dictionary, max_len)
    b_result = bmm(sentence, dictionary, max_len)
    if f_result == b_result:
        return f_result
    if len(f_result) != len(b_result):
        return f_result if len(f_result) < len(b_result) else b_result
    f_singles = _single_char_count(f_result)
    b_singles = _single_char_count(b_result)
    if f_singles != b_singles:
        return f_result if f_singles < b_singles else b_result
    return b_result
