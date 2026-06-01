"""正向最大匹配 (FMM)。"""
from __future__ import annotations


def fmm(sentence: str, dictionary: set[str], max_len: int) -> list[str]:
    result: list[str] = []
    index = 0
    n = len(sentence)
    while index < n:
        matched = False
        size_limit = min(max_len, n - index)
        for size in range(size_limit, 0, -1):
            piece = sentence[index : index + size]
            if piece in dictionary:
                result.append(piece)
                index += size
                matched = True
                break
        if not matched:
            result.append(sentence[index])
            index += 1
    return result
