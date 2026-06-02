"""逆向最大匹配 (BMM)。"""
from __future__ import annotations


def bmm(sentence: str, dictionary: dict[str, int], max_len: int) -> list[str]:
    result: list[str] = []
    index = len(sentence)
    while index > 0:
        matched = False
        size_limit = min(max_len, index)
        for size in range(size_limit, 0, -1):
            start = index - size
            piece = sentence[start:index]
            if piece in dictionary:
                result.insert(0, piece)
                index -= size
                matched = True
                break
        if not matched:
            result.insert(0, sentence[index - 1])
            index -= 1
    return result
