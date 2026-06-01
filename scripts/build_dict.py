#!/usr/bin/env python3
"""从 THUOCL / jieba dict 构建 dict_base.txt（>=2000 词）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logger_util import journal, log

RAW_THUOCL = ROOT / "data" / "raw" / "thuocl"
RAW_JIEBA = ROOT / "data" / "raw" / "jieba_dict.txt"
OUT = ROOT / "data" / "dict" / "dict_base.txt"
MIN_WORDS = 2000


def _valid_word(w: str) -> bool:
    if not w or len(w) < 2:
        return False
    if len(w) > 7:
        return False
    # 至少含一个汉字
    return any("\u4e00" <= c <= "\u9fff" for c in w)


def load_thuocl() -> set[str]:
    words: set[str] = set()
    if not RAW_THUOCL.exists():
        return words
    for path in RAW_THUOCL.glob("*.txt"):
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip().split()[0] if line.strip() else ""
                if _valid_word(w):
                    words.add(w)
    return words


def load_jieba_raw(limit: int = 15000) -> set[str]:
    words: set[str] = set()
    if not RAW_JIEBA.exists():
        return words
    with RAW_JIEBA.open(encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= limit * 3:
                break
            parts = line.strip().split()
            if not parts:
                continue
            w = parts[0]
            if _valid_word(w):
                words.add(w)
            if len(words) >= limit:
                break
    return words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = parser.parse_args()

    words = load_thuocl()
    sources = []
    if words:
        sources.append(f"THUOCL:{len(words)}")
    if len(words) < args.min_words:
        jieba_words = load_jieba_raw()
        words |= jieba_words
        sources.append(f"jieba_dict_fallback:{len(jieba_words)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sorted_words = sorted(words)
    OUT.write_text("\n".join(sorted_words), encoding="utf-8")

    msg = (
        f"dict_base 生成完成 | 词数={len(sorted_words)} | 来源={sources} | "
        f"路径={OUT}"
    )
    log("phase1_dict_corpus.txt", msg, phase="1", task="build_dict", also_journal=True)
    journal(msg)
    print(msg)
    if len(sorted_words) < args.min_words:
        print(
            f"警告：词数不足 {args.min_words}，请下载 THUOCL（IT词汇/财经/成语/地名/动物/饮食）"
            f" 到 data/raw/thuocl/"
        )


if __name__ == "__main__":
    main()
