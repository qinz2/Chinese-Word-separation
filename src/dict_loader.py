"""词典加载与 MAX_LEN 推断（上限 7）。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT_DIR = ROOT / "data" / "dict"
MAX_LEN_CAP = 7


def _read_word_file(path: Path) -> set[str]:
    words: set[str] = set()
    if not path.exists():
        return words
    with path.open(encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith("#"):
                words.add(w)
    return words


def load_dictionary(
    include_ambiguity: bool = True,
    include_oov: bool = True,
) -> tuple[set[str], int]:
    words = _read_word_file(DICT_DIR / "dict_base.txt")
    words |= _read_word_file(DICT_DIR / "dict_core.txt")
    if include_ambiguity:
        words |= _read_word_file(DICT_DIR / "dict_ambiguity.txt")
    if include_oov:
        words |= _read_word_file(DICT_DIR / "dict_oov.txt")
    if not words:
        raise FileNotFoundError(
            "词典为空，请先运行 scripts/build_dict.py 生成 data/dict/dict_base.txt"
        )
    max_len = min(max(len(w) for w in words), MAX_LEN_CAP)
    return words, max_len
