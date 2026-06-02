"""词典加载与 MAX_LEN 推断（上限 7）。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT_DIR = ROOT / "data" / "dict"
MAX_LEN_CAP = 7


def _read_word_file(path: Path) -> dict[str, int]:
    words: dict[str, int] = {}
    if not path.exists():
        return words
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            w = parts[0]
            freq = int(parts[1]) if len(parts) > 1 else 1
            words[w] = words.get(w, 0) + freq
    return words


def load_dictionary(
    include_ambiguity: bool = True,
    include_oov: bool = True,
) -> tuple[dict[str, int], int]:
    words = _read_word_file(DICT_DIR / "dict_base.txt")
    
    core_words = _read_word_file(DICT_DIR / "dict_core.txt")
    for w, freq in core_words.items():
        words[w] = words.get(w, 0) + freq * 3
    
    if include_ambiguity:
        ambiguity_words = _read_word_file(DICT_DIR / "dict_ambiguity.txt")
        for w, freq in ambiguity_words.items():
            words[w] = words.get(w, 0) + freq * 2
    
    if include_oov:
        oov_words = _read_word_file(DICT_DIR / "dict_oov.txt")
        for w, freq in oov_words.items():
            words[w] = words.get(w, 0) + freq
    
    if not words:
        raise FileNotFoundError(
            "词典为空，请先运行 scripts/build_dict.py 生成 data/dict/dict_base.txt"
        )
    max_len = min(max(len(w) for w in words), MAX_LEN_CAP)
    return words, max_len
