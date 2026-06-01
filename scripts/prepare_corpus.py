#!/usr/bin/env python3
"""
从人民日报语料随机抽取训练句（默认 10000 行）。
--num_lines 可指定行数（如 1000 对比实验）。
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logger_util import journal, log

RAW = ROOT / "data" / "raw" / "pku_train.txt"
RAW_CANDIDATES = [
    RAW,
    ROOT / "data" / "raw" / "19980101-train.txt",
    ROOT / "data" / "raw" / "pku199801.txt",
    ROOT / "data" / "raw" / "1998-01-2003版-带音.txt",
]


def _read_file_with_encoding(file_path: Path) -> list[str]:
    """尝试多种编码读取文件。"""
    for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
        try:
            with file_path.open(encoding=encoding, errors='strict') as f:
                return f.readlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 最后尝试忽略错误
    with file_path.open(encoding='utf-8', errors='ignore') as f:
        return f.readlines()

def _find_raw_corpus() -> Path | None:
    for p in RAW_CANDIDATES:
        if p.exists() and p.stat().st_size > 10000:
            return p
    raw = ROOT / "data" / "raw"
    if raw.exists():
        for p in raw.glob("*.txt"):
            if p.name in ("jieba_dict.txt", "README_DOWNLOAD.txt"):
                continue
            if p.stat().st_size > 500_000:
                return p
    return None
OUT = ROOT / "data" / "corpus" / "train_corpus.txt"
DEFAULT_NUM_LINES = 10000


def parse_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    
    # 去除注音标记 {xxx}
    line = re.sub(r'\{[^}]*\}', '', line)
    
    if "/" in line and " " not in line:
        words = [w.split("/")[0] for w in line.split() if w]
    elif "/" in line:
        words = []
        for token in line.split():
            words.append(token.split("/")[0])
    else:
        words = line.split()
    
    # 过滤掉文档ID（如 19980108-01-002-003）和空词
    words = [w for w in words if w and w not in {" ", "\t"} and not re.match(r'^\d{8}-\d{2}-\d{3}-\d{3}$', w)]
    
    if not words:
        return None
    return " ".join(words)


def generate_fallback(num: int, seed: int = 42) -> list[str]:
    """无 pku 语料时，从 jieba 词典高频词合成训练句。"""
    dict_path = ROOT / "data" / "raw" / "jieba_dict.txt"
    pool: list[str] = []
    if dict_path.exists():
        with dict_path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    w, freq = parts[0], int(parts[1]) if parts[1].isdigit() else 0
                    if 2 <= len(w) <= 7 and freq >= 500:
                        if re.search(r"[\u4e00-\u9fff]", w):
                            pool.append(w)
                if len(pool) >= 5000:
                    break
    if len(pool) < 100:
        pool = list("中国科学院北京市自然语言处理技术研究发展".replace("", " "))  # noqa
        pool = ["中国", "科学院", "北京", "市", "自然语言", "处理", "技术", "研究", "发展"]
    rng = random.Random(seed)
    lines = []
    for _ in range(num):
        k = rng.randint(8, 18)
        words = [rng.choice(pool) for _ in range(k)]
        lines.append(" ".join(words))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_lines",
        type=int,
        default=DEFAULT_NUM_LINES,
        help="随机抽取训练句数（默认 10000）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(OUT))
    args = parser.parse_args()

    parsed: list[str] = []
    raw_path = _find_raw_corpus()
    if raw_path:
        lines = _read_file_with_encoding(raw_path)
        for line in lines:
            s = parse_line(line)
            if s:
                parsed.append(s)
        rng = random.Random(args.seed)
        if len(parsed) > args.num_lines:
            parsed = rng.sample(parsed, args.num_lines)
        source = f"{raw_path.name}_random_sample n={args.num_lines}"
    else:
        parsed = generate_fallback(args.num_lines, args.seed)
        source = f"synthetic_fallback n={args.num_lines}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parsed), encoding="utf-8")

    msg = f"训练语料 | {source} | 句数={len(parsed)} | 输出={out_path}"
    log("phase1_dict_corpus.txt", msg, phase="1", task="prepare_corpus", also_journal=True)
    journal(msg)
    print(msg)
    if not raw_path:
        print("提示：将人民日报1998分词语料放到 data/raw/pku_train.txt 可提升 HMM 质量")


if __name__ == "__main__":
    main()
