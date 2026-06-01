#!/usr/bin/env python3
"""训练 HMM 参数并保存；支持 --num_lines 指定语料规模。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hmm import train_and_save

DEFAULT_NUM_LINES = 10000
from src.logger_util import journal, log

CORPUS = ROOT / "data" / "corpus" / "train_corpus.txt"
MODEL = ROOT / "models" / "hmm_params.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_lines",
        type=int,
        default=DEFAULT_NUM_LINES,
        help="训练语料行数（默认 10000，可改为 1000 做对比实验）",
    )
    parser.add_argument("--corpus", type=str, default=str(CORPUS))
    parser.add_argument("--model", type=str, default=str(MODEL))
    args = parser.parse_args()

    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_corpus.py"),
            "--num_lines",
            str(args.num_lines),
        ],
        check=True,
        cwd=str(ROOT),
    )
    corpus_path = Path(args.corpus)

    params = train_and_save(corpus_path, Path(args.model))
    vocab_size = len(params["vocab"])
    msg = (
        f"HMM 训练完成 | 语料={corpus_path} | 句数目标={args.num_lines} | "
        f"|V|={vocab_size} | 平滑=转移分母count(s)+4,发射分母count(s)+|V| | 模型={args.model}"
    )
    log("phase3_hmm_baseline.txt", msg, phase="3", task="train_hmm", also_journal=True)
    log("phase4_bimm_hmm_opt.txt", msg, phase="4", task="hmm_trained", also_journal=True)
    journal(msg)
    print(msg)


if __name__ == "__main__":
    main()
