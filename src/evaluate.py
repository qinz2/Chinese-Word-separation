"""批量评测、计时、jieba 对比（jieba 仅在此模块使用）。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import jieba

from .bimm import bimm
from .bmm import bmm
from .dict_loader import load_dictionary
from .fmm import fmm
from .hmm import baseline_params, load_params, segment as hmm_segment
from .hybrid import hybrid_segment
from .logger_util import log

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "models" / "hmm_params.json"

ALGORITHMS = [
    "FMM",
    "BMM",
    "BiMM",
    "HMM_baseline",
    "HMM_trained",
    "Hybrid",
]


def load_test_sentences(name: str) -> list[str]:
    path = TEST_DIR / name
    lines = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def _match_rate(a: list[str], b: list[str]) -> bool:
    return a == b


def run_segmentation(
    sentence: str,
    algo: str,
    dictionary: set[str],
    max_len: int,
    hmm_trained: dict[str, Any],
    hmm_base: dict[str, Any],
) -> list[str]:
    if algo == "FMM":
        return fmm(sentence, dictionary, max_len)
    if algo == "BMM":
        return bmm(sentence, dictionary, max_len)
    if algo == "BiMM":
        return bimm(sentence, dictionary, max_len)
    if algo == "HMM_baseline":
        return hmm_segment(sentence, hmm_base)
    if algo == "HMM_trained":
        return hmm_segment(sentence, hmm_trained)
    if algo == "Hybrid":
        return hybrid_segment(sentence, dictionary, max_len, hmm_trained)
    raise ValueError(algo)


def benchmark_all() -> dict[str, Any]:
    dictionary, max_len = load_dictionary()
    hmm_base = apply_baseline_compat(baseline_params())
    hmm_trained = load_params(MODEL_PATH) if MODEL_PATH.exists() else hmm_base

    test_files = {
        "basic": "basic.txt",
        "ambiguity": "ambiguity_20.txt",
        "oov": "oov_15.txt",
        "long": "long_10.txt",
    }
    stats: dict[str, Any] = {"timing": {}, "jieba_agree": {}, "hybrid_vs_bimm_ambiguity": {}}

    seg_dir = RESULTS_DIR / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)

    for key, fname in test_files.items():
        sentences = load_test_sentences(fname)
        out_lines = ["sentence\talgorithm\tresult\ttime_ms\tjieba_match"]
        timing: dict[str, float] = {a: 0.0 for a in ALGORITHMS}
        agree = {a: 0 for a in ALGORITHMS}
        total = len(sentences)

        for sent in sentences:
            jieba_res = list(jieba.cut(sent))
            for algo in ALGORITHMS:
                t0 = time.perf_counter()
                res = run_segmentation(
                    sent, algo, dictionary, max_len, hmm_trained, hmm_base
                )
                elapsed = (time.perf_counter() - t0) * 1000
                timing[algo] += elapsed
                match = _match_rate(res, jieba_res)
                if match:
                    agree[algo] += 1
                out_lines.append(
                    f"{sent}\t{algo}\t{'/'.join(res)}\t{elapsed:.4f}\t{match}"
                )

        out_path = seg_dir / f"{key}_all_algos.tsv"
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        stats["timing"][key] = {a: timing[a] / max(total, 1) for a in ALGORITHMS}
        stats["jieba_agree"][key] = {a: agree[a] / max(total, 1) for a in ALGORITHMS}

    # 歧义集：Hybrid vs BiMM
    amb_sents = load_test_sentences("ambiguity_20.txt")
    hybrid_better = 0
    bimm_better = 0
    same = 0
    bimm_match_count = 0
    hybrid_match_count = 0
    amb_compare_lines = ["sentence\tBiMM\tHybrid\tjieba"]
    for sent in amb_sents:
        b_res = bimm(sent, dictionary, max_len)
        h_res = hybrid_segment(sent, dictionary, max_len, hmm_trained)
        j_res = list(jieba.cut(sent))
        amb_compare_lines.append(
            f"{sent}\t{'/'.join(b_res)}\t{'/'.join(h_res)}\t{'/'.join(j_res)}"
        )
        b_match = _match_rate(b_res, j_res)
        h_match = _match_rate(h_res, j_res)
        if b_match:
            bimm_match_count += 1
        if h_match:
            hybrid_match_count += 1
        if h_match and not b_match:
            hybrid_better += 1
        elif b_match and not h_match:
            bimm_better += 1
        else:
            same += 1

    amb_path = seg_dir / "ambiguity_compare.txt"
    amb_path.write_text("\n".join(amb_compare_lines), encoding="utf-8")
    n_amb = max(len(amb_sents), 1)
    agree_bimm = bimm_match_count / n_amb
    agree_hybrid = hybrid_match_count / n_amb
    stats["hybrid_vs_bimm_ambiguity"] = {
        "hybrid_jieba_win": hybrid_better / n_amb,
        "bimm_jieba_win": bimm_better / n_amb,
        "tie": same / n_amb,
        "bimm_agree_rate": agree_bimm,
        "hybrid_agree_rate": agree_hybrid,
        "agree_rate_improvement": agree_hybrid - agree_bimm,
    }

    _write_timing(stats)
    _write_jieba_tsv(test_files)
    return stats


def apply_baseline_compat(_params: dict[str, Any]) -> dict[str, Any]:
    """基线：仅用参考文档示例句估计的极小 HMM。"""
    from .hmm import apply_delta_smoothing, train_from_corpus

    lines = [
        "我 来 到 北京 清华大学 学习 自然语言 处理",
        "自然 语言 处理",
    ]
    return apply_delta_smoothing(train_from_corpus(lines), delta=1.0)


def _write_timing(stats: dict[str, Any]) -> None:
    lines = ["test_set\talgorithm\tavg_time_ms\tjieba_agree_rate"]
    for key in stats["timing"]:
        for algo in ALGORITHMS:
            lines.append(
                f"{key}\t{algo}\t{stats['timing'][key][algo]:.4f}\t"
                f"{stats['jieba_agree'][key][algo]:.4f}"
            )
    h = stats["hybrid_vs_bimm_ambiguity"]
    lines.append("")
    lines.append("# 歧义集 Hybrid 相对 BiMM（相对 jieba 一致率）")
    lines.append(f"hybrid_jieba_win_rate\t{h['hybrid_jieba_win']:.4f}")
    lines.append(f"bimm_jieba_win_rate\t{h['bimm_jieba_win']:.4f}")
    lines.append(
        f"hybrid_minus_bimm_paired_win\t{h['hybrid_jieba_win'] - h['bimm_jieba_win']:.4f}"
    )
    lines.append(f"bimm_agree_rate_ambiguity\t{h.get('bimm_agree_rate', 0):.4f}")
    lines.append(f"hybrid_agree_rate_ambiguity\t{h.get('hybrid_agree_rate', 0):.4f}")
    lines.append(
        f"hybrid_agree_rate_improvement\t{h.get('agree_rate_improvement', 0):.4f}"
    )
    (RESULTS_DIR / "timing_benchmark.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_jieba_tsv(test_files: dict[str, str]) -> None:
    dictionary, max_len = load_dictionary()
    hmm_trained = load_params(MODEL_PATH)
    lines = ["test_set\tsentence\tBiMM\tHybrid\tjieba\tBiMM_match\tHybrid_match"]
    for key, fname in test_files.items():
        for sent in load_test_sentences(fname):
            b = bimm(sent, dictionary, max_len)
            h = hybrid_segment(sent, dictionary, max_len, hmm_trained)
            j = list(jieba.cut(sent))
            lines.append(
                f"{key}\t{sent}\t{'/'.join(b)}\t{'/'.join(h)}\t{'/'.join(j)}\t"
                f"{b == j}\t{h == j}"
            )
    (RESULTS_DIR / "comparison_jieba.tsv").write_text("\n".join(lines), encoding="utf-8")
