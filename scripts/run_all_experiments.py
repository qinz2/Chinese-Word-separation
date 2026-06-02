#!/usr/bin/env python3
"""一键运行：建词典 -> 语料 -> HMM -> 全量评测 -> 错误分析。"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bimm import bimm
from src.bmm import bmm
from src.dict_loader import MAX_LEN_CAP, load_dictionary
from src.evaluate import benchmark_all, load_test_sentences
from src.fmm import fmm
from src.hmm import baseline_params, load_params, segment as hmm_segment, train_and_save
from src.hybrid import hybrid_segment
from src.logger_util import journal, log


def run_cmd(args: list[str]) -> None:
    subprocess.run([sys.executable] + args, check=True, cwd=str(ROOT))


def phase2_logs() -> None:
    dictionary, max_len = load_dictionary()
    log(
        "phase2_fmm_bmm.txt",
        f"DESIGN | MAX_LEN={max_len} | cap={MAX_LEN_CAP}",
        phase="2",
        task="config",
    )
    for fname in ["basic.txt", "ambiguity_20.txt"]:
        for sent in load_test_sentences(fname):
            t0 = time.perf_counter()
            fr = fmm(sent, dictionary, max_len)
            t1 = time.perf_counter()
            br = bmm(sent, dictionary, max_len)
            t2 = time.perf_counter()
            log(
                "phase2_fmm_bmm.txt",
                f"sentence={sent} | FMM={fr} | {((t1-t0)*1000):.3f}ms",
                phase="2",
                task="FMM",
            )
            log(
                "phase2_fmm_bmm.txt",
                f"sentence={sent} | BMM={br} | {((t2-t1)*1000):.3f}ms | diff={fr != br}",
                phase="2",
                task="BMM",
            )

    amb_lines = ["sentence\tFMM\tBMM\tsame"]
    diff_count = 0
    for sent in load_test_sentences("ambiguity_20.txt"):
        fr = fmm(sent, dictionary, max_len)
        br = bmm(sent, dictionary, max_len)
        same = fr == br
        if not same:
            diff_count += 1
        amb_lines.append(f"{sent}\t{'/'.join(fr)}\t{'/'.join(br)}\t{same}")
    out = ROOT / "results" / "segmentation" / "ambiguity_compare.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(amb_lines), encoding="utf-8")


def phase3_hmm_compare() -> None:
    dictionary, max_len = load_dictionary()
    base = baseline_params()
    # 基线：极小词汇上的手工统计等价
    from src.hmm import apply_delta_smoothing, train_from_corpus

    base_trained = apply_delta_smoothing(
        train_from_corpus(["我 来 到 北京 清华大学 学习 自然语言 处理"]), delta=1.0
    )
    opt = load_params(ROOT / "models" / "hmm_params.json")
    lines = ["sentence\tHMM_baseline\tHMM_trained"]
    for sent in load_test_sentences("oov_15.txt"):
        b = hmm_segment(sent, base_trained)
        o = hmm_segment(sent, opt)
        lines.append(f"{sent}\t{'/'.join(b)}\t{'/'.join(o)}")
        log(
            "phase3_hmm_baseline.txt",
            f"OOV sentence={sent} | baseline={b} | trained={o}",
            phase="3",
            task="compare",
        )
    p = ROOT / "results" / "segmentation" / "hmm_before_after.txt"
    p.write_text("\n".join(lines), encoding="utf-8")


def phase4_bimm() -> None:
    dictionary, max_len = load_dictionary()
    agree_f = agree_b = agree_bi = 0
    n = 0
    for sent in load_test_sentences("ambiguity_20.txt"):
        n += 1
        f = fmm(sent, dictionary, max_len)
        b = bmm(sent, dictionary, max_len)
        bi = bimm(sent, dictionary, max_len)
        if f == bi:
            agree_f += 1
        if b == bi:
            agree_b += 1
        if f == b:
            agree_bi += 1
        log(
            "phase4_bimm_hmm_opt.txt",
            f"sentence={sent} | BiMM={bi} | FMM={f} | BMM={b}",
            phase="4",
            task="BiMM",
        )
    log(
        "phase4_bimm_hmm_opt.txt",
        f"歧义集统计 | n={n} | BiMM=FMM {agree_f}/{n} | BiMM=BMM {agree_b}/{n}",
        phase="4",
        task="stats",
        also_journal=True,
    )


def phase5_hybrid() -> None:
    dictionary, max_len = load_dictionary()
    hmm = load_params(ROOT / "models" / "hmm_params.json")
    sample = "2024年12月31日DeepSeek发布大模型售价三千元"
    res = hybrid_segment(sample, dictionary, max_len, hmm)
    log(
        "phase5_hybrid_jieba.txt",
        f"Layer1=BiMM Layer2=HMM(>6字OOV) Layer3示例 | sentence={sample} | result={res}",
        phase="5",
        task="hybrid_rules",
        also_journal=True,
    )


def main() -> None:
    journal("=== 实验流水线开始（THUOCL/语料已更新则重跑） ===")
    for name in [
        "phase1_dict_corpus.txt",
        "phase2_fmm_bmm.txt",
        "phase3_hmm_baseline.txt",
        "phase4_bimm_hmm_opt.txt",
        "phase5_hybrid_jieba.txt",
    ]:
        p = ROOT / "logs" / name
        if p.exists():
            p.unlink()
    run_cmd([str(ROOT / "scripts" / "build_dict.py")])
    run_cmd([str(ROOT / "scripts" / "prepare_corpus.py"), "--num_lines", "10000"])
    run_cmd([str(ROOT / "scripts" / "train_hmm.py"), "--num_lines", "10000"])

    phase2_logs()
    phase3_hmm_compare()
    phase4_bimm()
    phase5_hybrid()

    stats = benchmark_all()
    run_cmd([str(ROOT / "scripts" / "generate_error_analysis.py")])
    run_cmd([str(ROOT / "scripts" / "generate_run_summary.py")])

    hstat = stats["hybrid_vs_bimm_ambiguity"]
    imp = hstat.get("agree_rate_improvement", 0)
    journal(
        f"评测完成 | 歧义集Hybrid一致率={hstat.get('hybrid_agree_rate',0):.2%} "
        f"BiMM={hstat.get('bimm_agree_rate',0):.2%} | 提升={imp:.2%} | "
        f"验收>=5%: {'PASS' if imp >= 0.05 else 'CHECK'}"
    )

    print("\n=== 日志与结果 ===")
    for p in [
        "logs/experiment_journal.txt",
        "logs/phase1_dict_corpus.txt",
        "logs/phase2_fmm_bmm.txt",
        "logs/phase3_hmm_baseline.txt",
        "logs/phase4_bimm_hmm_opt.txt",
        "logs/phase5_hybrid_jieba.txt",
        "results/timing_benchmark.txt",
        "results/comparison_jieba.tsv",
        "results/error_analysis.txt",
        "results/segmentation/",
    ]:
        print(ROOT / p)


if __name__ == "__main__":
    main()
