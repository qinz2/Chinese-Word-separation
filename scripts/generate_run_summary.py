#!/usr/bin/env python3
"""生成实验运行摘要（报告素材索引）。"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dict_loader import load_dictionary
from src.evaluate import benchmark_all  # noqa: F401

OUT = ROOT / "results" / "RUN_SUMMARY.txt"
THUOCL = ROOT / "data" / "raw" / "thuocl"
PKU_CANDIDATES = [
    ROOT / "data" / "raw" / "pku_train.txt",
    ROOT / "data" / "raw" / "19980101-train.txt",
    ROOT / "data" / "raw" / "1998-01-2003版-带音.txt",
]
TIMING = ROOT / "results" / "timing_benchmark.txt"


def main() -> None:
    lines = [
        f"# 实验运行摘要",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 数据资源",
    ]
    thuocl_files = list(THUOCL.glob("*.txt")) if THUOCL.exists() else []
    lines.append(f"- THUOCL: {len(thuocl_files)} 个文件 -> {', '.join(p.name for p in thuocl_files) or '未找到'}")
    if (ROOT / "data" / "dict" / "dict_base.txt").exists():
        n = len((ROOT / "data" / "dict" / "dict_base.txt").read_text(encoding="utf-8").splitlines())
        lines.append(f"- dict_base.txt: {n} 词")
    
    # 检测 PKU 语料
    pku_found = None
    for p in PKU_CANDIDATES:
        if p.exists() and p.stat().st_size > 10000:
            pku_found = p
            break
    
    if pku_found:
        sz = pku_found.stat().st_size // 1024
        lines.append(f"- PKU语料: 已就绪 ({pku_found.name}, {sz} KB)")
    else:
        lines.append("- PKU语料: **未检测到**，HMM 仍使用合成语料 train_corpus.txt")
    corpus = ROOT / "data" / "corpus" / "train_corpus.txt"
    if corpus.exists():
        lines.append(f"- 训练语料: {len(corpus.read_text(encoding='utf-8').splitlines())} 句")
    try:
        d, ml = load_dictionary()
        lines.append(f"- 合并词典规模: {len(d)} | MAX_LEN={ml}")
    except Exception as e:
        lines.append(f"- 词典加载: {e}")

    lines.extend(["", "## 关键指标（歧义集）"])
    if TIMING.exists():
        for line in TIMING.read_text(encoding="utf-8").splitlines():
            if "ambiguity" in line and "Hybrid" in line:
                lines.append(f"- {line.replace(chr(9), ' | ')}")
            if line.startswith("hybrid_") or line.startswith("bimm_agree"):
                lines.append(f"- {line.replace(chr(9), ' = ')}")

    lines.extend(
        [
            "",
            "## 报告素材路径",
            "- logs/experiment_journal.txt",
            "- logs/phase1_dict_corpus.txt ~ phase5_hybrid_jieba.txt",
            "- results/segmentation/*.tsv",
            "- results/timing_benchmark.txt",
            "- results/comparison_jieba.tsv",
            "- results/error_analysis.txt",
            "- models/hmm_params.json",
            "",
            "## 说明",
            "- 与 jieba 一致率为弱基准（无人工 gold）",
            "- OOV 句中自研词典含新词时，与 jieba 切分差异可能反映词典策略不同而非绝对错误",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
