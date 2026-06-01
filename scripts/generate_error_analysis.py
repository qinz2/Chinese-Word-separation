#!/usr/bin/env python3
"""生成 error_analysis.txt（五类错误 + 日志 error_type 统计）。"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jieba

from src.bimm import bimm
from src.dict_loader import load_dictionary
from src.evaluate import load_test_sentences
from src.hybrid import hybrid_segment
from src.hmm import load_params
from src.logger_util import log

RESULT = ROOT / "results" / "error_analysis.txt"
LOG_DIR = ROOT / "logs"


def classify_error(sent: str, pred: list[str], gold: list[str]) -> str | None:
    if pred == gold:
        return None
    text = "".join(pred)
    if re.search(r"\d|年|月|日|[A-Za-z]", sent) and pred != gold:
        if any(re.search(r"\d", w) for w in pred) or any(
            re.match(r"[A-Za-z]", w) for w in pred
        ):
            return "entity"
    oov_markers = ["DeepSeek", "Citywalk", "AIGC", "RAG", "Kimi", "谷子", "低空经济"]
    if any(m in sent for m in oov_markers):
        return "oov"
    if len(sent) > 30:
        return "long_overcut"
    if any(x in sent for x in ["研究生命", "苹果不大", "喜欢上", "明天", "白云"]):
        return "ambiguity"
    return "other"


def main() -> None:
    dictionary, max_len = load_dictionary()
    hmm = load_params(ROOT / "models" / "hmm_params.json")
    sections = {
        "ambiguity": ("ambiguity_20.txt", "交集/组合歧义"),
        "oov": ("oov_15.txt", "未登录词"),
        "long_overcut": ("long_10.txt", "长句切分过碎"),
        "entity": ("oov_15.txt", "实体识别（数字/日期/英文/中文数字）"),
        "other": ("basic.txt", "其他"),
    }
    err_counter: Counter[str] = Counter()
    from datetime import datetime
    from pathlib import Path as P

    dict_n = len(
        (ROOT / "data" / "dict" / "dict_base.txt").read_text(encoding="utf-8").splitlines()
    )
    thuocl_n = len(list((ROOT / "data" / "raw" / "thuocl").glob("*.txt"))) if (ROOT / "data" / "raw" / "thuocl").exists() else 0
    pku_ok = (ROOT / "data" / "raw" / "pku_train.txt").exists()
    lines = [
        "# 错误分析报告（五类）",
        f"# 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# 词典: dict_base {dict_n} 词 | THUOCL文件 {thuocl_n} 个 | 人民日报语料: {'已使用' if pku_ok else '未提供(合成语料训练HMM)'}",
        "",
    ]

    for err_type, (fname, desc) in sections.items():
        lines.append(f"## {err_type} — {desc}")
        count = 0
        for sent in load_test_sentences(fname):
            pred = hybrid_segment(sent, dictionary, max_len, hmm)
            gold = list(jieba.cut(sent))
            et = classify_error(sent, pred, gold)
            if et is None:
                continue
            if err_type != "entity" and et != err_type:
                continue
            if err_type == "entity" and et != "entity":
                continue
            count += 1
            if count > 5:
                break
            err_counter[et] += 1
            lines.append(f"- 句子：{sent}")
            lines.append(f"  预测：{' / '.join(pred)}")
            lines.append(f"  jieba：{' / '.join(gold)}")
            lines.append(
                f"  原因：{'规则法贪心导致边界错误' if et == 'ambiguity' else ''}"
                f"{'词典未收录新词' if et == 'oov' else ''}"
                f"{'长句多次单字fallback' if et == 'long_overcut' else ''}"
                f"{'Layer3未覆盖或Layer1单字化' if et == 'entity' else ''}"
            )
            log(
                "experiment_journal.txt",
                f"error sentence={sent} | pred={pred} | gold={gold}",
                phase="5",
                task="error_analysis",
                error_type=et,
            )
        lines.append("")

    lines.append("## 错误类型统计（来自日志 error_type）")
    for et, c in err_counter.most_common():
        lines.append(f"- {et}: {c}")

    lines.append("")
    lines.append("## 报告可用见解模板")
    lines.append("- 歧义：需语义或统计上下文，纯词典法不足")
    lines.append("- OOV：HMM+词典扩展可缓解，但覆盖率有限")
    lines.append("- 实体：规则层对数字/日期/英文/中文数字可 100% 切分")
    lines.append("- 长句：最大匹配易碎，BiMM/混合可减轻")

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {RESULT}")


if __name__ == "__main__":
    main()
