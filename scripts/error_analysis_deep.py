#!/usr/bin/env python3
"""深入错误分析：细粒度错误分类、统计分布、标注粒度差异讨论。"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
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

RESULT = ROOT / "results" / "error_analysis_deep.txt"


def align_words(pred: list[str], gold: list[str]) -> list[tuple[str, str]]:
    """简单对齐：将pred和gold的词映射到字符位置，找出差异区间。"""
    def to_spans(words):
        spans = []
        pos = 0
        for w in words:
            spans.append((pos, pos + len(w)))
            pos += len(w)
        return spans

    p_spans = to_spans(pred)
    g_spans = to_spans(gold)
    return p_spans, g_spans


def classify_error_fine(sent: str, pred: list[str], gold: list[str]) -> dict:
    """细粒度错误分类。"""
    errors = []
    p_spans, g_spans = align_words(pred, gold)

    # 找出pred和gold在字符边界上的差异
    p_bounds = set()
    for s, e in p_spans:
        p_bounds.add(s)
        p_bounds.add(e)
    g_bounds = set()
    for s, e in g_spans:
        g_bounds.add(s)
        g_bounds.add(e)

    diff_bounds = p_bounds.symmetric_difference(g_bounds)

    # 判断错误类型
    has_entity = bool(re.search(r'\d|年|月|日|[A-Za-z]', sent))
    oov_markers = ["DeepSeek", "Citywalk", "AIGC", "RAG", "Kimi", "谷子", "低空经济",
                   "新质生产力", "具身智能", "多模态", "Sora", "尊嘟", "挖呀挖"]
    has_oov = any(m in sent for m in oov_markers)
    is_long = len(sent) > 30
    ambiguity_markers = ["研究生命", "苹果不大", "喜欢上", "白云", "乒乓球拍卖",
                         "中出", "和尚面", "咬死猎人", "学生会上", "新闻发布"]
    has_ambiguity = any(m in sent for m in ambiguity_markers)

    # 粒度差异分析
    pred_total = len(pred)
    gold_total = len(gold)
    pred_avg_len = sum(len(w) for w in pred) / max(len(pred), 1)
    gold_avg_len = sum(len(w) for w in gold) / max(len(gold), 1)

    # 计算边界差异的具体位置
    gran_diff = "pred更细" if pred_total > gold_total else ("pred更粗" if pred_total < gold_total else "词数相同但边界不同")

    # 具体边界差异词对
    diff_pairs = []
    p_pos = 0
    g_pos = 0
    p_idx = 0
    g_idx = 0
    while p_idx < len(pred) and g_idx < len(gold):
        pw = pred[p_idx]
        gw = gold[g_idx]
        if pw == gw:
            p_idx += 1
            g_idx += 1
        else:
            # 找到下一个对齐点
            diff_pairs.append((pw, gw))
            if len(pw) <= len(gw):
                p_idx += 1
            else:
                g_idx += 1
            if len(diff_pairs) > 5:
                break

    return {
        "sentence": sent,
        "pred": pred,
        "gold": gold,
        "pred_word_count": pred_total,
        "gold_word_count": gold_total,
        "pred_avg_word_len": round(pred_avg_len, 2),
        "gold_avg_word_len": round(gold_avg_len, 2),
        "granularity_diff": gran_diff,
        "diff_bound_count": len(diff_bounds),
        "diff_pairs": diff_pairs[:5],
        "has_entity": has_entity,
        "has_oov": has_oov,
        "is_long": is_long,
        "has_ambiguity": has_ambiguity,
    }


def main() -> None:
    dictionary, max_len = load_dictionary()
    hmm = load_params(ROOT / "models" / "hmm_params.json")

    test_files = {
        "basic": "basic.txt",
        "ambiguity": "ambiguity_20.txt",
        "oov": "oov_15.txt",
        "long": "long_10.txt",
    }

    all_errors = []
    all_correct = []
    error_type_counter = Counter()
    test_set_counter = defaultdict(int)
    total_counter = defaultdict(int)

    for key, fname in test_files.items():
        for sent in load_test_sentences(fname):
            pred = hybrid_segment(sent, dictionary, max_len, hmm)
            gold = list(jieba.cut(sent))
            total_counter[key] += 1
            if pred == gold:
                all_correct.append({"sentence": sent, "test_set": key})
                continue
            info = classify_error_fine(sent, pred, gold)
            info["test_set"] = key
            all_errors.append(info)
            test_set_counter[key] += 1

            # 统计错误类型
            if info["has_ambiguity"]:
                error_type_counter["歧义消解失败"] += 1
            if info["has_oov"]:
                error_type_counter["OOV新词未识别"] += 1
            if info["has_entity"]:
                error_type_counter["实体边界错误"] += 1
            if info["is_long"]:
                error_type_counter["长句切分过碎"] += 1
            if not info["has_ambiguity"] and not info["has_oov"] and not info["has_entity"] and not info["is_long"]:
                error_type_counter["词典粒度差异"] += 1

    # 粒度差异统计
    gran_counter = Counter(e["granularity_diff"] for e in all_errors)
    avg_diff_bound = sum(e["diff_bound_count"] for e in all_errors) / max(len(all_errors), 1)

    # 生成报告
    lines = [
        "# 深入错误分析报告",
        f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# 词典: dict_base 95886 词 | THUOCL 6 个 | 训练语料 10000 句",
        "",
        "=" * 60,
        "## 一、总体错误分布",
        "=" * 60,
        "",
    ]

    for key in test_files:
        total = total_counter[key]
        errors = test_set_counter[key]
        rate = errors / max(total, 1)
        lines.append(f"- {key}集: {errors}/{total} 句有差异 ({rate:.1%})")

    lines.append(f"- 总计: {len(all_errors)}/{sum(total_counter.values())} 句与jieba不同")
    lines.append("")

    # 错误类型统计
    lines.append("=" * 60)
    lines.append("## 二、细粒度错误类型统计（可重叠）")
    lines.append("=" * 60)
    lines.append("")
    for et, count in error_type_counter.most_common():
        lines.append(f"- {et}: {count} 句")
    lines.append("")

    # 粒度差异分析
    lines.append("=" * 60)
    lines.append("## 三、标注粒度差异分析")
    lines.append("=" * 60)
    lines.append("")
    lines.append("### 3.1 粒度方向统计")
    for g, count in gran_counter.most_common():
        lines.append(f"- {g}: {count} 句")
    lines.append(f"- 平均边界差异数: {avg_diff_bound:.1f}")
    lines.append("")

    # 粒度差异典型案例
    lines.append("### 3.2 典型粒度差异案例")
    lines.append("")

    # pred更细的案例
    finer = [e for e in all_errors if e["granularity_diff"] == "pred更细"]
    if finer:
        lines.append("#### (A) 系统切分比jieba更细（系统拆、jieba合）")
        for e in finer[:5]:
            lines.append(f"- 「{e['sentence'][:30]}...」" if len(e['sentence']) > 30 else f"- 「{e['sentence']}」")
            lines.append(f"  系统({e['pred_word_count']}词): {' / '.join(e['pred'])}")
            lines.append(f"  jieba({e['gold_word_count']}词): {' / '.join(e['gold'])}")
            lines.append(f"  差异词对: {e['diff_pairs'][:3]}")
        lines.append("")

    # pred更粗的案例
    coarser = [e for e in all_errors if e["granularity_diff"] == "pred更粗"]
    if coarser:
        lines.append("#### (B) 系统切分比jieba更粗（系统合、jieba拆）")
        for e in coarser[:5]:
            lines.append(f"- 「{e['sentence'][:30]}...」" if len(e['sentence']) > 30 else f"- 「{e['sentence']}」")
            lines.append(f"  系统({e['pred_word_count']}词): {' / '.join(e['pred'])}")
            lines.append(f"  jieba({e['gold_word_count']}词): {' / '.join(e['gold'])}")
            lines.append(f"  差异词对: {e['diff_pairs'][:3]}")
        lines.append("")

    # 边界不同但词数相同
    same_count = [e for e in all_errors if e["granularity_diff"] == "词数相同但边界不同"]
    if same_count:
        lines.append("#### (C) 词数相同但边界不同")
        for e in same_count[:5]:
            lines.append(f"- 「{e['sentence'][:30]}...」" if len(e['sentence']) > 30 else f"- 「{e['sentence']}」")
            lines.append(f"  系统({e['pred_word_count']}词): {' / '.join(e['pred'])}")
            lines.append(f"  jieba({e['gold_word_count']}词): {' / '.join(e['gold'])}")
        lines.append("")

    # 粒度差异本质讨论
    lines.append("=" * 60)
    lines.append("## 四、标注粒度差异的本质讨论")
    lines.append("=" * 60)
    lines.append("")
    lines.append('### 4.1 什么是"标注粒度差异"？')
    lines.append("")
    lines.append("中文分词不存在唯一正确的切分标准。同一句话，不同标注体系")
    lines.append('可能给出不同但都合理的切分结果。这种差异称为"标注粒度差异"。')
    lines.append("")
    lines.append("典型例子：")
    lines.append('- "自然语言处理" → 粗粒度: [自然语言处理] / 中粒度: [自然语言, 处理] / 细粒度: [自然, 语言, 处理]')
    lines.append('- "中国科学院计算所" → 粗粒度: [中国科学院, 计算所] / 细粒度: [中国, 科学院, 计算所]')
    lines.append("")
    lines.append("### 4.2 粒度差异的来源")
    lines.append("")
    lines.append("1. **词典策略差异**: 本系统使用dict_base+THUOCL合并词典，")
    lines.append("   jieba使用自有词典。词典收录的词长不同直接导致切分粒度不同。")
    lines.append("   例: dict_base收录'自然语言处理'为整词，jieba拆为'自然语言/处理'。")
    lines.append("")
    lines.append("2. **训练语料差异**: HMM的训练语料（合成语料10000句）与jieba的")
    lines.append("   训练语料（人民日报等真实语料）标注规范不同，导致BMES标签分布差异。")
    lines.append("")
    lines.append("3. **算法策略差异**: BiMM优先最长匹配，倾向于粗粒度；")
    lines.append("   HMM基于统计概率，倾向于更细粒度。jieba的HMM也类似。")
    lines.append("")
    lines.append("4. **新词处理差异**: 对OOV新词，本系统用HMM+OOV词典拆分，")
    lines.append("   jieba用Viterbi+HMM发现新词，策略不同导致粒度差异。")
    lines.append("   例: '黑神话悟空' → 本系统[黑神话, 悟空] vs jieba[黑, 神话, 悟空]")
    lines.append("")
    lines.append("### 4.3 粒度差异≠错误")
    lines.append("")
    lines.append("关键认识：与jieba不一致不等于错误。以下情况属于合理差异：")
    lines.append("- '自然语言处理' 作为整词 vs 拆分为'自然语言/处理'，两者都合理")
    lines.append("- '中国科学院/计算所' vs '中国/科学院/计算所'，取决于应用场景")
    lines.append("- '不大/好吃' vs '不/大/好吃'，取决于语法分析粒度需求")
    lines.append("")
    lines.append("真正的错误是：")
    lines.append("- 实体被切碎: 'DeepSeek' → 'Deep/S/ek'（Layer0应保护）")
    lines.append("- 歧义消解错误: '研究/生命/起源' → '研究生/命/起源'（需上下文）")
    lines.append("- 长句过碎: 连续单字输出（HMM训练不充分）")
    lines.append("")
    lines.append("### 4.4 改进方向")
    lines.append("")
    lines.append("1. 统一标注规范: 建立明确的粒度标准（如CTB、PKU、MSRA规范）")
    lines.append("2. 增加gold标注: 人工标注200句作为客观评估基准")
    lines.append("3. 词典对齐: 将jieba词典中的细粒度词加入dict_base")
    lines.append("4. 语料升级: 使用真实标注语料（如人民日报标注语料）训练HMM")

    RESULT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {RESULT}")

    # 写入日志
    log("experiment_journal.txt",
        f"深入错误分析完成 | 总差异句={len(all_errors)} | "
        f"歧义={error_type_counter.get('歧义消解失败',0)} | "
        f"OOV={error_type_counter.get('OOV新词未识别',0)} | "
        f"实体={error_type_counter.get('实体边界错误',0)} | "
        f"长句={error_type_counter.get('长句切分过碎',0)} | "
        f"粒度差异={error_type_counter.get('词典粒度差异',0)}",
        phase="6", task="error_analysis_deep")


if __name__ == "__main__":
    main()
