"""双向最大匹配 (BiMM)。"""
from __future__ import annotations

import json
import os
from typing import Optional

from .bmm import bmm
from .fmm import fmm

# 全局bigram模型
_bigram_probs = None
_unigram_probs = None


def _load_bigram_model() -> None:
    """加载bigram概率模型"""
    global _bigram_probs, _unigram_probs
    if _bigram_probs is not None:
        return
    
    model_path = 'models/bigram_params.json'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                model = json.load(f)
                _bigram_probs = model.get('bigram_probs', {})
                _unigram_probs = model.get('unigram_probs', {})
        except Exception as e:
            print(f"加载bigram模型失败: {e}")
            _bigram_probs = {}
            _unigram_probs = {}
    else:
        _bigram_probs = {}
        _unigram_probs = {}


def _single_char_count(words: list[str]) -> int:
    return sum(1 for w in words if len(w) == 1)


def _calculate_bigram_score(words: list[str]) -> float:
    """计算分词结果的bigram概率分数"""
    _load_bigram_model()
    
    if not _bigram_probs:
        return 0.0
    
    score = 0.0
    # 添加句子开始标记
    padded = ['<BOS>'] + words + ['<EOS>']
    
    for i in range(len(padded) - 1):
        w1 = padded[i]
        w2 = padded[i + 1]
        key = f"{w1}||{w2}"
        if key in _bigram_probs:
            score += _bigram_probs[key]
        else:
            # 未知bigram，使用unigram概率的乘积作为默认值
            p1 = _unigram_probs.get(w1, 1e-10)
            p2 = _unigram_probs.get(w2, 1e-10)
            score += p1 * p2 * 0.1  # 降低未知bigram的权重
    
    return score


def bimm(sentence: str, dictionary: dict[str, int], max_len: int) -> list[str]:
    f_result = fmm(sentence, dictionary, max_len)
    b_result = bmm(sentence, dictionary, max_len)
    
    # 规则1：FMM和BMM结果一致
    if f_result == b_result:
        return f_result
    
    # 规则2：词数不同，选择词数少的
    if len(f_result) != len(b_result):
        return f_result if len(f_result) < len(b_result) else b_result
    
    # 规则3：单字数不同，选择单字数少的
    f_singles = _single_char_count(f_result)
    b_singles = _single_char_count(b_result)
    if f_singles != b_singles:
        return f_result if f_singles < b_singles else b_result
    
    # 规则4：使用bigram概率选择最优解
    f_score = _calculate_bigram_score(f_result)
    b_score = _calculate_bigram_score(b_result)
    if f_score != b_score:
        return f_result if f_score > b_score else b_result
    
    # 规则5：默认选择BMM结果
    return b_result
