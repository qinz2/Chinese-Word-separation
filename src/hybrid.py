"""四层混合：长句分段 -> Layer0 实体预处理 -> Layer1 BiMM(词典最长匹配) -> Layer2 HMM(长未登录串) -> Layer3 实体规则。"""
from __future__ import annotations

import re
from typing import Any

from .bimm import bimm
from .hmm import segment as hmm_segment
from .rules import merge_entities, RE_DATE, RE_ENGLISH, RE_ARABIC_NUM, RE_CN_NUM, RE_MIXED

# 标点符号集合，用于长句分段
SENTENCE_PUNCTUATION = set('，。！？；、：\n\r\t')
CLAUSE_PUNCTUATION = set('，；、')


def _segment_long_sentence(text: str, max_segment_len: int = 30) -> list[str]:
    """
    长句分段处理：将超长句子按标点切分为多个较短片段
    
    策略：
    1. 优先在句子结束标点（。！？）处切分
    2. 其次在分句标点（，；、）处切分
    3. 最长不超过max_segment_len字符
    """
    if len(text) <= max_segment_len:
        return [text]
    
    segments = []
    start = 0
    i = 0
    
    while i < len(text):
        # 检查是否达到最大长度
        if i - start >= max_segment_len:
            # 在最大长度内寻找最近的标点
            found = False
            for j in range(i, max(start, i - 10), -1):
                if text[j] in CLAUSE_PUNCTUATION:
                    segments.append(text[start:j+1])
                    start = j + 1
                    i = start
                    found = True
                    break
            if not found:
                # 实在找不到标点，强制切分
                segments.append(text[start:i])
                start = i
                i = start
        
        # 检查句子结束标点
        if i < len(text) and text[i] in SENTENCE_PUNCTUATION:
            segments.append(text[start:i+1])
            start = i + 1
            i = start
        
        i += 1
    
    # 添加剩余部分
    if start < len(text):
        segments.append(text[start:])
    
    return segments


def _pre_extract_entities(sentence: str) -> list[tuple[str, str]]:
    """Layer0 预处理：提取英文/数字/日期实体，返回[(类型, 文本), ...]
    
    规则优先级：日期 > 英文 > 阿拉伯数字 > 中文数字 > 混合实体
    这样可以避免日期被混合实体规则错误匹配
    """
    spans = []
    # 按优先级顺序匹配，日期优先
    for pat in (RE_DATE, RE_ENGLISH, RE_ARABIC_NUM, RE_CN_NUM, RE_MIXED):
        for m in pat.finditer(sentence):
            spans.append((m.start(), m.end()))
    
    if not spans:
        return [("chinese", sentence)]
    
    # 合并重叠区间
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s >= merged[-1][1]:
            merged.append((s, e))
        elif e > merged[-1][1]:
            merged[-1] = (merged[-1][0], e)
    
    result = []
    pos = 0
    for s, e in merged:
        if s > pos:
            result.append(("chinese", sentence[pos:s]))
        result.append(("entity", sentence[s:e]))
        pos = e
    if pos < len(sentence):
        result.append(("chinese", sentence[pos:]))
    return result


def _load_oov_coarse_words() -> set[str]:
    """加载dict_oov.txt中的粗粒度新词（排除细粒度部分）"""
    coarse_words = set()
    try:
        with open('data/dict/dict_oov.txt', 'r', encoding='utf-8') as f:
            in_coarse_section = True
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    # 检测是否进入细粒度部分（精确匹配）
                    if line.startswith('# 细粒度'):
                        in_coarse_section = False
                    continue
                if in_coarse_section:
                    coarse_words.add(line)
    except FileNotFoundError:
        pass  # 如果文件不存在，返回空集合
    return coarse_words


def _greedy_subword_split(word: str, dictionary: set[str], max_len: int) -> list[str]:
    """
    贪心子词拆分（排除整词，优先匹配最长可用子词）
    
    关键改进：
    1. 第一次尝试时限制最大长度为len(word)-1，避免匹配整词
    2. 允许部分拆分+单字补全（如果前面已有有效拆分）
    """
    if len(word) <= 3:
        return []
    
    subwords = []
    pos = 0
    while pos < len(word):
        matched = False
        # 计算当前可尝试的最大长度
        max_try_len = min(max_len, len(word) - pos)
        
        # 关键：如果是第一次匹配且剩余长度等于原词长度，排除整词
        if pos == 0 and len(word) <= max_len:
            max_try_len = len(word) - 1
        
        for length in range(max_try_len, 0, -1):
            subword = word[pos:pos + length]
            if subword in dictionary:
                subwords.append(subword)
                pos += length
                matched = True
                break
        
        if not matched:
            # 如果前面已有有效拆分，允许单字补全
            if subwords:
                subwords.append(word[pos])
                pos += 1
            else:
                return []  # 完全无法拆分
    
    return subwords if len(subwords) >= 2 else []


def _layer2_process(
    words: list[str],
    dictionary: set[str],
    max_len: int,
    hmm_params: dict[str, Any],
) -> list[str]:
    """Layer2: 对BiMM结果中的连续未登录单字启用HMM，并对oov粗粒度长词应用贪心拆分"""
    # 加载oov粗粒度词典（仅包含需要细粒度处理的新词）
    oov_coarse_words = _load_oov_coarse_words()
    
    refined: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        
        # 情况1：oov粗粒度词且长度>3 → 尝试贪心子词拆分
        if len(w) > 3 and w in oov_coarse_words:
            subwords = _greedy_subword_split(w, dictionary, max_len)
            if subwords:
                refined.extend(subwords)  # 使用贪心拆分的子词
            else:
                refined.append(w)  # 无法合理拆分，保持原词
            i += 1
        
        # 情况2：原有的未登录单字处理逻辑
        elif len(w) == 1 and w not in dictionary:
            buf = w
            i += 1
            while i < len(words) and len(words[i]) == 1 and words[i] not in dictionary:
                buf += words[i]
                i += 1
            # 仅在长度>=3时启用HMM（避免短串被切碎）
            if len(buf) >= 3:
                refined.extend(hmm_segment(buf, hmm_params))
            else:
                refined.append(buf)
        
        # 情况3：其他词保持原样（包括dict_base中的专有名词）
        else:
            refined.append(w)
            i += 1
    return refined


def hybrid_segment(
    sentence: str,
    dictionary: set[str],
    max_len: int,
    hmm_params: dict[str, Any],
) -> list[str]:
    """
    新流程：
    长句分段 - 将超长句子切分为较短片段
    Layer0：预处理 - 提取英文/数字/日期实体（保护不被BiMM切碎）
    Layer1：BiMM - 只处理剩余中文片段
    Layer2：HMM - 对连续未登录单字再切分
    Layer3：后处理 - 合并相邻实体
    """
    # 长句分段预处理：避免超长句子导致的切分过碎问题
    sentence_segments = _segment_long_sentence(sentence)
    
    # 对每个分段进行处理
    final_result: list[str] = []
    for seg in sentence_segments:
        # Layer0: 先用规则提取英文/数字/日期，得到span信息
        segments = _pre_extract_entities(seg)
        
        # Layer1+2: 对每个中文片段进行BiMM+HMM处理
        refined: list[str] = []
        for seg_type, seg_text in segments:
            if seg_type == "entity":
                refined.append(seg_text)  # 直接保留实体
            else:
                words = bimm(seg_text, dictionary, max_len)
                # Layer2: HMM处理连续未登录单字
                refined.extend(_layer2_process(words, dictionary, max_len, hmm_params))
        
        # Layer3: 后处理合并
        final_result.extend(merge_entities(refined))
    
    return final_result


def fmm_with_spans(
    sentence: str, dictionary: set[str], max_len: int
) -> list[tuple[str, bool]]:
    """保留 FMM 块格式 API，供实验对比 Layer1 FMM 行为。"""
    from .fmm import fmm

    words = fmm(sentence, dictionary, max_len)
    spans: list[tuple[str, bool]] = []
    oov_buf = ""
    for w in words:
        if len(w) == 1 and w not in dictionary:
            oov_buf += w
        else:
            if oov_buf:
                spans.append((oov_buf, False))
                oov_buf = ""
            spans.append((w, True))
    if oov_buf:
        spans.append((oov_buf, False))
    return spans
