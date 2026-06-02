"""HMM 分词：BMES、语料训练、+1 平滑、O(4) 空间 Viterbi（含双步回溯指针）。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATES = ["B", "M", "E", "S"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = 4
LOG_ZERO = -1e16


def word_to_tags(word: str) -> list[str]:
    if len(word) == 1:
        return ["S"]
    return ["B"] + ["M"] * (len(word) - 2) + ["E"]


def sentence_to_tags(words: list[str]) -> list[str]:
    tags: list[str] = []
    for w in words:
        tags.extend(word_to_tags(w))
    return tags


def baseline_params() -> dict[str, Any]:
    """教学用小规模基线参数。"""
    return {
        "pi": {"B": 0.5, "M": 0.0, "E": 0.0, "S": 0.5},
        "A": {
            "B": {"M": 0.5, "E": 0.5},
            "M": {"M": 0.5, "E": 0.5},
            "E": {"B": 0.5, "S": 0.5},
            "S": {"B": 0.5, "S": 0.5},
        },
        "B": {
            "B": {"自": 0.5, "清": 0.5},
            "M": {"然": 0.5, "华": 0.5},
            "E": {"言": 0.5, "学": 0.5},
            "S": {"我": 0.5, "来": 0.5},
        },
        "vocab": ["自", "然", "言", "清", "华", "学", "我", "来"],
    }


def train_from_corpus(lines: list[str]) -> dict[str, Any]:
    pi_count = {s: 0 for s in STATES}
    trans_count = {s: {t: 0 for t in STATES} for s in STATES}
    emit_count = {s: {} for s in STATES}
    vocab: set[str] = set()
    char_train_freq: dict[str, int] = {}

    for line in lines:
        words = [w for w in line.strip().split() if w]
        if not words:
            continue
        tags = sentence_to_tags(words)
        chars = "".join(words)
        if len(tags) != len(chars):
            continue
        pi_count[tags[0]] += 1
        for ch, tag in zip(chars, tags):
            vocab.add(ch)
            char_train_freq[ch] = char_train_freq.get(ch, 0) + 1
            emit_count[tag][ch] = emit_count[tag].get(ch, 0) + 1
        for i in range(1, len(tags)):
            trans_count[tags[i - 1]][tags[i]] += 1

    return {
        "pi_count": pi_count,
        "trans_count": trans_count,
        "emit_count": emit_count,
        "vocab": sorted(vocab),
        "char_train_freq": char_train_freq,
    }


def apply_delta_smoothing(raw: dict[str, Any], delta: float = 0.1) -> dict[str, Any]:
    """
    +δ 平滑（替代 +1 平滑）：
    - 转移：P(t|s) = (count(s->t) + delta) / (count(s) + delta * N_STATES)
    - 发射：P(c|s) = (count(s,c) + delta) / (count(s) + delta * |V|)
    """
    vocab = raw["vocab"]
    v_size = len(vocab)

    pi_total = sum(raw["pi_count"].values()) + delta * N_STATES
    pi = {s: (raw["pi_count"][s] + delta) / pi_total for s in STATES}

    A: dict[str, dict[str, float]] = {s: {} for s in STATES}
    for s in STATES:
        s_total = sum(raw["trans_count"][s].values()) + delta * N_STATES
        for t in STATES:
            A[s][t] = (raw["trans_count"][s][t] + delta) / s_total

    B: dict[str, dict[str, float]] = {s: {} for s in STATES}
    for s in STATES:
        s_emit_total = sum(raw["emit_count"][s].values()) + delta * v_size
        for c in vocab:
            B[s][c] = (raw["emit_count"][s].get(c, 0) + delta) / s_emit_total

    return {"pi": pi, "A": A, "B": B, "vocab": vocab, "char_train_freq": raw.get("char_train_freq", {})}


def params_to_log_matrices(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vocab = params["vocab"]
    char_idx = {c: i for i, c in enumerate(vocab)}
    v_size = len(vocab)

    pi = np.full(N_STATES, LOG_ZERO)
    for s in STATES:
        pi[STATE_IDX[s]] = math.log(max(params["pi"].get(s, 1e-8), 1e-12))

    A = np.full((N_STATES, N_STATES), LOG_ZERO)
    for s in STATES:
        for t in STATES:
            A[STATE_IDX[s], STATE_IDX[t]] = math.log(
                max(params["A"].get(s, {}).get(t, 1e-8), 1e-12)
            )

    B = np.full((N_STATES, v_size), LOG_ZERO)
    for s in STATES:
        for c, j in char_idx.items():
            B[STATE_IDX[s], j] = math.log(max(params["B"].get(s, {}).get(c, 1e-8), 1e-12))

    return pi, A, B, vocab


def viterbi_o4(sentence: str, params: dict[str, Any]) -> list[str]:
    """
    O(4) 空间 Viterbi：仅保留当前/上一步概率；
    回溯需同时保留 prev_bp（上一步）与 cur_bp（当前步）指针数组。
    支持 OOV 字符分级回退策略。
    """
    if not sentence:
        return []

    pi_log, A, B, vocab = params_to_log_matrices(params)
    char_idx = {c: i for i, c in enumerate(vocab)}
    v_size = len(vocab)
    
    char_train_freq = params.get("char_train_freq", {})

    n = len(sentence)
    prev_v = pi_log.copy()
    prev_bp = np.zeros(N_STATES, dtype=np.int32)

    for t in range(n):
        ch = sentence[t]
        if ch in char_idx:
            emit_col = np.array([B[s, char_idx[ch]] for s in range(N_STATES)])
        else:
            freq = char_train_freq.get(ch, 0)
            if freq > 10:
                emit_default = math.log(0.5 / v_size)
            elif freq > 0:
                emit_default = math.log(0.1 / v_size)
            else:
                emit_default = math.log(0.01 / v_size)
            emit_col = np.full(N_STATES, emit_default)
        cur_v = np.full(N_STATES, LOG_ZERO)
        cur_bp = np.zeros(N_STATES, dtype=np.int32)
        for y in range(N_STATES):
            scores = prev_v + A[:, y]
            best = int(np.argmax(scores))
            cur_v[y] = scores[best] + emit_col[y]
            cur_bp[y] = best
        prev_v = cur_v
        if t == 0:
            path_bp = [cur_bp.copy()]
        else:
            path_bp.append(cur_bp.copy())
        prev_v, prev_bp = cur_v, cur_bp

    last_y = int(np.argmax(prev_v))
    tags = [STATES[last_y]]
    for t in range(n - 1, 0, -1):
        last_y = int(path_bp[t][last_y])
        tags.append(STATES[last_y])
    tags.reverse()
    return tags


def tags_to_words(sentence: str, tags: list[str]) -> list[str]:
    result: list[str] = []
    buf = ""
    for ch, tag in zip(sentence, tags):
        if tag == "S":
            if buf:
                result.append(buf)
                buf = ""
            result.append(ch)
        elif tag == "B":
            if buf:
                result.append(buf)
            buf = ch
        elif tag == "M":
            buf += ch
        elif tag == "E":
            buf += ch
            result.append(buf)
            buf = ""
    if buf:
        result.append(buf)
    return result


def segment(sentence: str, params: dict[str, Any]) -> list[str]:
    tags = viterbi_o4(sentence, params)
    return tags_to_words(sentence, tags)


def save_params(params: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def load_params(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def train_and_save(corpus_path: Path, out_path: Path, delta: float = 0.1) -> dict[str, Any]:
    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    raw = train_from_corpus(lines)
    params = apply_delta_smoothing(raw, delta)
    save_params(params, out_path)
    return params
