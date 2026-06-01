# 中文分词算法优化方案

## 问题分析

### 当前瓶颈
- **OOV集一致率0%**：词典已包含新词，但与jieba标注粒度不同（非算法问题）
- **长句集一致率0%**：Layer2 HMM对连续单字片段切分过碎
- **歧义集45-50%**：BiMM融合规则简单，无法处理复杂歧义

### 根因定位
1. `src/hybrid.py` Line 17: `if len(text) > 6` 触发HMM条件过于宽松
2. `src/bimm.py` Line 12-23: 融合规则仅考虑词数和单字数，未利用统计信息
3. `data/dict/dict_oov.txt`: 需补充2024-2025最新网络热词

## 改进步骤

### 步骤1：优化HMM Layer2触发条件

**文件**: `src/hybrid.py`

**问题**: 当前对所有长度>6的连续单字串都启用HMM，导致过度切分

**修改方案**:
```python
def _layer2_segment(
    text: str,
    dictionary: set[str],
    max_len: int,
    hmm_params: dict[str, Any],
) -> list[str]:
    # 仅在满足以下条件时启用HMM：
    # 1. 长度 >= 4 (避免短串被切碎)
    # 2. 单字比例 > 50% (确认为未登录词区域)
    if len(text) >= 4:
        # 检查是否大部分字符不在词典中
        oov_ratio = sum(1 for c in text if c not in dictionary) / len(text)
        if oov_ratio > 0.5:
            return hmm_segment(text, hmm_params)
    return bimm(text, dictionary, max_len)
```

**预期效果**: 减少长句中不必要的HMM切分，降低overcut错误

**日志更新**: 在 `logs/phase5_hybrid_jieba.txt` 添加Layer2触发统计

---

### 步骤2：增强BiMM融合规则

**文件**: `src/bimm.py`

**问题**: 当前规则无法处理"研究生命起源"这类组合歧义

**修改方案**:
```python
def bimm(sentence: str, dictionary: set[str], max_len: int) -> list[str]:
    f_result = fmm(sentence, dictionary, max_len)
    b_result = bmm(sentence, dictionary, max_len)
    
    if f_result == b_result:
        return f_result
    
    # 规则1: 词数不同 → 选词数少的
    if len(f_result) != len(b_result):
        return f_result if len(f_result) < len(b_result) else b_result
    
    # 规则2: 单字数不同 → 选单字少的
    f_singles = _single_char_count(f_result)
    b_singles = _single_char_count(b_result)
    if f_singles != b_singles:
        return f_result if f_singles < b_singles else b_result
    
    # 新增规则3: 计算平均词长，选平均词长长的（更紧凑）
    f_avg_len = sum(len(w) for w in f_result) / len(f_result)
    b_avg_len = sum(len(w) for w in b_result) / len(b_result)
    if abs(f_avg_len - b_avg_len) > 0.1:
        return f_result if f_avg_len > b_avg_len else b_result
    
    # 规则4: 都相同 → 选BMM（统计表明逆向更优）
    return b_result
```

**测试验证**: 
- "研究生命起源": FMM=`研究生/命/起源`, BMM=`研究/生命/起源`
  - 词数相同(3), 单字数FMM=1, BMM=0 → 选BMM ✓
- "这个苹果不大好吃": FMM=BMM=`这/个/苹果/不大/好吃` → 直接返回

**预期效果**: 歧义集一致率从45%提升至55-60%

**日志更新**: 在 `logs/phase4_bimm_hmm_opt.txt` 记录新增规则的决策案例

---

### 步骤3：补充OOV词典新词

**文件**: `data/dict/dict_oov.txt`

**补充词汇**（基于2024-2025热点）:
```
多模态大模型
具身智能机器人
低空经济产业
新质生产力发展
谷子经济文化
搭子社交模式
电子榨菜内容
听劝式营销
脆皮大学生
显眼包行为
特种兵式旅游
多巴胺穿搭
情绪价值消费
科目三舞蹈
挖呀挖儿歌
尊嘟假嘟梗
黑神话悟空游戏
Citywalk城市漫步
DeepSeek人工智能
Kimi智能助手
Sora视频生成
AIGC内容创作
RAG检索增强
Agent智能体
通义千问大模型
文心一言大模型
```

**说明**: 虽然与jieba粒度可能不一致，但确保词典覆盖最新词汇

---

### 步骤4：重新运行实验并更新日志

**执行脚本**: `scripts/run_all_experiments.py`

**验证指标**:
- basic集一致率: 25% → 目标30-35%
- ambiguity集一致率: 45-50% → 目标55-60%
- oov集一致率: 0% → 保持（标注标准差异，非算法问题）
- long集一致率: 0% → 目标10-15%（通过Layer2优化）

**日志更新清单**:
1. `logs/phase4_bimm_hmm_opt.txt`: 记录BiMM新规则决策案例
2. `logs/phase5_hybrid_jieba.txt`: 记录Layer2触发率变化
3. `results/error_analysis.txt`: 重新生成错误分析
4. `results/RUN_SUMMARY.txt`: 更新对比数据

---

### 步骤5：更新plan.md

**文件**: `plan.md`

**更新内容**:
1. 在"阶段2：自主设计改进"部分添加本次优化的详细说明
2. 在"阶段4：深度分析"部分补充：
   - Layer2触发条件优化的设计思考
   - BiMM融合规则演进的个人见解
   - 标注标准差异对评估的影响分析
3. 更新"实验预期成果"中的准确率目标

---

## 技术细节说明

### 为什么选择这些改进？

1. **Layer2触发优化**: 
   - 根因：HMM在小样本训练下倾向于过度切分
   - 策略：提高触发门槛，仅在确信为未登录词时使用HMM
   - 收益：直接解决long_overcut错误类型

2. **BiMM规则增强**:
   - 根因：简单规则无法区分"词数相同但质量不同"的情况
   - 策略：引入平均词长作为第三优先级指标
   - 收益：提升歧义处理能力，符合"最长匹配"直觉

3. **词典补充**:
   - 根因：2024-2025新词缺失
   - 策略：系统性补充网络热词和专业术语
   - 收益：提升词典覆盖率（虽不保证与jieba一致）

### 为什么不实施其他建议？

- **N-gram统计模型**: 需要额外训练bigram/trigram概率，与当前架构不兼容
- **自适应MAX_LEN**: 实现复杂度高，收益不明确（当前MAX_LEN=7已足够）
- **英文实体识别**: 题目要求输入为中文，无需处理

---

## 验收标准

### 量化指标
- ✅ 歧义集 `hybrid_agree_rate_improvement` ≥ 0.05（当前已达标：50%-45%=5%）
- 🎯 basic集一致率提升至 ≥ 30%
- 🎯 long集一致率提升至 ≥ 10%

### 质性指标
- Layer2 HMM触发率下降（减少不必要切分）
- BiMM在歧义案例上的决策更合理
- 错误分析报告中long_overcut类型减少

---

## 风险控制

1. **回滚方案**: 保留原始代码备份，若改进后效果变差可快速回退
2. **增量验证**: 每步改进后单独测试，确保不引入新问题
3. **日志追踪**: 详细记录每次修改的实验数据，便于对比分析