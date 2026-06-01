# 中文分词算法实验

独立实现 FMM、BMM、BiMM、HMM（+1 平滑、O(4) Viterbi）、三层混合模型；`jieba` **仅**用于 `src/evaluate.py` 基准对比。

## 环境

```bash
pip install -r requirements.txt
```

## 一键运行

```bash
python scripts/run_all_experiments.py
```

## 需手动下载的资源

### 1. THUOCL 词表（推荐）

- 打开 [http://thuocl.thunlp.org/](http://thuocl.thunlp.org/)
- 下载 **IT词汇、财经词汇、成语俗语、地名、动物、饮食**（至少 3 个）`.txt`
- 放入：`data/raw/thuocl/*.txt`

若未下载，脚本会用 `data/raw/jieba_dict.txt` 补足 ≥2000 词（仅作词表，不调用 jieba 分词）。

### 2. 人民日报 1998 分词语料（HMM 训练）

- 下载 `19980101-train.txt` 等空格/词性标注文件
- 保存为：`data/raw/pku_train.txt`
- **随机抽取** `--num_lines` 行（默认 **10000**），非取前 N 行

```bash
python scripts/prepare_corpus.py --num_lines 10000
python scripts/train_hmm.py --num_lines 1000   # 对比实验可用 1000
```

## 实施规范（相对初版计划的修正）

| 项 | 说明 |
|----|------|
| THUOCL | 分类名为 **饮食**（非「食品」） |
| 语料采样 | **随机**抽取 `--num_lines` 行（默认 10000） |
| MAX_LEN | 上限 **7**（`src/dict_loader.py`） |
| HMM 平滑 | 转移分母 = count(s)+**4**；发射分母 = count(s)+**\|V\|** |
| Viterbi O(4) | 保留 **path_bp** 每步回溯指针数组 |
| 混合 Layer1 | `[词1, 连续单字片段1, 词2, ...]`，仅 OOV 片段进 HMM |
| Layer3 | 数字、日期、英文、**中文数字** |
| 错误分析 | 五类：歧义 / OOV / 长句 / **实体** / 其他 |
| 日志 | 可选字段 **error_type** |
| 验收 | 歧义集 `hybrid_agree_rate_improvement` ≥ 0.05（见 `timing_benchmark.txt`）；需 THUOCL+人民日报语料时更易达标，当前为 jieba 词典回退时可显示 CHECK |

## 目录与报告素材

| 路径 | 内容 |
|------|------|
| `logs/experiment_journal.txt` | 里程碑日志 |
| `logs/phase1~5*.txt` | 分阶段明细 |
| `results/segmentation/*.tsv` | 各算法分词结果 |
| `results/timing_benchmark.txt` | 耗时与 jieba 一致率 |
| `results/error_analysis.txt` | 五类错误分析 |
| `results/comparison_jieba.tsv` | BiMM/Hybrid vs jieba |

## 时间规划建议

- HMM 训练与优化：**1.5 天**
- 混合模型与 evaluate：**0.5 天**
