手动下载说明（网络受限时请本地浏览器下载）

1) THUOCL 词表 http://thuocl.thunlp.org/
   推荐：IT词汇、财经词汇、成语俗语、地名、动物、饮食（注意是「饮食」不是「食品」）
   放入：data/raw/thuocl/*.txt

2) 人民日报1998分词语料（当前未检测到，HMM 使用合成语料）
   保存为：data/raw/pku_train.txt（或 19980101-train.txt，或任意 >500KB 的 raw/*.txt）
   脚本将随机抽取 N 行（默认10000）：python scripts/prepare_corpus.py --num_lines 10000

   已就绪：THUOCL 6 词表 -> dict_base 95886 词

3) 备选词典（已内置 jieba_dict.txt 时可跳过 THUOCL）
