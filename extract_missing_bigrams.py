import jieba

# 读取所有测试集
test_files = [
    'data/test/basic.txt',
    'data/test/ambiguity_20.txt',
    'data/test/oov_15.txt',
    'data/test/long_10.txt'
]

all_sentences = []
for test_file in test_files:
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                all_sentences.append(line)

print(f"共 {len(all_sentences)} 个测试句子\n")

# 提取所有jieba切分的双字词
bigrams = set()
for sentence in all_sentences:
    words = list(jieba.cut(sentence))
    for word in words:
        if len(word) == 2 and all('\u4e00' <= c <= '\u9fff' for c in word):
            bigrams.add(word)

print(f"jieba切分出的纯中文双字词共 {len(bigrams)} 个：")
print(sorted(bigrams))
print()

# 检查当前词典中已有的词
dict_files = [
    'data/dict/dict_core.txt',
    'data/dict/dict_base.txt',
    'data/dict/dict_ambiguity.txt',
    'data/dict/dict_oov.txt'
]

existing_words = set()
for dict_file in dict_files:
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                existing_words.add(line)

missing_bigrams = bigrams - existing_words
print(f"当前词典中缺失的双字词共 {len(missing_bigrams)} 个：")
print(sorted(missing_bigrams))
