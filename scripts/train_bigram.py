"""训练bigram概率表，用于歧义消解"""
import json
import os
from collections import defaultdict


def parse_pku_corpus(file_path):
    """解析PKU语料，提取词语"""
    words_list = []
    with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割词（格式：词/词性）
            items = line.split()
            words = []
            for item in items:
                # 去掉词性标注，只保留词
                if '/' in item:
                    word = item.rsplit('/', 1)[0]
                    # 跳过时间标记（如19980101-01-001-001/m）
                    if not word.startswith('1998') and not word.startswith('2003'):
                        words.append(word)
            if words:
                words_list.append(words)
    return words_list


def train_bigram(words_list):
    """训练bigram模型"""
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    
    for words in words_list:
        # 添加句子开始和结束标记
        padded = ['<BOS>'] + words + ['<EOS>']
        for i in range(len(padded) - 1):
            w1 = padded[i]
            w2 = padded[i + 1]
            unigram_counts[w1] += 1
            bigram_counts[(w1, w2)] += 1
    
    # 计算bigram概率（带平滑）
    total_words = sum(unigram_counts.values())
    vocab_size = len(unigram_counts)
    
    # 转换为条件概率 P(w2|w1) = count(w1,w2) / count(w1)
    # 使用加1平滑
    bigram_probs = {}
    for (w1, w2), count in bigram_counts.items():
        prob = (count + 1) / (unigram_counts[w1] + vocab_size)
        bigram_probs[f"{w1}||{w2}"] = prob
    
    # 保存unigram用于计算先验概率
    unigram_probs = {}
    for word, count in unigram_counts.items():
        unigram_probs[word] = (count + 1) / (total_words + vocab_size)
    
    return {
        'bigram_probs': bigram_probs,
        'unigram_probs': unigram_probs,
        'vocab_size': vocab_size,
        'total_bigrams': len(bigram_counts)
    }


def main():
    corpus_path = 'data/raw/1998-01-2003版-带音.txt'
    output_path = 'models/bigram_params.json'
    
    if not os.path.exists(corpus_path):
        print(f"错误：语料文件不存在 {corpus_path}")
        return
    
    print("正在解析PKU语料...")
    words_list = parse_pku_corpus(corpus_path)
    print(f"共解析 {len(words_list)} 句子")
    
    print("正在训练bigram模型...")
    model = train_bigram(words_list)
    
    # 确保输出目录存在
    os.makedirs('models', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    
    print(f"bigram模型已保存到 {output_path}")
    print(f"词汇量: {model['vocab_size']}")
    print(f"bigram数量: {model['total_bigrams']}")


if __name__ == '__main__':
    main()