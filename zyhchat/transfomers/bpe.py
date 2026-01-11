import re
from collections import defaultdict, Counter

def get_vocab(corpus):
    """统计每个“词”的出现频率"""
    vocab = defaultdict(int)
    for line in corpus:
        for word in line.strip().split():
            # 在词尾加 </w> 标记
            vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    """统计所有相邻符号对的频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """将 vocab 中的所有 pair 合并"""
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        new_word = p.sub(''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def bpe_train(corpus, vocab_size=50):
    vocab = get_vocab(corpus)
    merges = []  # 记录合并规则
    
    while len(merges) < vocab_size - len(set(''.join(vocab).replace(' ', '').replace('</w>', ''))):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_vocab(best, vocab)
        print(f"合并: {best} → {''.join(best)}")
    
    # 构建最终词表
    vocab_set = set()
    for word in vocab:
        vocab_set.update(word.split())
    return merges, vocab_set

# 分词函数（使用训练好的合并规则）
def bpe_encode(word, merges):
    # 初始化：每个字符分开，加 </w>
    tokens = list(word) + ['</w>']
    while True:
        # 找到当前 tokens 中最早匹配 merges 的 pair
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        # 按 merges 顺序找第一个可合并的
        merged = False
        for pair in merges:
            if pair in pairs:
                i = pairs.index(pair)
                # 合并 tokens[i] 和 tokens[i+1]
                tokens = tokens[:i] + [''.join(pair)] + tokens[i+2:]
                merged = True
                break
        if not merged:
            break
    return tokens

# ===== 测试 =====
if __name__ == "__main__":
    corpus = [
        "low lower lowest",
        "new newer newest",
        "how hello tester writer"
    ]
    
    merges, vocab = bpe_train(corpus, vocab_size=30)
    print("\n最终词表大小:", len(vocab))
    print("部分词表:", sorted(list(vocab))[:20])
    
    # 测试分词
    test_word = "tester"
    encoded = bpe_encode(test_word, merges)
    print(f"\n'{test_word}' → {encoded}")