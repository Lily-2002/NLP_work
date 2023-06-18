import math
import os
import random
import torch
from collections import Counter
from vocab import *
from torch import nn
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE

def read_ptb(path):   # 分割句子
    with open(path) as f:
        raw_text = f.read()
    s = [line.split() for line in raw_text.split('\n')]
    return s
def subsample(sentences, vocab):  # 下采样：有概率的丢弃高无意义高频词
    # 排除未知词元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]

    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())
    def keep(token):
        return (random.uniform(0, 1) <
                math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)
# 中心词 上下文提取
def batchify(data):
    """返回带有负采样的跳元模型的⼩批量样本"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += \
        [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
    contexts_negatives), torch.tensor(masks), torch.tensor(labels))

def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
    # 要形成“中⼼词-上下⽂词”对，每个句⼦⾄少需要有2个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下⽂窗⼝中间i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                             min(len(line), i + 1 + window_size)))
            # 从上下⽂词中排除中⼼词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

def compare_counts(token,sentences,subsampled):
    return (f'"{token}"的数量：'
            f'之前={sum([l.count(token) for l in sentences])}, '
            f'之后={sum([l.count(token) for l in subsampled])}')

def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下⽂词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

def load_data_ptb(batch_size, max_window_size, num_noise_words,path):
    """下载PTB数据集，然后将其加载到内存中"""
    # num_workers = 4
    sentences = read_ptb(path)
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives
        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
            self.negatives[index])
        def __len__(self):
            return len(self.centers)
    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True,
        collate_fn=batchify)
    return data_iter, vocab

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

class SigmoidBCELoss(nn.Module):
# 带掩码的⼆元交叉熵损失
    def __init__(self):
        super().__init__()
    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

def train(net, data_iter, lr, num_epochs, device= torch.device('cpu'),save_path=None):
    loss = SigmoidBCELoss()
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = Animator(xlabel='epoch', ylabel='loss',
        xlim=[1, num_epochs])
# 规范化的损失之和，规范化的损失数
    metric = Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                    data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                            (metric[0] / metric[1],))
        print(f'loss {metric[0] / metric[1]:.3f}, '
                f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
    if save_path is not None:
        torch.save(net.state_dict(), save_path)
        # 返回训练后的模型
    return net

def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
    torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]: # 删除输⼊词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')


if __name__ == '__main__':
    path = r'C:\Users\fulian\Desktop\NLP_work\data\ptb.train.txt'
    embed_size = 100
    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = load_data_ptb(batch_size, max_window_size,
                                         num_noise_words,path)
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size))
    lr, num_epochs = 0.002, 5
    train(net, data_iter, lr, num_epochs,save_path = 'model_epoch1.pt')
    get_similar_tokens('the', 3, net[0])
    torch.save(net.state_dict(), 'my_model.pt')
