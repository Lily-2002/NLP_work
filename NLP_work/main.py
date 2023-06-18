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
import numpy as np
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
from Adam import CustomAdam


def read_ptb(path):   
    with open(path) as f:
        raw_text = f.read()
    s = [line.split() for line in raw_text.split('\n')]
    return s
def subsample(sentences, vocab): 
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]

    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())
    def keep(token):
        return (random.uniform(0, 1) <
                math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

def batchify(data): # small batch
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
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下⽂窗⼝中间i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                             min(len(line), i + 1 + window_size)))
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

def get_negatives(all_contexts, vocab, counter, K):
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def load_data_ptb(batch_size, max_window_size, num_noise_words,path):
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
    pred = torch.bmm(v,u.permute(0, 2, 1))
    return pred

class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))
def train(net, train_iter, val_iter,num_epochs, device=torch.device('cpu'), save_path=None):
    loss = SigmoidBCELoss()
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer =  CustomAdam(net.parameters())
    train_metric = Accumulator(2)
    val_metric = Accumulator(2)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        timer, train_num_batches = Timer(), len(train_iter)
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                 / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            train_metric.add(l.sum(), l.numel())
        train_loss = train_metric[0] / train_metric[1]
        val_loss = evaluate(net, val_iter, loss, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path is not None:
                torch.save(net.state_dict(), save_path)
        print(f'epoch {epoch+1}, train_loss {train_loss:.3f}, val_loss {val_loss:.3f}, '
              f'{train_metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
    return net
def evaluate(net, data_iter, loss, device):
    metric = Accumulator(2)
    for batch in data_iter:
        center, context_negative, mask, label = [data.to(device) for data in batch]
        pred = skip_gram(center, context_negative, net[0], net[1])
        l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
             / mask.sum(axis=1) * mask.shape[1])
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
    torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

def reduce_dimensions(path,words,test_iter,embed):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)
    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for i in range(len(vocab.idx_to_token)):
        W = embed.weight.data
        query = vocab.idx_to_token[i]
        x = W[vocab[query]]
        k = 3
        cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                          torch.sum(x * x) + 1e-9)
        topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
        vectors.append(topk)
        labels.append(query)
    # convert both lists into numpy vectors for reduction
    labels = np.asarray(labels)
    vectors = np.asarray(vectors)
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]
    plot(data, filename='C:/Users/fulian/Desktop/NLP_work/data/word-embedding-plot.html')

if __name__ == '__main__':
    path_train = r'C:\Users\fulian\Desktop\NLP_work\data\ptb.train.txt'
    embed_size = 100
    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = load_data_ptb(batch_size, max_window_size,
                                         num_noise_words,path_train)
    dropout_rate = 0.5
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size))
    num_epochs =10
    path_val = r'C:\Users\fulian\Desktop\NLP_work\data\ptb.valid.txt'
    val_iter, vocab_t = load_data_ptb(batch_size, max_window_size,
                                     num_noise_words, path_val)
    train(net, data_iter,val_iter, num_epochs, save_path='model.pt')
    get_similar_tokens('conference', 3, net[0])

    path_test = r'C:\Users\fulian\Desktop\NLP_work\data\ptb.test.txt'
    test_iter, vocab_test = load_data_ptb(batch_size, max_window_size,
                                         num_noise_words,path_test)

    x_vals, y_vals, labels = reduce_dimensions(path_test,vocab_test,test_iter,net[0])
    plot_with_plotly(x_vals, y_vals, labels)
