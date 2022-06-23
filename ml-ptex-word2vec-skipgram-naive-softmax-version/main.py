import argparse
import re
from collections import Counter
from datetime import datetime

import datasets
import numpy as np
import torch
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from scipy.spatial import distance
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import set_seed


class Word2VecDataset(Dataset):
    def __init__(self, dataset, vocab_size):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.data = [i for s in dataset['moving_window'] for i in s]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # return center, context as list


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.expand = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, inputs):
        # Encode input to lower-dimensional representation
        hidden = self.embed(inputs)
        # Expand hidden layer to predictions
        logits = self.expand(hidden)
        return logits


def pretrain_model(epochs, device, dataloader, model, loss_fn, optimizer, score_fn):
    model.to(device)
    loss_fn.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            center, context = batch
            center, context = center.to(device), context.to(device)

            optimizer.zero_grad()
            logits = model(center)
            loss = loss_fn(logits, context)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'\nTotal epoch loss for {epoch} epoch : {epoch_loss}')
        score_fn(model)


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=50000, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--window_size', default=6, type=int)

    args = parser.parse_known_args()[0]
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    # Device --
    device = args.device

    # Hyper parameter --
    set_seed(args.seed)
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    window_size = args.window_size
    embed_size = 200

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    dataset = datasets.load_dataset('tweets_hate_speech_detection', split="train")
    ss = SnowballStemmer('english')
    sw = stopwords.words('english')

    def split_tweet_into_tokens(row):
        tokenized_row = [ss.stem(i) for i in re.split(r" +", re.sub(r"[^a-z@# ]", "", row['tweet'].lower())) if (i not in sw) and len(i)]
        row['all_tokens'] = tokenized_row
        return row

    dataset = dataset.map(split_tweet_into_tokens)

    counts = Counter([i for s in dataset['all_tokens'] for i in s])
    counts = {k: v for k, v in counts.items() if v > 10}  # Filter rare tokens
    vocab = list(counts.keys())
    vocab_size = len(vocab)
    id2tok = dict(enumerate(vocab))
    tok2id = {token: id for id, token in id2tok.items()}

    def remove_rare_tokens(row):
        row['tokens'] = [t for t in row['all_tokens'] if t in vocab]
        return row

    dataset = dataset.map(remove_rare_tokens)

    # win_size 에 따라 center 하나에 대해서 (center, context) 가 2개, 3개 생성됨
    # 이해하기 쉽게 win_size=batch_size로 하자
    def slide_window_for_tokenized_tweet(row, win_size=window_size):
        doc = row['tokens']
        out = []
        for i, wd in enumerate(doc):
            target = tok2id[wd]
            window = [i + j for j in
                      range(-win_size, win_size + 1, 1)
                      if (i + j >= 0) &
                      (i + j < len(doc)) &
                      (j != 0)]

            out += [(target, tok2id[doc[w]]) for w in window]
        row['moving_window'] = out
        return row

    dataset = dataset.map(slide_window_for_tokenized_tweet)
    dataset = Word2VecDataset(dataset, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = Word2Vec(vocab_size, embed_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    def print_similar_words(m):

        def get_distance_matrix(wv, met):
            dist_matrix = distance.squareform(distance.pdist(wv, met))
            return dist_matrix

        def get_k_similar_words(wp, dm, k=10):
            idx = tok2id[wp]
            dists = dm[idx]
            ind = np.argpartition(dists, k)[:k + 1]
            ind = ind[np.argsort(dists[ind])][1:]
            out = [(i, id2tok[i], dists[i]) for i in ind]
            return out

        word2vec_param = m.expand.weight.cpu().detach().numpy()  # using latter matrix
        tokens = ['good', 'father', 'school', 'hate']

        distance_matrix = get_distance_matrix(word2vec_param, 'cosine')

        print("")
        for word in tokens:
            print(word, [t[1] for t in get_k_similar_words(word, distance_matrix)])

    pretrain_model(epochs, device, dataloader, model, loss_fn, optimizer, print_similar_words)


if __name__ == "__main__":
    main()
