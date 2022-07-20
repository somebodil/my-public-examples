import argparse
import logging
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
from transformers import set_seed

from util import train_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Word2VecDataset(Dataset):
    def __init__(self, dataset, vocab_size):
        self.dataset = dataset
        self.vocab_size = vocab_size

        self.data = {
            'center': [],
            'context': [],
        }

        for pair_list in dataset['moving_window']:
            for pair in pair_list:
                self.data['center'].append(pair[0])
                self.data['context'].append(pair[1])

        self.data_len = len(self.data['center'])

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return {
            'input_ids': self.data['center'][idx],
            'labels': self.data['context'][idx],
        }


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.expand = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        # Encode input to lower-dimensional representation
        hidden = self.embed(input_ids)
        # Expand hidden layer to predictions
        logits = self.expand(hidden)
        return logits


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    parser.add_argument('--batch_max_size', default=500, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--window_size', default=6, type=int)

    args = parser.parse_known_args()[0]
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    logger.info('[List of arguments]')
    for a in args.__dict__:
        logger.info(f'{a}: {args.__dict__[a]}')

    # Device & Seed --
    device = args.device
    set_seed(args.seed)

    # Hyper parameter --
    batch_max_size = args.batch_max_size
    epochs = args.epochs
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
    dataloader = DataLoader(dataset, batch_size=batch_max_size)
    model = Word2Vec(vocab_size, embed_size)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def loss_fn(predicts, batch, batch_size):
        return criterion(predicts, batch['labels'])

    def after_each_step_fn(train_callback_args):
        train_callback_args.clear_train_score_args()
        if train_callback_args.is_start_of_train() or train_callback_args.is_end_of_train():
            def get_k_similar_words(w, dm, k=10):
                idx = tok2id[w]
                dists = dm[idx]
                ind = np.argpartition(dists, k)[:k + 1]
                ind = ind[np.argsort(dists[ind])][1:]
                out = [(i, id2tok[i], dists[i]) for i in ind]
                return out

            word2vec_param = train_callback_args.best_model.expand.weight.cpu().detach().numpy()  # using latter matrix
            tokens = ['good', 'father', 'school', 'hate']
            distance_matrix = distance.squareform(distance.pdist(word2vec_param, 'cosine'))

            for word in tokens:
                logger.info(f'{word} {[t[1] for t in get_k_similar_words(word, distance_matrix)]}')

    train_model(
        epochs,
        device,
        dataloader,
        model,
        loss_fn,
        optimizer,
        after_each_step_fn=after_each_step_fn
    )


if __name__ == "__main__":
    main()
