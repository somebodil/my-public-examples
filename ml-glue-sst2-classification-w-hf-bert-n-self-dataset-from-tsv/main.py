import argparse
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, set_seed, BertTokenizer, BertConfig


class BertForClassification(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertForClassification, self).__init__()

        self.hidden_size = BertConfig.from_pretrained(bert_model_name).hidden_size
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=self.hidden_size,
                                out_features=num_labels)

    def forward(self, batch):
        _, bert_out = self.bert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], return_dict=False)
        linear_out = self.linear(bert_out)
        return linear_out


class GlueSst2Dataset(Dataset):

    def __init__(self, data_frame, tokenizer, max_length):
        self.len = len(data_frame['label'])
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.data_frame['sentence'][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")

        return {
            "labels": self.data_frame['label'][idx],
            "input_ids": encodings['input_ids'].squeeze(0),
            "attention_mask": encodings['attention_mask'].squeeze(0)
        }


def evaluate_model(device, dataloader, model, loss_fn, score_fn):
    model.to(device)
    loss_fn.to(device)

    eval_loss = 0
    eval_pred = []
    eval_label = []

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            predict = model(batch)
            loss = loss_fn(predict, batch['labels'])

            eval_loss += loss.clone().cpu().item()
            eval_pred.extend(predict.clone().cpu().tolist())
            eval_label.extend(batch['labels'].clone().cpu().tolist())

    eval_score = score_fn(eval_pred, eval_label)
    return eval_loss, eval_score


def train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer, score_fn):
    model.to(device)
    loss_fn.to(device)

    best_model = None
    best_val_score = -10
    best_val_epoch = -10
    best_val_loss = -10

    for epoch in range(1, epochs + 1):
        train_loss = 0
        train_pred = []
        train_label = []

        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            predict = model(batch)
            loss = loss_fn(predict, batch['labels'])
            loss.backward()
            optimizer.step()

            train_loss += loss.clone().cpu().item()
            train_pred.extend(predict.clone().cpu().tolist())
            train_label.extend(batch['labels'].clone().cpu().tolist())

        train_score = score_fn(train_pred, train_label)
        val_loss, val_score = evaluate_model(device, validation_dataloader, model, loss_fn, score_fn)
        if best_val_score < val_score:
            best_val_score = val_score
            best_val_epoch = epoch
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print(f'\nEpoch {epoch} (train loss, train score), (Val loss, Val score): ({train_loss:.4}, {train_score:.4}), ({val_loss:.4}, {val_score:.4})')

    print(f'Val best (epoch, loss, score) : ({best_val_epoch}, {best_val_loss:0.4}, {best_val_score:.4})')
    return best_model


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-cased', type=str)  # should be bert-xxx
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

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
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    seq_max_length = args.seq_max_length
    model_name = args.model_name

    # Dataset --
    df_train = pd.read_csv('./glue_sst2_train.tsv', delimiter='\t')
    df_train = df_train[:50]  # FIXME remove
    df_train, df_val = np.split(df_train, [int(.8 * len(df_train))])
    df_val = df_val.reset_index(drop=True)

    df_test = pd.read_csv('./glue_sst2_dev.tsv', delimiter='\t')
    dataset_num_labels = 2

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = GlueSst2Dataset(df_train, tokenizer, seq_max_length)
    validation_dataset = GlueSst2Dataset(df_val, tokenizer, seq_max_length)
    test_dataset = GlueSst2Dataset(df_test, tokenizer, seq_max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = BertForClassification(model_name, dataset_num_labels)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    def score_fn(pred, label):
        return accuracy_score(np.argmax(pred, axis=1), label)

    model = train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer, score_fn)
    test_loss, test_score = evaluate_model(device, test_dataloader, model, loss_fn, score_fn)
    print(f"\nTest (loss, score) with best model: ({test_loss:.4} / {test_score:.4})")


if __name__ == "__main__":
    main()
