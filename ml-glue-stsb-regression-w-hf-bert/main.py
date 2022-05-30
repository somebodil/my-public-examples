import argparse
import copy
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from scipy import stats
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import set_seed


class BertForRegression(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertForRegression, self).__init__()

        self.hidden_size = BertConfig.from_pretrained(bert_model_name).hidden_size
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        _, bert_out = self.bert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'], return_dict=False)
        linear_out = self.linear(bert_out)
        sigmoid_out = self.sigmoid(linear_out) * 5
        return sigmoid_out.squeeze()


def get_score(output, label):
    score = stats.pearsonr(output, label)[0]
    return score


def validate_model(device, dataloader, model, loss_fn):
    model.to(device)
    loss_fn.to(device)

    loss = 0
    pred = []
    label = []

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            predict = model(batch)
            loss = loss_fn(predict, batch['labels'])
            loss += loss.item()
            pred.extend(predict.clone().cpu().tolist())
            label.extend(batch['labels'].clone().cpu().tolist())

        score = get_score(np.array(pred), np.array(label))

    return loss, score


def train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer):
    model.to(device)
    loss_fn.to(device)

    best_score = 0
    best_model = None

    for t in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            predict = model(batch)
            loss = loss_fn(predict, batch['labels'])
            loss.backward()
            optimizer.step()

        cur_loss, cur_score = validate_model(device, validation_dataloader, model, loss_fn)
        if best_score < cur_score:
            best_score = cur_score
            best_model = copy.deepcopy(model)

        print(f"\nValidation loss / cur_val_score / best_score : {cur_loss} / {cur_score} / {best_score}")

    return best_model


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-cased', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=1, type=int)  # TODO dev
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
    train_dataset = load_dataset('glue', 'stsb', split="train")
    validation_dataset = load_dataset('glue', 'stsb', split="validation")
    data_labels_num = 1

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode_input(examples):
        encoded = tokenizer(examples['sentence1'], examples['sentence2'], max_length=seq_max_length, truncation=True, padding='max_length')
        return encoded

    def format_target(examples):
        changed = {'labels': examples['label']}
        return changed

    train_dataset = train_dataset.map(encode_input)
    train_dataset = train_dataset.map(format_target)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])

    validation_dataset = validation_dataset.map(encode_input)
    validation_dataset = validation_dataset.map(format_target)
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])

    model = BertForRegression(model_name, data_labels_num)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer)
    best_loss, best_acc = validate_model(device, validation_dataloader, model, loss_fn)
    print(f"\nValidation best_loss / best_score with best model: {best_loss} / {best_acc}")


if __name__ == '__main__':
    main()
