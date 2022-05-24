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
from transformers import BertModel, BertTokenizer
from transformers import set_seed


class BertForRegression(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_labels):
        super(BertForRegression, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        linear_out = self.linear(bert_out)
        sigmoid_out = self.sigmoid(linear_out) * 5
        return sigmoid_out.squeeze()


def get_score(output, label):
    score = stats.pearsonr(output, label)[0]
    return score


def train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer):
    model.to(device)
    loss_fn.to(device)

    best_val_score = 0
    best_model = None

    for t in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            predict = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            loss = loss_fn(predict, batch['labels'])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_pred = []
            val_label = []

            for _, val_batch in enumerate(validation_dataloader):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                predict = model(val_batch['input_ids'], val_batch['attention_mask'], val_batch['token_type_ids'])  # validate
                loss = loss_fn(predict, val_batch['labels'])

                val_loss += loss.item()
                val_pred.extend(predict.clone().cpu().tolist())
                val_label.extend(val_batch['labels'].clone().cpu().tolist())

            val_score = get_score(np.array(val_pred), np.array(val_label))
            if best_val_score < val_score:
                best_val_score = val_score
                best_model = copy.deepcopy(model)

            print(f"\nValidation loss / cur_val_score / best_val_score : {val_loss} / {val_score} / {best_val_score}")

    return best_model


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-cased', type=str)
    parser.add_argument('--hidden_size', default='768', type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    args = parser.parse_args()
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
    hidden_size = args.hidden_size

    # Dataset --
    train_dataset = load_dataset('glue', 'stsb', split="train[:10%]")
    validation_dataset = load_dataset('glue', 'stsb', split="validation[:10%]")
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

    model = BertForRegression(model_name, hidden_size, data_labels_num)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer)


if __name__ == '__main__':
    main()
