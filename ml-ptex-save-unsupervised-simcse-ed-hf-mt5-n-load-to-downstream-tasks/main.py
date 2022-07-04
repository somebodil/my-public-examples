import argparse
import copy
import os
from datetime import datetime

import pandas as pd
import torch
from datasets import load_dataset
from scipy import stats
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import set_seed, T5Tokenizer, MT5Config, MT5EncoderModel


class Mt5ForRegression(nn.Module):
    def __init__(self, model_name, num_labels, state_dict=None, config_dict=None):
        super(Mt5ForRegression, self).__init__()

        if model_name is None:
            if state_dict is None or config_dict is None:
                raise ValueError("If model_name is None, state_dict and config must be specified.")

            self.config = MT5Config.from_dict(config_dict)
            self.mt5 = MT5EncoderModel(config=self.config)
            self.mt5.load_state_dict(state_dict)

        else:
            if state_dict is not None or config_dict is not None:
                raise ValueError("If model_name is not None, state_dict and config must be None.")

            self.config = MT5Config.from_pretrained(model_name)
            self.mt5 = MT5EncoderModel.from_pretrained(model_name)

        self.linear = nn.Linear(in_features=self.config.hidden_size, out_features=num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, **kwargs):
        mt5_out = self.mt5(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
        pooler_out = torch.mean(mt5_out, dim=1)
        linear_out = self.linear(pooler_out)
        sigmoid_out = self.sigmoid(linear_out) * 5
        return sigmoid_out.squeeze()


class MT5ForFurtherTrain(nn.Module):
    def __init__(self, model_name, temperature):
        super(MT5ForFurtherTrain, self).__init__()

        self.config = MT5Config.from_pretrained(model_name)
        self.mt5 = MT5EncoderModel.from_pretrained(model_name)

        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    def forward(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]

        # copy
        input_ids = input_ids.unsqueeze(1).repeat(1, 2, 1)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, 2, 1)

        # flat
        input_ids = input_ids.view((-1, input_ids.shape[-1]))
        attention_mask = attention_mask.view((-1, input_ids.shape[-1]))

        # encode
        mt5_out = self.mt5(input_ids, attention_mask)[0]
        pooler_out = torch.mean(mt5_out, dim=1)

        # revert flat
        pooler_out = pooler_out.view((batch_size, 2, pooler_out.shape[-1]))

        # cos sim
        z1, z2 = pooler_out[:, 0], pooler_out[:, 1]
        cos_sim_out = self.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature

        return cos_sim_out


class SimCSEDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_length):
        self.len = len(data_frame['sentence'])
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.data_frame['sentence'][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")

        return {
            "input_ids": encodings['input_ids'].squeeze(0),
            "attention_mask": encodings['attention_mask'].squeeze(0)
        }


def load_model_config(path):
    config = torch.load(path, map_location='cpu')
    return config['model_name'], config['model_state_dict'], config['model_config_dict']


def save_model_config(path, model_name, model_state_dict, model_config_dict):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    torch.save({
        'model_name': model_name,
        'model_state_dict': model_state_dict,
        'model_config_dict': model_config_dict
    }, path)


def pretrain_model(epochs, device, dataloader, model, loss_fn, optimizer, _, model_save_fn):
    model.to(device)
    loss_fn.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), torch.arange(dataloader.batch_size).long().to(device)

            optimizer.zero_grad()
            predict = model(input_ids, attention_mask)
            loss = loss_fn(predict, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 1000 == 0 or i == len(dataloader) - 1:
                model_save_fn(model)
                print(f'\n{i}th iteration (train loss): ({train_loss:.4})')
                train_loss = 0


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
            predict = model(**batch)
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
            predict = model(**batch)
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
    parser.add_argument('--model_name', default='google/mt5-small', type=str)  # should be t5 base
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--pretrain_dataset', default='kowikitext_20200920.train', type=str)
    parser.add_argument('--model_state_name', default='model_state.pt', type=str)  # for downstream tasks, if model_state.pt exist, model_name will be ignored
    parser.add_argument('--task', default='klue_sts', type=str)

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
    model_name = args.model_name
    batch_size = args.batch_size
    seq_max_length = args.seq_max_length
    epochs = args.epochs
    learning_rate = args.lr
    task = args.task
    model_state_name = args.model_state_name

    if task == "klue_sts":
        # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
        if model_state_name:
            model_name, model_state_dict, model_config_dict = load_model_config(f'checkpoint/{model_state_name}')

        train_dataset = load_dataset('klue', 'sts', split="train[:100]")  # FIXME change back to train[:80%]
        validation_dataset = load_dataset('klue', 'sts', split="train[-100:]")  # FIXME change back to train[-20%:]
        test_dataset = load_dataset('klue', 'sts', split="validation")
        data_labels_num = 1

        tokenizer = T5Tokenizer.from_pretrained(model_name)

        def format_input(examples):
            encoded = tokenizer(examples['sentence1'], examples['sentence2'], max_length=seq_max_length, truncation=True, padding='max_length')
            return encoded

        def format_target(example):
            return {'labels': example['labels']['label']}

        def preprocess_dataset(dataset):
            dataset = dataset.map(format_input, batched=True)
            dataset = dataset.map(format_target)
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            return dataset

        train_dataset = preprocess_dataset(train_dataset)
        validation_dataset = preprocess_dataset(validation_dataset)
        test_dataset = preprocess_dataset(test_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        def score_fn(pred, label):
            return stats.pearsonr(pred, label)[0]

        if model_state_name:
            model = Mt5ForRegression(None, data_labels_num, model_state_dict, model_config_dict)
        else:
            model = Mt5ForRegression(model_name, data_labels_num)

        loss_fn = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        model = train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer, score_fn)
        test_loss, test_score = evaluate_model(device, test_dataloader, model, loss_fn, score_fn)
        print(f"\nTest (loss, score) with best model: ({test_loss:.4} / {test_score:.4})")

    else:
        # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
        temperature = args.temperature
        pretrain_dataset = args.pretrain_dataset

        with open(f"dataset/{pretrain_dataset}", encoding='utf8') as f:
            lines = []
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

            df_train = pd.DataFrame(lines, columns=['sentence'])

        tokenizer = T5Tokenizer.from_pretrained(model_name)
        train_dataset = SimCSEDataset(df_train, tokenizer, seq_max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        model = MT5ForFurtherTrain(model_name, temperature)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        def model_save_fn(pretrained_model):
            save_model_config(f'checkpoint/{model_state_name}', model_name, pretrained_model.mt5.state_dict(), pretrained_model.config.to_dict())

        pretrain_model(epochs, device, train_dataloader, model, loss_fn, optimizer, None, model_save_fn)


if __name__ == "__main__":
    main()
