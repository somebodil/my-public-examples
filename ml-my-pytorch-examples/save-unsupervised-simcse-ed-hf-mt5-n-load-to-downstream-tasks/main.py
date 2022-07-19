import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy import stats
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed, T5Tokenizer, MT5Config, MT5EncoderModel

from util_fn import train_model, evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class Mt5ForClassification(nn.Module):
    def __init__(self, model_name, num_labels, state_dict=None, config_dict=None):
        super(Mt5ForClassification, self).__init__()

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

    def forward(self, input_ids, attention_mask, **kwargs):
        mt5_out = self.mt5(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
        pooler_out = torch.mean(mt5_out, dim=1)
        linear_out = self.linear(pooler_out)
        return linear_out.squeeze()


class MT5ForFurtherTrain(nn.Module):
    def __init__(self, model_name, temperature):
        super(MT5ForFurtherTrain, self).__init__()

        self.config = MT5Config.from_pretrained(model_name)  # uses default dropout rate = 0.1
        self.mt5 = MT5EncoderModel.from_pretrained(model_name)

        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    def forward(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]

        # copy (batch_size, seq_len) => (batch_size, 2, seq_len)
        input_ids = input_ids.unsqueeze(1).repeat(1, 2, 1)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, 2, 1)

        # flat => (batch_size * 2, seq_len)
        input_ids = input_ids.view((-1, input_ids.shape[-1]))
        attention_mask = attention_mask.view((-1, input_ids.shape[-1]))

        # encode => (batch_size * 2, hidden_size)
        mt5_out = self.mt5(input_ids, attention_mask)[0]
        pooler_out = torch.mean(mt5_out, dim=1)

        # revert flat => (batch_size, 2, hidden_size)
        pooler_out = pooler_out.view((batch_size, 2, pooler_out.shape[-1]))

        # cos sim
        z1, z2 = pooler_out[:, 0], pooler_out[:, 1]  # (batch_size, hidden_size)
        cos_sim_out = self.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature  # (batch_size, 1, hidden_size), (batch_size, hidden_size, 1)

        return cos_sim_out


class UnsupervisedSimCseDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_length, column_name='sentence'):
        self.len = len(data_frame)
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.column_name = column_name

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.data_frame[self.column_name][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")

        return {
            "input_ids": encodings['input_ids'].squeeze(0),
            "attention_mask": encodings['attention_mask'].squeeze(0),
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


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    parser.add_argument('--model_name', default='google/mt5-small', type=str)  # should be t5 base
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--batch_max_size', default=12, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--task', default='unsup_simcse', type=str)
    parser.add_argument('--model_state_name', default='model_state.pt', type=str)  # for unsup_simcse - should exist, for other tasks - if model_state.pt exist, model_name will be ignored

    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--pretrain_dataset', default='kowikitext_20200920.train', type=str)  # for pretrained task

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
    model_name = args.model_name
    seq_max_length = args.seq_max_length
    batch_max_size = args.batch_max_size
    epochs = args.epochs
    learning_rate = args.lr

    task = args.task
    model_state_name = args.model_state_name

    if task == "klue_sts":
        # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
        train_dataset = load_dataset('klue', 'sts', split="train[:100]")  # FIXME change back to train[:80%]
        validation_dataset = load_dataset('klue', 'sts', split="train[-100:]")  # FIXME change back to train[-20%:]
        test_dataset = load_dataset('klue', 'sts', split="validation")
        data_labels_num = 1

        if model_state_name:
            model_name, model_state_dict, model_config_dict = load_model_config(f'checkpoint/{model_state_name}')
            model = Mt5ForRegression(None, data_labels_num, model_state_dict, model_config_dict)
        else:
            model = Mt5ForRegression(model_name, data_labels_num)

        tokenizer = T5Tokenizer.from_pretrained(model_name)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

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

        train_dataloader = DataLoader(train_dataset, batch_size=batch_max_size)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_max_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_max_size)

        def fn_loss(predicts, batch, batch_size):
            return criterion(predicts, batch['labels'])

        def fn_score(pred, label):
            return stats.pearsonr(pred, label)[0]

        def cb_after_each_step(train_callback_args):
            if train_callback_args.is_end_of_epoch():
                train_score = 0
                for i in range(train_callback_args.train_num_batches):
                    train_score += fn_score(train_callback_args.train_predicts[i], train_callback_args.train_batches[i]['labels'])

                train_score /= train_callback_args.train_num_batches

                val_loss, val_score = evaluate_model(device, validation_dataloader, train_callback_args.model, fn_loss, fn_score, param_disable_tqdm=True)
                if train_callback_args.is_greater_than_best_val_score(val_score):
                    train_callback_args.set_best_val_args(val_loss, val_score)

                logger.info(f'Epoch {train_callback_args.epoch} train loss, train score, val loss, val score: [{train_callback_args.train_loss:.2}, {train_score:.2}, {val_loss:.2}, {val_score:.2}]')
                train_callback_args.clear_train_score_args()

        model, best_val_epoch, best_val_loss, best_val_score = train_model(
            epochs,
            device,
            train_dataloader,
            model,
            fn_loss,
            optimizer,
            cb_after_each_step=cb_after_each_step,
            param_disable_tqdm=True
        )

        test_loss, test_score = evaluate_model(device, test_dataloader, model, fn_loss, fn_score, param_disable_tqdm=True)
        logger.info(f"Test (loss, score) with best val model (epoch, loss, score) : ({test_loss:.2} / {test_score:.2}), ({best_val_epoch}, {best_val_loss:.2} / {best_val_score:.2})")

    elif task == "klue_nli":
        # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
        train_dataset = load_dataset('klue', 'nli', split="train[:100]")  # FIXME change back to train[:80%]
        validation_dataset = load_dataset('klue', 'nli', split="train[-100:]")  # FIXME change back to train[-20%:]
        test_dataset = load_dataset('klue', 'nli', split="validation")
        data_labels_num = 3

        if model_state_name:
            model_name, model_state_dict, model_config_dict = load_model_config(f'checkpoint/{model_state_name}')
            model = Mt5ForClassification(None, data_labels_num, model_state_dict, model_config_dict)
        else:
            model = Mt5ForClassification(model_name, data_labels_num)

        tokenizer = T5Tokenizer.from_pretrained(model_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        def format_input(examples):
            encoded = tokenizer(examples['premise'], examples['hypothesis'], max_length=seq_max_length, truncation=True, padding='max_length')
            return encoded

        def format_target(example):
            return {'labels': example['label']}

        def preprocess_dataset(dataset):
            dataset = dataset.map(format_input, batched=True)
            dataset = dataset.map(format_target)
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            return dataset

        train_dataset = preprocess_dataset(train_dataset)
        validation_dataset = preprocess_dataset(validation_dataset)
        test_dataset = preprocess_dataset(test_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_max_size)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_max_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_max_size)

        def fn_loss(predicts, batch, batch_size):
            return criterion(predicts, batch['labels'])

        def fn_score(pred, label):
            return accuracy_score(label, np.argmax(pred, axis=1))

        def cb_after_each_step(train_callback_args):
            if train_callback_args.is_end_of_epoch():
                train_score = 0
                for i in range(train_callback_args.train_num_batches):
                    train_score += fn_score(train_callback_args.train_predicts[i], train_callback_args.train_batches[i]['labels'])

                train_score /= train_callback_args.train_num_batches

                val_loss, val_score = evaluate_model(device, validation_dataloader, train_callback_args.model, fn_loss, fn_score, param_disable_tqdm=True)
                if train_callback_args.is_greater_than_best_val_score(val_score):
                    train_callback_args.set_best_val_args(val_loss, val_score)

                logger.info(f'Epoch {train_callback_args.epoch} train loss, train score, val loss, val score: [{train_callback_args.train_loss:.2}, {train_score:.2}, {val_loss:.2}, {val_score:.2}]')
                train_callback_args.clear_train_score_args()

        model, best_val_epoch, best_val_loss, best_val_score = train_model(
            epochs,
            device,
            train_dataloader,
            model,
            fn_loss,
            optimizer,
            cb_after_each_step=cb_after_each_step,
            param_disable_tqdm=True
        )

        test_loss, test_score = evaluate_model(device, test_dataloader, model, fn_loss, fn_score, param_disable_tqdm=True)
        logger.info(f"Test (loss, score) with best val model (epoch, loss, score) : ({test_loss:.2} / {test_score:.2}), ({best_val_epoch}, {best_val_loss:.2} / {best_val_score:.2})")

    elif task == 'unsup_simcse':
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
        train_dataset = UnsupervisedSimCseDataset(df_train, tokenizer, seq_max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_max_size)
        model = MT5ForFurtherTrain(model_name, temperature)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        def fn_loss(predicts, batch, batch_size):
            labels = torch.arange(batch_size).long().to(device)  # [1,2,3,4,5]
            return criterion(predicts, labels)

        def cb_after_each_step(train_callback_args):
            if train_callback_args.is_step_interval(10):
                save_model_config(f'checkpoint/{model_state_name}', model_name, train_callback_args.best_model.mt5.state_dict(), train_callback_args.best_model.config.to_dict())
                logger.info(f'Saved model config at step {train_callback_args.get_cumulated_step()}')

        train_model(
            epochs,
            device,
            train_dataloader,
            model,
            fn_loss,
            optimizer,
            cb_after_each_step=cb_after_each_step
        )

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
