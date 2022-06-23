import argparse
import os
from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import set_seed, T5Tokenizer, MT5Config, MT5EncoderModel


class SentenceT5EncoderModel(nn.Module):
    def __init__(self, model_name):  # model_name should be mt5 related ...
        super(SentenceT5EncoderModel, self).__init__()
        self.hidden_size = MT5Config.from_pretrained(model_name).hidden_size
        self.mt5 = MT5EncoderModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, **kwargs):
        mt5_out = self.mt5(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
        sentence_out = torch.mean(mt5_out, dim=1)
        return sentence_out


class MT5ForFurtherTrain(nn.Module):
    def __init__(self, model_name, temperature):
        super(MT5ForFurtherTrain, self).__init__()
        self.sentence_t5 = SentenceT5EncoderModel(model_name)
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
        sentence_t5_out = self.sentence_t5(input_ids, attention_mask)

        # revert flat
        sentence_t5_out = sentence_t5_out.view((batch_size, 2, sentence_t5_out.shape[-1]))

        # cos sim
        z1, z2 = sentence_t5_out[:, 0], sentence_t5_out[:, 1]
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


def save_model_state(model_state, path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    torch.save(model_state, path)


def pretrain_model(epochs, device, dataloader, model, loss_fn, optimizer, _):
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
            if i % 5000 == 0:
                print(f'\nTrain loss for 5000 iteration: ({train_loss:.4})')
                train_loss = 0
                save_model_state(model.state_dict(), "checkpoint/_state.pt")


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='google/mt5-large', type=str)  # should be t5 base
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--dataset', default='kowikitext_20200920.train', type=str)

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
    temperature = args.temperature
    dataset = args.dataset

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    with open(f"dataset/{dataset}", encoding='utf8') as f:
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

    pretrain_model(epochs, device, train_dataloader, model, loss_fn, optimizer, None)


if __name__ == "__main__":
    main()
