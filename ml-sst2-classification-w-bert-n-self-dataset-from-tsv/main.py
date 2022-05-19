import argparse
import copy
from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, set_seed, BertTokenizer


class BertForClassification(nn.Module):
    def __init__(self, bert_model_name, hidden_size, out_features):
        super(BertForClassification, self).__init__()

        self.model = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=out_features)

    def forward(self, input_ids, attention_mask):
        _, bert_out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
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


def train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer):
    model.to(device)
    loss_fn.to(device)

    best_acc = 0
    best_model = None

    for t in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            predict = model(batch['input_ids'], batch['attention_mask'])
            loss = loss_fn(predict, batch['labels'])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_len = len(validation_dataloader)
            correct_val = 0

            for _, val_batch in enumerate(validation_dataloader):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                predict = model(val_batch['input_ids'], val_batch['attention_mask'])  # validate
                correct_val += (predict.argmax(dim=1) == val_batch['labels']).sum().item()  # cumulate correct predict

            current_acc = correct_val / val_len
            if best_acc < current_acc:
                best_acc = current_acc
                best_model = copy.deepcopy(model)

            print(f"\nValidation current_acc / best_acc : {current_acc}, {best_acc}")

    return best_model


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-cased', type=str)  # should be bert-xxx
    parser.add_argument('--hidden_size', default=768, type=int)  # hidden size of bert-base-cased
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=5, type=int)
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

    df_train = pd.read_csv('./glue_sst2_train.tsv', delimiter='\t')
    df_val = pd.read_csv('./glue_sst2_dev.tsv', delimiter='\t')
    data_label_num = 2

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = GlueSst2Dataset(df_train, tokenizer, seq_max_length)
    validation_dataset = GlueSst2Dataset(df_val, tokenizer, seq_max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = BertForClassification(model_name, hidden_size, data_label_num)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer)


if __name__ == "__main__":
    main()
