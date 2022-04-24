import argparse
import copy
import datetime

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig, set_seed


class BertForClassification(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_classes):
        super(BertForClassification, self).__init__()

        config = BertConfig(hidden_size=hidden_size)
        self.model = BertModel(config).from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        _, x = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.linear(x)
        return x


def train(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer):
    best_acc = 0
    best_model = None

    for t in range(epochs):
        model.to(device)
        loss_fn.to(device)

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
            val_len = len(validation_dataloader.dataset['labels'])
            correct_val = 0

            for _, val_batch in enumerate(validation_dataloader):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                predict = model(val_batch['input_ids'], val_batch['attention_mask'])  # validate
                correct_val += (predict.argmax(dim=1) == val_batch['labels']).sum().item()  # cumulate correct predict

            current_acc = correct_val / val_len
            if best_acc < current_acc:
                best_acc = current_acc
                best_model = copy.deepcopy(model)

            print(f"Validation current_acc / best_acc : {current_acc}, {best_acc}")

    return best_model


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-cased', type=str)  # should be bert-xxx
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))

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
    hidden_size = args.hidden_size
    model_name = args.model_name

    # Dataset --
    train_dataset = load_dataset('glue', 'sst2', split="train")
    validation_dataset = load_dataset('glue', 'sst2', split="validation")

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode(examples):
        return tokenizer(examples['sentence'], max_length=seq_max_length, truncation=True, padding='max_length')

    train_dataset = train_dataset.map(encode, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    validation_dataset = validation_dataset.map(encode, batched=True)
    validation_dataset = validation_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    validation_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = BertForClassification(model_name, hidden_size, np.unique(train_dataset['label']).shape[0])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer)


if __name__ == '__main__':
    main()
