import argparse
import copy
from datetime import datetime

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
    parser.add_argument('--model_name', default='bert-base-cased', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)  # TODO dev
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
    train_dataset = load_dataset('glue', 'stsb', split="train[:100]")  # FIXME change back to train[:80%]
    validation_dataset = load_dataset('glue', 'stsb', split="train[-100:]")  # FIXME change back to train[-20%:]
    test_dataset = load_dataset('glue', 'stsb', split="validation[:100]")  # FIXME change back to validation
    data_labels_num = 1

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def format_input(examples):
        encoded = tokenizer(examples['sentence1'], examples['sentence2'], max_length=seq_max_length, truncation=True, padding='max_length')
        return encoded

    def format_target(examples):
        changed = {'labels': examples['label']}
        return changed

    def preprocess_dataset(dataset):
        dataset = dataset.map(format_input, batched=True)
        dataset = dataset.map(format_target, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
        return dataset

    train_dataset = preprocess_dataset(train_dataset)
    validation_dataset = preprocess_dataset(validation_dataset)
    test_dataset = preprocess_dataset(test_dataset)

    model = BertForRegression(model_name, data_labels_num)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    def score_fn(pred, label):
        return stats.pearsonr(pred, label)[0]

    model = train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer, score_fn)
    test_loss, test_score = evaluate_model(device, test_dataloader, model, loss_fn, score_fn)
    print(f"\nTest (loss, score) with best model: ({test_loss:.4} / {test_score:.4})")


if __name__ == '__main__':
    main()
