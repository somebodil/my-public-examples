import copy

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig


class BertClassifier(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads, num_classes, bert_model_name):
        super(BertClassifier, self).__init__()

        # bert_config = BertConfig(hidden_size=hidden_size,
        #                          num_hidden_layers=num_hidden_layers,
        #                          num_attention_heads=num_attention_heads)
        # self.model = BertModel(bert_config).from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        _, x = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.linear(x)
        return x


def train(device, train_dataloader, validation_dataloader, model, loss_fn, optimizer):
    model.train()
    model.to(device)
    loss_fn.to(device)

    for i, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Compute prediction error
        predict = model(batch['input_ids'], batch['attention_mask'])
        loss = loss_fn(predict, batch['labels'])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    best_model = None

    with torch.no_grad():
        val_len = len(validation_dataloader.dataset['labels'])
        correct_val = 0
        best_acc = 0

        for i, batch in enumerate(validation_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            predict = model(batch['input_ids'], batch['attention_mask'])  # validate
            correct_val += (predict.argmax(dim=1) == batch['labels']).sum().item()  # cumulate correct predict

        current_acc = correct_val / val_len
        if best_acc < current_acc:
            best_acc = current_acc
            best_model = copy.deepcopy(model)

        print(f"Validation current_acc / best_acc : {current_acc}, {best_acc}")

    return best_model


def main():
    # Device --
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyper parameter --
    np.random.seed(4885)
    learning_rate = 1e-3
    batch_size = 16
    epochs = 5
    max_length = 128
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    model_name = "bert-base-cased"

    # Dataset --
    train_dataset = load_dataset('glue', 'sst2', split="train")
    validation_dataset = load_dataset('glue', 'sst2', split="validation")

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode(examples):
        return tokenizer(examples['sentence'], max_length=max_length, truncation=True, padding='max_length')

    train_dataset = train_dataset.map(encode, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    validation_dataset = validation_dataset.map(encode, batched=True)
    validation_dataset = validation_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    validation_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = BertClassifier(hidden_size, num_hidden_layers, num_attention_heads, np.unique(train_dataset['label']).shape[0], model_name)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        train(device, train_dataloader, validation_dataloader, model, loss_fn, optimizer)


if __name__ == '__main__':
    main()
