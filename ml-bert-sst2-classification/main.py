import numpy as np
from datasets import load_dataset
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig


class BertClassifier(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads, num_classes, is_cased):
        super(BertClassifier, self).__init__()

        bert_model_name = 'bert-base-cased' if is_cased else 'bert-base-uncased'
        bert_config = BertConfig(hidden_size=hidden_size,
                                 num_hidden_layers=num_hidden_layers,
                                 num_attention_heads=num_attention_heads)
        self.model = BertModel(bert_config).from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        _, x = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.linear(x)
        return x


def train(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train().to(device)
    loss_fn.to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Compute prediction error
        predict = model(batch['input_ids'], batch['attention_mask'])
        loss = loss_fn(predict, batch['labels'])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            loss, current = loss.item(), i * dataloader.batch_size
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def encode(examples, tokenizer, max_length):
    encoded = tokenizer(examples['sentence'], max_length=max_length, truncation=True, padding='max_length')
    return encoded


def main():
    # Device --
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # Hyper parameter --
    np.random.seed(4885)
    learning_rate = 1e-3
    batch_size = 16
    epochs = 5
    max_length = 512
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12

    # Dataset --
    train_dataset = load_dataset('glue', 'sst2', split="train")

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = train_dataset.map(lambda examples: encode(examples, tokenizer, max_length), batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    model = BertClassifier(hidden_size, num_hidden_layers, num_attention_heads, np.unique(train_dataset['label']).shape[0], True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        train(device, train_dataloader, model, loss_fn, optimizer)


if __name__ == '__main__':
    main()
