import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertConfig


class GlueSst2Dataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        self.labels = [label for label in data['label']]

        self.sentences = [
            tokenizer(sentence, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
            for sentence in data['sentence']
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class BertClassifier(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads, max_length, num_classes, is_cased):
        super(BertClassifier, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bert_model_name = 'bert-base-cased' if is_cased else 'bert-base-uncased'
        bert_config = BertConfig(hidden_size=hidden_size,
                                 num_hidden_layers=num_hidden_layers,
                                 num_attention_heads=num_attention_heads)
        self.model = BertModel(bert_config).from_pretrained(bert_model_name).to(device)
        self.linear = nn.Linear(hidden_size * max_length, num_classes).to(device)

    def forward(self, input_ids, attention_mask):
        model_out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        return None


def train(dataloader, model, loss_fn, optimizer):
    model.train()


def main():
    # Hyper parameter --
    np.random.seed(4885)
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    max_length = 512
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12

    # Dataset --
    training_data = pd.read_csv('./data/glue_data/SST-2/train.tsv', delimiter='\t')
    test_data = pd.read_csv('./data/glue_data/SST-2/test.tsv', delimiter='\t')

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataloader = DataLoader(GlueSst2Dataset(tokenizer, training_data, max_length), batch_size=batch_size)
    test_dataloader = DataLoader(GlueSst2Dataset(tokenizer, test_data, max_length), batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer, device)


if __name__ == '__main__':
    main()
