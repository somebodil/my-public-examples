import argparse
import random
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, BertTokenizer, BertConfig, BertModel, BertForMaskedLM


class BertForFurtherTrainByMLM(nn.Module):
    def __init__(self, model_name):
        super(BertForFurtherTrainByMLM, self).__init__()

        config = BertConfig.from_pretrained(model_name)
        hidden_size, vocab_size = config.hidden_size, config.vocab_size
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        bert_out, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        linear_out = self.linear(bert_out)
        return linear_out


def further_train_model(epochs, device, dataloader, model, loss_fn, optimizer, score_fn):
    model.to(device)
    loss_fn.to(device)

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0
        train_sentence = []
        train_pred = []
        train_label = []

        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # predict = model(**batch)
            input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']  # For Bert-MLM
            predict = model(input_ids, attention_mask, token_type_ids).logits

            loss = torch.tensor(0, dtype=torch.float32).to(device)
            for j, masked_arr in enumerate(batch['masked_arr']):
                masked_arr_indices = masked_arr.nonzero(as_tuple=True)[0]
                predicted_token_id = predict[j, masked_arr_indices]
                loss += loss_fn(predicted_token_id, batch['labels'][j, masked_arr_indices])

                train_loss += loss.clone().cpu().item()
                train_sentence.extend(batch['input_ids'])
                train_pred.extend(predicted_token_id.clone().cpu().tolist())
                train_label.extend(batch['labels'][j, masked_arr_indices].clone().cpu().tolist())

            loss /= dataloader.batch_size
            loss.backward()
            optimizer.step()

        train_score = score_fn(train_pred, train_label)
        print(f'\nEpoch {epoch} (train loss, train score): ({train_loss:.4}, {train_score:.4})')


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-cased', type=str)  # Should be bert base model
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)  # TODO dev
    parser.add_argument('--lr', default=1e-6, type=float)
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
    eli5 = load_dataset("eli5", split="train_asks[:100]")
    eli5 = eli5.flatten()

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def mask_random_token(element):
        if element == tokenizer.cls_token_id or element == tokenizer.sep_token_id or element == tokenizer.pad_token_id:
            return element
        elif random.uniform(0, 1) <= 0.15:
            return torch.tensor(tokenizer.mask_token_id)

        return element

    def format_input_target(example):
        long_text = " ".join(example["answers.text"])
        encodings = tokenizer(long_text, max_length=seq_max_length, truncation=True, padding='max_length', return_tensors='pt')
        encodings["labels"] = encodings["input_ids"].clone()

        for k in encodings.keys():
            encodings[k] = encodings[k].squeeze()

        encodings["input_ids"] = torch.tensor(list(map(mask_random_token, encodings['input_ids'])))
        encodings["masked_arr"] = encodings["input_ids"] != encodings['labels']
        return encodings

    eli5 = eli5.map(format_input_target)
    eli5.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'masked_arr', 'labels'])
    # model = BertForFurtherTrainByMLM(model_name)
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    eli5_dataloader = DataLoader(eli5, batch_size=batch_size)

    def score_fn(pred, label):
        return accuracy_score(np.argmax(pred, axis=-1), np.array(label))

    further_train_model(epochs, device, eli5_dataloader, model, loss_fn, optimizer, score_fn)


if __name__ == "__main__":
    main()
