import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, set_seed, BertTokenizer, BertConfig

from util import train_model, evaluate_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BertForClassification(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertForClassification, self).__init__()

        self.hidden_size = BertConfig.from_pretrained(bert_model_name).hidden_size
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=self.hidden_size,
                                out_features=num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        _, bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        linear_out = self.linear(bert_out)
        return linear_out


class GlueSst2Dataset(Dataset):

    def __init__(self, data_frame):
        self.len = len(data_frame)
        self.data_frame = data_frame

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            'sentence': self.data_frame['sentence'][idx],
            'label': self.data_frame['label'][idx],
        }


def main():
    # Parser --
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    parser.add_argument('--model_name', default='bert-base-cased', type=str)  # should be bert-xxx
    parser.add_argument('--batch_max_size', default=32, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)

    args = parser.parse_known_args()[0]
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    logger.debug('[List of arguments]')
    for a in args.__dict__:
        logger.debug(f'{a}: {args.__dict__[a]}')

    # Device & Seed --
    device = args.device
    set_seed(args.seed)

    # Hyper parameter --
    model_name = args.model_name
    batch_max_size = args.batch_max_size
    epochs = args.epochs
    learning_rate = args.lr

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(model_name)

    df_train = pd.read_csv('glue_sst2_train.tsv', delimiter='\t')
    df_train = df_train[:50]  # FIXME remove
    df_train, df_val = np.split(df_train, [int(.8 * len(df_train))])
    df_val = df_val.reset_index(drop=True)

    df_test = pd.read_csv('glue_sst2_dev.tsv', delimiter='\t')
    dataset_num_labels = 2

    train_dataset = GlueSst2Dataset(df_train)
    validation_dataset = GlueSst2Dataset(df_val)
    test_dataset = GlueSst2Dataset(df_test)

    def collate_fn(batch):
        batch = pd.DataFrame(batch)
        tokenized = tokenizer(batch['sentence'].tolist(), padding=True, truncation=True, return_tensors="pt")

        return {
            **tokenized,
            'labels': torch.tensor(batch['label'])
        }

    train_dataloader = DataLoader(train_dataset, batch_size=batch_max_size, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_max_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_max_size, collate_fn=collate_fn)

    model = BertForClassification(model_name, dataset_num_labels)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def loss_fn(predicts, batch, batch_size):
        return criterion(predicts, batch['labels'])

    def score_fn(predicts, labels):
        return accuracy_score(labels, np.argmax(predicts, axis=1))

    def after_each_step_fn(train_callback_args):
        if train_callback_args.is_end_of_epoch():
            train_epoch, _ = train_callback_args.get_epoch_step()
            train_loss, train_num_batches, train_predicts, train_batches, train_batch_sizes = train_callback_args.get_n_clear_train_args()

            train_score = 0
            for i in range(train_num_batches):
                train_score += score_fn(train_predicts[i], train_batches[i]['labels'])
            train_score /= train_num_batches

            val_loss, val_score = evaluate_model(
                device,
                validation_dataloader,
                train_callback_args.model,
                score_fn,
                loss_fn=loss_fn,
                disable_tqdm=True
            )

            if train_callback_args.is_greater_than_best_val_score(val_score):
                train_callback_args.set_best_val_args(val_score, val_loss)

            logger.debug(
                f'Epoch {train_epoch} train loss, train score, val loss, val score: '
                f'[{train_loss:.2}, {train_score:.2}, {val_loss:.2}, {val_score:.2}]'
            )

    model, best_val_epoch, best_val_acc_step, best_val_loss, best_val_score = train_model(
        epochs,
        device,
        train_dataloader,
        model,
        loss_fn,
        optimizer,
        after_each_step_fn=after_each_step_fn,
        disable_tqdm=True
    )

    test_loss, test_score = evaluate_model(
        device,
        test_dataloader,
        model,
        score_fn,
        loss_fn=loss_fn,
        disable_tqdm=True
    )

    logger.debug(
        f"Test (loss, score) with best val model (epoch, loss, score) : "
        f"({test_loss:.2} / {test_score:.2}), ({best_val_epoch}, {best_val_loss:.2} / {best_val_score:.2})"
    )


if __name__ == "__main__":
    main()
