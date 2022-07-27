import argparse
import logging
from datetime import datetime

import pandas as pd
import torch
from datasets import load_dataset
from scipy import stats
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import set_seed

from util import train_model, evaluate_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BertForRegression(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertForRegression, self).__init__()

        self.hidden_size = BertConfig.from_pretrained(bert_model_name).hidden_size
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        _, pooler_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        linear_out = self.linear(pooler_out)
        return linear_out.squeeze(1)


def main():
    # Parser --
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    parser.add_argument('--model_name', default='bert-base-cased', type=str)  # Should be bert base model
    parser.add_argument('--batch_max_size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

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

    train_dataset = load_dataset('glue', 'stsb', split="train[:100]")  # FIXME change back to train[:80%]
    validation_dataset = load_dataset('glue', 'stsb', split="train[-100:]")  # FIXME change back to train[-20%:]
    test_dataset = load_dataset('glue', 'stsb', split="validation[:100]")  # FIXME change back to validation
    data_labels_num = 1

    def collate_fn(batch):
        batch = pd.DataFrame(batch)
        tokenized = tokenizer(
            batch['sentence1'].tolist(),
            batch['sentence2'].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            **tokenized,
            'labels': torch.tensor(batch['label'], dtype=torch.float32)
        }

    train_dataloader = DataLoader(train_dataset, batch_size=batch_max_size, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_max_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_max_size, collate_fn=collate_fn)

    model = BertForRegression(model_name, data_labels_num)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    def loss_fn(predicts, batch, batch_size):
        return criterion(predicts, batch['labels'])

    def score_fn(predicts, labels):
        return stats.pearsonr(predicts, labels)[0]

    def after_each_step_fn(train_callback_args):
        if train_callback_args.is_end_of_epoch():
            train_epoch, _ = train_callback_args.get_epoch_step()
            train_loss, train_num_batches, train_predicts, train_batches, train_batch_sizes = train_callback_args.get_train_score_args()

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

    model, best_val_epoch, best_val_loss, best_val_score = train_model(
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


if __name__ == '__main__':
    main()
