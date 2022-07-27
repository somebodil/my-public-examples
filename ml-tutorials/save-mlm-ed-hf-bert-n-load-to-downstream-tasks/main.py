import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import set_seed, BertTokenizer, BertConfig, BertModel

from util import train_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BertForFurtherTrainByMLM(nn.Module):
    def __init__(self, model_name):
        super(BertForFurtherTrainByMLM, self).__init__()

        self.config = BertConfig.from_pretrained(model_name)
        hidden_size, vocab_size = self.config.hidden_size, self.config.vocab_size
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids, masked_arr, **kwargs):
        bert_out, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        linear_out = self.linear(bert_out)
        return linear_out


class BertForDownstreamTask(nn.Module):
    def __init__(self, model_name, num_labels, state_dict=None, config_dict=None):
        super(BertForDownstreamTask, self).__init__()

        if model_name is None:
            if state_dict is None or config_dict is None:
                raise ValueError("If model_name is None, state_dict and config must be specified.")

            self.config = BertConfig.from_dict(config_dict)
            self.bert = BertModel(config=self.config)
            self.bert.load_state_dict(state_dict)

        else:  # This 'else' block of code is not used in this project
            if state_dict is not None or config_dict is not None:
                raise ValueError("If model_name is not None, state_dict and config must be None.")

            self.config = BertConfig.from_pretrained(model_name)
            self.bert = BertModel.from_pretrained(model_name)

        self.linear = nn.Linear(in_features=self.config.hidden_size, out_features=num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        bert_out, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        linear_out = self.linear(bert_out)
        return linear_out


def load_model_config(path):
    config = torch.load(path, map_location='cpu')
    return config['model_name'], config['model_state_dict'], config['model_config_dict']


def save_model_config(path, model_name, model_state_dict, model_config_dict):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    torch.save({
        'model_name': model_name,
        'model_state_dict': model_state_dict,
        'model_config_dict': model_config_dict
    }, path)


def pretrain_main(model_save_path):
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    parser.add_argument('--model_name', default='bert-base-cased', type=str)  # Should be bert base model
    parser.add_argument('--batch_max_size', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int)
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

    eli5 = load_dataset("eli5", split="train_asks[:50]")  # FIXME revert
    eli5 = eli5.flatten()

    def mask_random_token(element):
        if element == tokenizer.cls_token_id or element == tokenizer.sep_token_id or element == tokenizer.pad_token_id:
            return element
        elif random.uniform(0, 1) <= 0.15:
            return torch.tensor(tokenizer.mask_token_id)

        return element

    def collate_fn(batch):
        batch = pd.DataFrame(batch)

        answer_texts = [' '.join(sentence_list) for sentence_list in batch["answers.text"].tolist()]
        encodings = tokenizer(answer_texts, padding=True, truncation=True, return_tensors='pt')
        encodings["labels"] = encodings["input_ids"].clone()

        for i, input_ids in enumerate(encodings['input_ids']):
            encodings["input_ids"][i] = torch.tensor(list(map(mask_random_token, input_ids)))

        encodings["masked_arr"] = encodings["input_ids"] != encodings['labels']
        return encodings

    eli5_dataloader = DataLoader(eli5, batch_size=batch_max_size, collate_fn=collate_fn)

    model = BertForFurtherTrainByMLM(model_name)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def loss_fn(predicts, batch, batch_size):
        return criterion(predicts[batch['masked_arr']], batch['labels'][batch['masked_arr']]) / batch_size

    def score_fn(predicts, labels):
        return accuracy_score(labels, np.argmax(predicts, axis=-1))

    def after_each_step_fn(train_callback_args):
        if train_callback_args.is_end_of_epoch():
            train_epoch, _ = train_callback_args.get_epoch_step()
            train_loss, train_num_batches, train_predicts, train_batches, train_batch_sizes = train_callback_args.get_train_score_args()

            train_score = 0
            for i in range(train_num_batches):
                train_score += score_fn(
                    train_predicts[i][train_batches[i]['masked_arr']],
                    train_batches[i]['labels'][train_batches[i]['masked_arr']]
                )
            train_score /= train_num_batches

            logger.debug(f'Epoch {train_epoch} train loss, train score: [{train_loss:.2}, {train_score:.2}]')

            save_model_config(
                model_save_path,
                model_name,
                train_callback_args.best_model.bert.state_dict(),
                train_callback_args.best_model.config.to_dict()
            )

    train_model(
        epochs,
        device,
        eli5_dataloader,
        model,
        loss_fn,
        optimizer,
        after_each_step_fn=after_each_step_fn,
        disable_tqdm=False
    )


def use_pretrained_model_in_down_stream_task_main(model_save_path):
    model_name, model_state_dict, model_config_dict = load_model_config(model_save_path)
    model = BertForDownstreamTask(None, 1, model_state_dict, model_config_dict)

    logger.debug(model_name)  # Use this to load tokenizer and parse dataset
    logger.debug(model)  # Use this for downstream task

    # Need code for downstream task, but will not write here.
    # See other examples.


if __name__ == "__main__":
    model_save_path = "checkpoint/model_state.pt"
    pretrain_main(model_save_path)
    use_pretrained_model_in_down_stream_task_main(model_save_path)
