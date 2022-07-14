import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import set_seed, BertTokenizer, BertConfig, BertModel

from util_fn import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForFurtherTrainByMLM(nn.Module):
    def __init__(self, model_name):
        super(BertForFurtherTrainByMLM, self).__init__()

        self.config = BertConfig.from_pretrained(model_name)
        hidden_size, vocab_size = self.config.hidden_size, self.config.vocab_size
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids, masked_arr, **kwargs):
        bert_out, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
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
        bert_out, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
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
    parser.add_argument('--model_name', default='bert-base-cased', type=str)  # Should be bert base model
    parser.add_argument('--batch_max_size', default=32, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    args = parser.parse_known_args()[0]
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    logger.info('[List of arguments]')
    for a in args.__dict__:
        logger.info(f'{a}: {args.__dict__[a]}')

    # Device --
    device = args.device

    # Hyper parameter --
    set_seed(args.seed)
    learning_rate = args.lr
    batch_max_size = args.batch_max_size
    epochs = args.epochs
    seq_max_length = args.seq_max_length
    model_name = args.model_name

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
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

    eli5 = load_dataset("eli5", split="train_asks[:50]")  # FIXME revert
    eli5 = eli5.flatten()
    eli5 = eli5.map(format_input_target)
    eli5.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'masked_arr', 'labels'])
    eli5_dataloader = DataLoader(eli5, batch_size=batch_max_size)

    model = BertForFurtherTrainByMLM(model_name)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def fn_loss(predicts, batch, batch_size):
        return criterion(predicts[batch['masked_arr']], batch['labels'][batch['masked_arr']]) / batch_size

    def fn_score(pred, label):
        return accuracy_score(label, np.argmax(pred, axis=-1))

    def cb_after_each_step(train_callback_args):
        if train_callback_args.is_end_of_epoch():
            train_score = 0
            for i in range(train_callback_args.train_num_batches):
                train_score += fn_score(train_callback_args.train_predicts[i][train_callback_args.train_batches[i]['masked_arr']],
                                        train_callback_args.train_batches[i]['labels'][train_callback_args.train_batches[i]['masked_arr']])

            train_score /= train_callback_args.train_num_batches

            logger.info(f'Epoch {train_callback_args.epoch} train loss, train score: [{train_callback_args.train_loss:.2}, {train_score:.2}]')
            train_callback_args.clear_train_score_args()

            save_model_config(model_save_path, model_name, train_callback_args.best_model.bert.state_dict(), train_callback_args.best_model.config.to_dict())

    train_model(
        epochs,
        device,
        eli5_dataloader,
        model,
        fn_loss,
        optimizer,
        cb_after_each_step=cb_after_each_step,
        param_disable_tqdm=True
    )


def use_pretrained_model_in_down_stream_task_main(model_save_path):
    model_name, model_state_dict, model_config_dict = load_model_config(model_save_path)
    model = BertForDownstreamTask(None, 1, model_state_dict, model_config_dict)

    logger.info(model_name)  # Use this to load tokenizer and parse dataset
    logger.info(model)  # Use this for downstream task

    # Need code for downstream task, but will not write here.
    # See other examples.


if __name__ == "__main__":
    model_save_path = "checkpoint/model_state.pt"
    pretrain_main(model_save_path)
    use_pretrained_model_in_down_stream_task_main(model_save_path)
