import argparse
import logging
import os
from datetime import datetime

import pandas as pd
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed, BertConfig, BertModel, BertTokenizer

from util import train_model, evaluate_model

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)


def encode(cls, input_ids, attention_mask, token_type_ids, **kwargs):
    last_hidden_state, _ = cls.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=False
    )

    return (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)  # avg


class BertForValidationOnStsb(nn.Module):
    def __init__(self, state_dict, config_dict):
        super(BertForValidationOnStsb, self).__init__()

        config = BertConfig.from_dict(config_dict)
        self.config = config
        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=None,
            torch_dtype=self.config.torch_dtype,
            config=self.config,
            state_dict=state_dict
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(
            self,
            input_ids_1,
            attention_mask_1,
            token_type_ids_1,
            input_ids_2,
            attention_mask_2,
            token_type_ids_2,
            **kwargs
    ):
        embedding1 = encode(self, input_ids_1, attention_mask_1, token_type_ids_1)
        embedding2 = encode(self, input_ids_2, attention_mask_2, token_type_ids_2)
        cos_sim_out = self.cosine_similarity(embedding1, embedding2)
        return cos_sim_out


class BertForFurtherTrain(nn.Module):
    def __init__(self, model_name, temperature):
        super(BertForFurtherTrain, self).__init__()

        config = BertConfig.from_pretrained(model_name)  # uses default dropout rate = 0.1
        self.config = config
        self.bert = BertModel.from_pretrained(model_name, config=self.config, torch_dtype=self.config.torch_dtype)
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        batch_size = input_ids.shape[0]

        # copy (batch_size, seq_len) => (batch_size, 2, seq_len)
        input_ids = input_ids.unsqueeze(1).repeat(1, 2, 1)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, 2, 1)
        token_type_ids = token_type_ids.unsqueeze(1).repeat(1, 2, 1)

        # flat => (batch_size * 2, seq_len)
        input_ids = input_ids.view((-1, input_ids.shape[-1]))
        attention_mask = attention_mask.view((-1, input_ids.shape[-1]))
        token_type_ids = token_type_ids.view((-1, input_ids.shape[-1]))

        # encode => (batch_size * 2, hidden_size)
        pooler_out = encode(self, input_ids, attention_mask, token_type_ids)

        # revert flat => (batch_size, 2, hidden_size)
        pooler_out = pooler_out.view((batch_size, 2, pooler_out.shape[-1]))

        # cos sim
        # (batch_size, hidden_size)
        z1, z2 = pooler_out[:, 0], pooler_out[:, 1]

        # => (batch_size, 1, hidden_size), (batch_size, hidden_size, 1)
        # => (batch_size, batch_size)
        cos_sim_out = self.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature
        return cos_sim_out


class UnsupervisedSimCseDataset(Dataset):
    def __init__(self, data_frame):
        self.len = len(data_frame)
        self.data_frame = data_frame

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            'sentence': self.data_frame['sentence'][idx]
        }


def save_model_config(path, model_name, model_state_dict, model_config_dict):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    torch.save({
        'model_name': model_name,
        'model_state_dict': model_state_dict,
        'model_config_dict': model_config_dict
    }, path)


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    parser.add_argument('--model_name', default='bert-base-cased', type=str)
    parser.add_argument('--batch_max_size', default=16, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)

    parser.add_argument('--model_state_name', default='unsup-bert.pt', type=str)

    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--pretrain_dataset', default='wiki1m_for_simcse.txt', type=str)

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

    model_state_name = args.model_state_name

    temperature = args.temperature
    pretrain_dataset = args.pretrain_dataset

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    model = BertForFurtherTrain(model_name, temperature)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    tokenizer = BertTokenizer.from_pretrained(model_name)

    with open(f"dataset/{pretrain_dataset}", encoding='utf8') as f:
        lines = []
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

        df_train = pd.DataFrame(lines, columns=['sentence'])

    train_dataset = UnsupervisedSimCseDataset(df_train)
    validation_dataset = load_dataset('glue', 'stsb', split="validation")

    def train_dataloader_collate_fn(batch):
        batch = pd.DataFrame(batch)
        tokenized = tokenizer(
            batch['sentence'].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return tokenized

    train_dataloader = DataLoader(train_dataset, batch_size=batch_max_size, collate_fn=train_dataloader_collate_fn)

    def validation_dataloader_collate_fn(batch):  # if is_glue true then glue, else klue
        batch = pd.DataFrame(batch)
        tokenized_sentence1 = tokenizer(
            batch['sentence1'].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokenized_sentence2 = tokenizer(
            batch['sentence2'].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids_1': tokenized_sentence1['input_ids'],
            'attention_mask_1': tokenized_sentence1['attention_mask'],
            'token_type_ids_1': tokenized_sentence1['token_type_ids'],
            'input_ids_2': tokenized_sentence2['input_ids'],
            'attention_mask_2': tokenized_sentence2['attention_mask'],
            'token_type_ids_2': tokenized_sentence2['token_type_ids'],
            'labels': torch.tensor(batch['label'])
        }

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_max_size,
        collate_fn=validation_dataloader_collate_fn
    )

    def loss_fn(predicts, batch, batch_size):
        labels = torch.arange(batch_size).long().to(device)  # ex) [1,2,3,4,5, ...]
        return criterion(predicts, labels)

    def score_fn(predicts, labels):
        try:
            score = spearmanr(predicts, labels, nan_policy="raise")[0]
        except ValueError:
            logger.debug(f"predicts : {predicts}")
            logger.debug(f"labels : {labels}")
            raise ValueError("predicts and labels should be same length")

        return score

    def after_each_step_fn(train_callback_args):
        if train_callback_args.is_step_interval(250) or train_callback_args.is_end_of_train():
            _, acc_step = train_callback_args.get_epoch_step()
            train_callback_args.get_n_clear_train_args()

            bert = train_callback_args.model.bert
            config = train_callback_args.model.config
            model = BertForValidationOnStsb(bert.state_dict(), config.to_dict())

            _, val_score = evaluate_model(
                device,
                validation_dataloader,
                BertForValidationOnStsb(model.bert.state_dict(), model.config.to_dict()),
                score_fn,
                disable_tqdm=True
            )

            if train_callback_args.is_greater_than_best_val_score(val_score):
                train_callback_args.set_best_val_args(val_score)

            logger.debug(f'Step {acc_step} val score: [{val_score:.2}]')

    _, val_score = evaluate_model(
        device,
        validation_dataloader,
        BertForValidationOnStsb(model.bert.state_dict(), model.config.to_dict()),
        score_fn,
        disable_tqdm=True
    )

    logger.debug(f'Before training, glue val score: [{val_score:.2}]')

    model, _, best_val_acc_step, best_val_loss, best_val_score = train_model(
        epochs,
        device,
        train_dataloader,
        model,
        loss_fn,
        optimizer,
        after_each_step_fn=after_each_step_fn
    )

    save_model_config(f'checkpoint/{model_state_name}', model_name, model.bert.state_dict(), model.config.to_dict())
    logger.debug(f"Best (glue + klue) acc_step && score : ({best_val_acc_step}, {best_val_score:.2})")


if __name__ == "__main__":
    main()
