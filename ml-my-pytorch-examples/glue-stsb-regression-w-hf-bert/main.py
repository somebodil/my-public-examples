import argparse
import logging
from datetime import datetime

import torch
from datasets import load_dataset
from scipy import stats
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import set_seed

from util_fn import train_model, evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForRegression(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertForRegression, self).__init__()

        self.hidden_size = BertConfig.from_pretrained(bert_model_name).hidden_size
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        _, bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        linear_out = self.linear(bert_out)
        sigmoid_out = self.sigmoid(linear_out) * 5
        return sigmoid_out.squeeze()


def main():
    # Parser --
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    parser.add_argument('--model_name', default='bert-base-cased', type=str)  # Should be bert base model
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--batch_max_size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)

    args = parser.parse_known_args()[0]
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    logger.info('[List of arguments]')
    for a in args.__dict__:
        logger.info(f'{a}: {args.__dict__[a]}')

    # Device & Seed --
    device = args.device
    set_seed(args.seed)

    # Hyper parameter --
    model_name = args.model_name
    seq_max_length = args.seq_max_length
    batch_max_size = args.batch_max_size
    epochs = args.epochs
    learning_rate = args.lr

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
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

    train_dataset = load_dataset('glue', 'stsb', split="train[:100]")  # FIXME change back to train[:80%]
    validation_dataset = load_dataset('glue', 'stsb', split="train[-100:]")  # FIXME change back to train[-20%:]
    test_dataset = load_dataset('glue', 'stsb', split="validation[:100]")  # FIXME change back to validation
    data_labels_num = 1

    train_dataset = preprocess_dataset(train_dataset)
    validation_dataset = preprocess_dataset(validation_dataset)
    test_dataset = preprocess_dataset(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_max_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_max_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_max_size)

    model = BertForRegression(model_name, data_labels_num)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    def fn_loss(predicts, batch, batch_size):
        return criterion(predicts, batch['labels'])

    def fn_score(pred, label):
        return stats.pearsonr(pred, label)[0]

    def cb_after_each_step(train_callback_args):
        if train_callback_args.is_end_of_epoch():
            train_score = 0
            for i in range(train_callback_args.train_num_batches):
                train_score += fn_score(train_callback_args.train_predicts[i], train_callback_args.train_batches[i]['labels'])

            train_score /= train_callback_args.train_num_batches

            val_loss, val_score = evaluate_model(device, validation_dataloader, train_callback_args.model, fn_loss, fn_score, param_disable_tqdm=True)
            if train_callback_args.is_greater_than_best_val_score(val_score):
                train_callback_args.set_best_val_args(val_loss, val_score)

            logger.info(f'Epoch {train_callback_args.epoch} train loss, train score, val loss, val score: [{train_callback_args.train_loss:.2}, {train_score:.2}, {val_loss:.2}, {val_score:.2}]')
            train_callback_args.clear_train_score_args()

    model, best_val_epoch, best_val_loss, best_val_score = train_model(
        epochs,
        device,
        train_dataloader,
        model,
        fn_loss,
        optimizer,
        cb_after_each_step=cb_after_each_step,
        param_disable_tqdm=True
    )

    test_loss, test_score = evaluate_model(device, test_dataloader, model, fn_loss, fn_score, param_disable_tqdm=True)
    logger.info(f"Test (loss, score) with best val model (epoch, loss, score) : ({test_loss:.2} / {test_score:.2}), ({best_val_epoch}, {best_val_loss:.2} / {best_val_score:.2})")


if __name__ == '__main__':
    main()
