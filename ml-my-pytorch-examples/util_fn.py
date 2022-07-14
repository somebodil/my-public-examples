import copy

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class TrainCallbackArgs:
    def __init__(self, model, step_of_epoch, num_of_epoch):
        self.model = model
        self.step_of_epoch = step_of_epoch
        self.num_of_epoch = num_of_epoch

        self.step = 0
        self.epoch = 0

        self.train_loss = 0.0
        self.train_num_batches = 0
        self.train_predicts = []
        self.train_batches = []
        self.train_batch_sizes = []

        self.best_val_epoch = 0
        self.best_val_loss = 0.0
        self.best_val_score = 0.0
        self.best_model = model

    def set_train_score_args(self, step, epoch, loss, predicts, input_batch, input_batch_sizes):
        self.step = step
        self.epoch = epoch

        self.train_loss += loss.item()

        self.train_num_batches += 1
        self.train_predicts.append(predicts.clone().detach().cpu())
        self.train_batches.append({k: v.clone().detach().cpu() for k, v in input_batch.items()})
        self.train_batch_sizes.append(input_batch_sizes)

    def set_best_val_args(self, loss, score):
        self.best_val_loss = loss
        self.best_val_score = score
        self.best_val_epoch = self.epoch
        self.best_model = copy.deepcopy(self.model).cpu()

    def clear_train_score_args(self):
        self.train_loss = 0.0
        self.train_num_batches = 0
        self.train_predicts = []
        self.train_batches = []
        self.train_batch_sizes = []

    def get_cumulated_step(self):
        return self.step + self.step_of_epoch * (self.epoch - 1)

    def is_end_of_epoch(self):
        return self.step == self.step_of_epoch

    def is_start_of_train(self):
        return self.step == 1 and self.epoch == 1

    def is_end_of_train(self):
        return self.epoch == self.num_of_epoch and self.step == self.step_of_epoch

    def is_greater_than_best_val_score(self, score):
        return self.best_val_score < score

    def is_step_interval(self, interval):
        return self.get_cumulated_step() % interval == 0


def train_model(
        epochs,
        device,
        dataloader,
        model,
        fn_loss,
        optimizer,
        cb_after_each_step=None,
        param_disable_tqdm=False):
    """
    Callback function cb_after_each_step is always called after every each step.
    Batch size is calculated using first column of input batch.
    Developer should not forget to call clear_train_score_args manually, or memory will explode.
    """

    model.to(device)
    train_callback_args = TrainCallbackArgs(model, len(dataloader), epochs)

    for epoch in range(1, epochs + 1):
        model.train()

        progress_bar = tqdm(dataloader, disable=param_disable_tqdm)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(progress_bar, 1):
            with logging_redirect_tqdm():
                batch_size = len(batch[next(iter(batch))])
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                predict = model(**batch)
                loss = fn_loss(predict, batch, batch_size)
                loss.backward()
                optimizer.step()

                if cb_after_each_step:
                    train_callback_args.set_train_score_args(step, epoch, loss, predict, batch, batch_size)
                    cb_after_each_step(train_callback_args)

    return train_callback_args.best_model, train_callback_args.best_val_epoch, train_callback_args.best_val_loss, train_callback_args.best_val_score


def evaluate_model(device, dataloader, model, fn_loss, fn_score, param_disable_tqdm=False):
    """
    Function always assumes input batch 'labels' column.
    Batch size is calculated using first column of input batch.
    """

    model.to(device)

    eval_loss = 0
    eval_pred = []
    eval_label = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate", disable=param_disable_tqdm):
            with logging_redirect_tqdm():
                batch_size = len(batch[next(iter(batch))])
                batch = {k: v.to(device) for k, v in batch.items()}

                predict = model(**batch)
                loss = fn_loss(predict, batch, batch_size)

                eval_loss += loss.item()
                eval_pred.extend(predict.tolist())
                eval_label.extend(batch['labels'].tolist())

    eval_score = fn_score(eval_pred, eval_label)
    return eval_loss, eval_score
