import argparse
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import set_seed, AutoTokenizer


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2', type=str)  # should be gpt2-xxx
    parser.add_argument('--hidden_size', default='768', type=int)  # should be determined by model_name
    parser.add_argument('--batch_size', default=4, type=int)  # FIXME dev
    parser.add_argument('--seq_max_length', default=128, type=int)  # FIXME check data if 128 is enough
    parser.add_argument('--epochs', default=1, type=int)  # FIXME dev
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    args = parser.parse_args()
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
    hidden_size = args.hidden_size

    # Dataset --
    train_dataset = load_dataset('klue', 'sts', split="train")
    validation_dataset = load_dataset('klue', 'sts', split="validation")
    data_label_num = 2

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # FIXME Extract to other file
    tokenizer.pad_token = tokenizer.eos_token  # FIXME Extract to other file

    def encode(examples):
        return tokenizer(examples['sentence'], max_length=seq_max_length, truncation=True, padding='max_length')

    train_dataset = train_dataset.map(encode, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    ab = 0

if __name__ == '__main__':
    main()
