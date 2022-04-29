import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, BatchEncoding


class GlueSst2Dataset(Dataset):

    def __init__(self, data_frame, tokenizer, max_length):
        self.len = len(data_frame['label'])
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {"labels": self.data_frame['label'][idx], **self.tokenizer(self.data_frame['sentence'][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv('./glue_sst2_train.tsv', delimiter='\t')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = GlueSst2Dataset(df_train, tokenizer, 256)  # Glue sst2 train data max token length is 64, so 256 is enough
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    for i, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        print(f"input_ids : {input_ids}\nattention_mask : {attention_mask}\nlabels : {labels}")


if __name__ == "__main__":
    main()
