import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


class GlueSst2Dataset(Dataset):

    def __init__(self, data_frame, tokenizer, max_length):
        self.labels = [label for label in data_frame['label']]
        self.inputs = [tokenizer(sentence, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt") for sentence in data_frame['sentence']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv('./glue_sst2_train.tsv', delimiter='\t')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = GlueSst2Dataset(df_train, tokenizer, 256)  # Glue sst2 train data max token length is 64, so 256 is enough
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    for train_input, train_label in enumerate(train_dataloader):
        input_ids = train_input['input_ids'].to(device)
        attention_mask = train_input['attention_mask'].to(device)
        labels = train_label.to(device)

        print(f"input_ids : {input_ids}\nattention_mask : {attention_mask}\nlabels : {labels}")


if __name__ == "__main__":
    main()
