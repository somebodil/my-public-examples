import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments, set_seed, BertConfig, BertTokenizer, BertModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="bert-base-uncased")  # Should be bert based model


@dataclass
class DataTrainingArguments:
    max_train_samples: Optional[int] = field(default=100)


def main():
    # Parser --
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(["--output_dir", "./output_dir"])
    logger.info('[List of arguments] %s, %s, %s', model_args, data_args, training_args)

    # Device & Seed --
    device = training_args.device
    set_seed(training_args.seed)

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    dataset = load_dataset("glue", "stsb")
    dataset_keys = ("sentence1", "sentence2")
    dataset_num_labels = 2

    config = BertConfig.from_pretrained(model_args.config_name)
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    model = BertModel.from_pretrained(model_args.model_name_or_path)


if __name__ == "__main__":
    main()
