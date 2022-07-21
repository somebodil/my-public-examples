import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments, set_seed

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataTrainingArguments:
    max_train_samples: Optional[int] = field(
        default=100,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."}
    )


def main():
    # Parser --
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(["--output_dir", "./output_dir"])
    logger.info('[List of arguments] %s, %s, %s', model_args, data_args, training_args)

    # Device & Seed --
    device = training_args.device
    set_seed(training_args.seed)

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    train_dataset = load_dataset("glue", "stsb")
    dataset_num_labels = 2


if __name__ == "__main__":
    main()
