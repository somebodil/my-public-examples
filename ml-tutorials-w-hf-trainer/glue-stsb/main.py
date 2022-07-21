import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."})



def main():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
