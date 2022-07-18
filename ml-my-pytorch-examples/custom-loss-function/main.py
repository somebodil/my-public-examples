import logging

import torch
from torch import nn
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def my_cross_entropy(x, y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    return loss


def main():
    cross_entropy_fn = nn.CrossEntropyLoss()

    batch_size = 5
    nb_classes = 10
    x = torch.randn(batch_size, nb_classes, requires_grad=True)
    y = torch.randint(0, nb_classes, (batch_size,))

    ref_loss = cross_entropy_fn(x, y)
    my_loss = my_cross_entropy(x, y)

    logger.info(f"ref_loss : {ref_loss}")  # ref_loss : 1.8656104803085327
    logger.info(f"my_loss : {my_loss}")  # my_loss : 1.8656104803085327


if __name__ == '__main__':
    main()
