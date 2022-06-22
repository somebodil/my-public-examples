import torch
from torch import nn

size = 10
input = 3


def one_hot_encode(input, size):
    vec = torch.zeros(size).float()
    vec[input] = 1.0
    return vec


ohe = one_hot_encode(input, size)
linear_layer = nn.Linear(size, 1, bias=False)

# Set edge weights from 0 to 9 for easy reference
with torch.no_grad():
    linear_layer.weight = nn.Parameter(torch.arange(10, dtype=torch.float).reshape(linear_layer.weight.shape))

print(linear_layer.weight)
print(ohe)
print(linear_layer(ohe))



