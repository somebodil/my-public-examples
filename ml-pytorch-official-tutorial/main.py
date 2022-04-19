import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()

        # Define the layers of the network
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        # Define how to forward the network
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), i * len(X)
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    total_test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            total_test_loss += loss_fn(y_hat, y).item()
            total_correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    print(f"Test Error: "
          f"Accuracy: {(100 * (total_correct / size)):>0.1f}%, "
          f"Avg loss: {(total_test_loss / num_batches):>8f} \n")


def main():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyper-parameters for this tutorial
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    # [loading data in PyTorch]
    # https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#working-with-data
    #
    # PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets.
    # The torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR, COCO.
    # Every TorchVision Dataset includes two arguments: transform and target_transform to modify the samples and labels respectively.
    #
    # labels of the data are as below (See https://github.com/zalandoresearch/fashion-mnist for details) :
    #  - 0: "T-Shirt",
    #  - 1: "Trouser",
    #  - 2: "Pullover",
    #  - 3: "Dress",
    #  - 4: "Coat",
    #  - 5: "Sandal",
    #  - 6: "Shirt",
    #  - 7: "Sneaker",
    #  - 8: "Bag",
    #  - 9: "Ankle Boot",

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # We pass the Dataset as an argument to DataLoader.
    # This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading.
    # Here we define a batch size of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels.
    #
    # Input size should be 28 * 28 because each img size is 28 * 28.
    # Output size should be 10 because target label is 10.

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    input_size = 28 * 28
    output_size = 10

    for X, y in test_dataloader:
        # Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28]), 64 (batch size) element, each element size is 1 * 28 * 28
        print("Shape of X [N, C, H, W]: ", X.shape)
        # Shape of y:  torch.Size([64]) torch.int64, 64 (batch size) truth label
        print("Shape of y: ", y.shape, y.dtype)
        break

    # [Creating Models]
    # https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#creating-models
    #
    # To define a neural network in PyTorch, we create a class that inherits from nn.Module.
    # We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function.

    model = MyModel(input_size, output_size).to(device)

    # [Optimizing the Model Parameters]
    # https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#working-with-data
    #
    # To train a model, we need a loss function and an optimizer.
    #
    # In a single training loop the model :
    #   1. makes predictions on the training dataset (fed to it in batches)
    #   2. backpropagates the prediction error to adjust the model’s parameters.

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # [Train and Test]

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(device, train_dataloader, model, loss_fn, optimizer)
        test(device, test_dataloader, model, loss_fn)

    print("Done!")


if __name__ == '__main__':
    main()
