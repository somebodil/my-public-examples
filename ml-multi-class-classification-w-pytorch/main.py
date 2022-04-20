import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch import nn, optim


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=200)
        self.linear2 = nn.Linear(in_features=200, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def generate_dataset():
    r = np.random.rand(10000) * 3
    theta = np.random.rand(10000) * 2 * np.pi
    y = r.astype(int)
    r = r * (np.cos(theta) + 1)
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    X = np.array([x1, x2]).T

    train_X, train_y = X[:8000, :], y[:8000]
    val_X, val_y = X[8000:9000, :], y[8000:9000]
    test_X, test_y = X[9000:, :], y[9000:]

    # ====== Visualize Each Dataset ====== #
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(train_X[:, 0], train_X[:, 1], c=train_y, s=0.7)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Train Set Distribution')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(val_X[:, 0], val_X[:, 1], c=val_y)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Validation Set Distribution')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(test_X[:, 0], test_X[:, 1], c=test_y)
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_title('Test Set Distribution')

    plt.show()

    return train_X, train_y, val_X, val_y, test_X, test_y


def main():
    # Hyper parameter ---
    epoch = 1000
    lr = 0.05

    # Generate data ---
    train_X, train_y, val_X, val_y, test_X, test_y = generate_dataset()

    # Prepare model, loss function, optimizer ---
    model = MLPModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train and Validate ---

    for i in range(epoch):
        # Train - Print or Save train error
        model.train()

        input_x = torch.Tensor(train_X)
        true_y = torch.Tensor(train_y).long()

        optimizer.zero_grad()
        pred_y = model(input_x)
        test_loss = loss_fn(pred_y, true_y)
        test_loss.backward()
        optimizer.step()

        # Validate - Print or Save Validate error
        model.eval()

        with torch.no_grad():
            input_x = torch.Tensor(val_X)
            true_y = torch.Tensor(val_y).long()
            pred_y = model(input_x)
            val_loss = loss_fn(pred_y, true_y)

            print(f'train loss : {test_loss.detach().numpy()}, val_loss : {val_loss.detach().numpy()}')

    # Test --
    model.eval()

    input_x = torch.Tensor(test_X)
    true_y = torch.Tensor(test_y).long()
    pred_y = model(input_x)

    test_loss = loss_fn(pred_y, true_y)
    print(f'test_loss : {test_loss.detach().numpy()}')

    # Report -- (optional)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(test_X[:, 0], test_X[:, 1], c=test_y)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('True test y')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(test_X[:, 0], test_X[:, 1], c=pred_y.max(dim=1)[1].numpy())
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Predicted test y')

    plt.show()

    print(f'Test accuracy : {accuracy_score(true_y, pred_y.max(dim=1)[1].numpy()) * 100} %')


if __name__ == '__main__':
    main()
