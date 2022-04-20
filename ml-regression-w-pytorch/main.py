import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, x):
        return self.linear(x)


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=200)
        self.linear2 = nn.Linear(in_features=200, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def generate_dataset(data_num):
    """
    y = 2*sin(x1)+log(1/2*x2^2)+e
    """

    # ====== Generating Dataset ====== #
    x1 = np.random.rand(data_num) * 10
    x2 = np.random.rand(data_num) * 10
    e = np.random.normal(0, 0.5, data_num)
    X = np.array([x1, x2]).T

    print(x1.shape)
    print(x2.shape)
    print(X.shape)

    y = 2 * np.sin(x1) + np.log(0.5 * x2 ** 2) + e

    print(y.shape)

    # ====== Split Dataset into Train, Validation, Test ====== #
    train_X, train_y = X[:1600, :], y[:1600]
    val_X, val_y = X[1600:2000, :], y[1600:2000]
    test_X, test_y = X[2000:, :], y[2000:]

    # ====== Visualize Each Dataset ====== #
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(train_X[:, 0], train_X[:, 1], train_y, c=train_y, cmap='jet')

    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('y')
    ax1.set_title('Train Set Distribution')
    ax1.invert_xaxis()

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(val_X[:, 0], val_X[:, 1], val_y, c=val_y, cmap='jet')

    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('y')
    ax2.set_title('Validation Set Distribution')
    ax2.invert_xaxis()

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(test_X[:, 0], test_X[:, 1], test_y, c=test_y, cmap='jet')

    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_zlabel('y')
    ax3.set_title('Test Set Distribution')
    ax3.invert_xaxis()

    plt.show()

    return train_X, train_y, val_X, val_y, test_X, test_y


def main():
    # Hyper parameter ---

    learning_rate = 0.005
    epoch = 1000

    # Generate data ---

    train_X, train_y, val_X, val_y, test_X, test_y = generate_dataset(2400)  # num of train val test = 1600 / 400 / 400

    # Prepare model, loss function, optimizer ---

    # model = LinearModel()  # Try MLPModel() and see the performance difference
    model = MLPModel()  # Try MLPModel() and see the performance difference
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Uncomment for testing Loss Function
    #
    # test_pred_y = torch.Tensor([0,0,0,0])
    # test_true_y = torch.Tensor([0,1,0,1])
    #
    # print(loss_fn(test_pred_y, test_true_y))
    # print(loss_fn(test_true_y, test_true_y))

    # Train and Validate ---

    for i in range(epoch):
        # Train ---

        model.train()

        input_x = torch.Tensor(train_X)
        true_y = torch.Tensor(train_y)

        optimizer.zero_grad()
        pred_y = model(input_x)
        train_loss = loss_fn(pred_y.squeeze(), true_y)
        train_loss.backward()
        optimizer.step()

        # Validate ---

        model.eval()

        with torch.no_grad():
            input_x = torch.Tensor(val_X)
            true_y = torch.Tensor(val_y)
            pred_y = model(input_x)

            val_loss = loss_fn(pred_y.squeeze(), true_y)

            print(f'test_loss : {train_loss.detach().numpy()}, val_loss : {val_loss.detach().numpy()}')

    # Test --

    model.eval()
    with torch.no_grad():
        input_x = torch.Tensor(test_X)
        true_y = torch.Tensor(test_y)
        pred_y = model(input_x)
        test_loss = loss_fn(pred_y.squeeze(), true_y)
        print(f'test_loss : {test_loss}')

    # Report -- (optional)
    # ...


if __name__ == '__main__':
    main()
