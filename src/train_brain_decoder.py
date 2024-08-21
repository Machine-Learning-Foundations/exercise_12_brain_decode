"""Train brain wave decoder."""

import os
import sys
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm

sys.path.append(".")

from src.load_eeg import load_train_valid_test


def normalize(
    data: th.Tensor,
    mean: Optional[Union[np.float64, None]] = None,
    std: Optional[Union[np.float64, None]] = None,
) -> Tuple[th.Tensor, np.float64, np.float64]:
    """Normalize the input array.

    After normalization the input
    distribution should be approximately standard normal.

    Args:
        data (np.array): The input array.
        mean (float): Data mean, re-computed if None.
            Defaults to None.
        std (float): Data standard deviation,
            re-computed if None. Defaults to None.

    Returns:
        np.array, float, float: Normalized data, mean and std.
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / std, mean, std


class BrainCNN(th.nn.Module):
    """Your Brain-CNN model."""

    def __init__(self):
        """Set up the Neural network.

        Use nn.Conv1d, nn.MaxPool1d, nn.Linear and nn.ReLU building blocks.
        See figure 1 of https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.23730
        for architectural inspiration.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=44, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.linear = nn.Linear(35456, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Run the forward pass of the network."""
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = th.reshape(x, [x.shape[0], -1])
        x = self.linear(x)
        return x


def get_acc(
    net: nn.Module,
    eeg_input: th.Tensor,
    labels: th.Tensor,
) -> th.Tensor:
    """Compute the accuracy.

    Args:
        net (nn.Module): Your cnn object.
        eeg_input (th.Tensor): An array containing the eeg brain waves.
        labels (th.Tensor): The action annotations in a array.

    Returns:
        th.Tensor: The accuracy in [%].
    """
    logits = net(eeg_input)
    accuracy = th.mean((th.argmax(logits, -1) == labels).type(th.float))
    return accuracy


if __name__ == "__main__":
    # Never forget to set the seed!
    th.manual_seed(42)
    low_cut_hz = 0
    subject_id = 1
    batch_size = 50
    epochs = 20

    train_filename = os.path.join("./data", "train/{:d}.mat".format(subject_id))
    test_filename = os.path.join("./data", "test/{:d}.mat".format(subject_id))

    # Create the dataset
    train_set, valid_set_np, test_set_np = load_train_valid_test(
        train_filename=train_filename,
        test_filename=test_filename,
        low_cut_hz=low_cut_hz,
    )

    train_set_x, mean, std = normalize(train_set.X)
    valid_set_x_np, _, _ = normalize(valid_set_np.X, mean, std)
    test_set_x_np, _, _ = normalize(test_set_np.X, mean, std)

    train_size = train_set.X.shape[0]
    train_input = np.array_split(train_set_x, train_size // batch_size)
    train_labels = np.array_split(train_set.y, train_size // batch_size)

    valid_set_y = th.tensor(valid_set_np.y)
    valid_set_x = th.tensor(valid_set_x_np)
    test_set_y = th.tensor(test_set_np.y)
    test_set_x = th.tensor(test_set_x_np)

    cnn = BrainCNN()
    opt = th.optim.Adam(cnn.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    val_acc_list = []
    for e in range(epochs):
        train_loop = tqdm(
            zip(train_input, train_labels),
            total=len(train_input),
            desc="Training Brain CNN",
        )
        for input_x, labels_y in train_loop:
            input_x, _, _ = normalize(input_x, mean, std)
            labels_y = th.tensor(labels_y)
            input_x = th.tensor(input_x)

            y_hat = cnn(input_x)
            cel = loss(y_hat, labels_y)
            cel.backward()
            opt.step()
            opt.zero_grad()
            train_loop.set_description("Loss: {:2.3f}".format(cel))

        val_accuracy = get_acc(cnn, valid_set_x, valid_set_y)
        print("Validation accuracy {:2.3f} at epoch {}".format(val_accuracy, e + 1))  # type: ignore
        val_acc_list.append(val_accuracy)

    test_accuracy = get_acc(cnn, test_set_x, test_set_y)
    print("Test accuracy: {:2.3f}".format(test_accuracy))  # type: ignore
    plt.plot(val_acc_list, label="Validation accuracy")
    plt.plot(len(val_acc_list) - 1, test_accuracy, ".", label="Test accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
