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
    # TODO: Return the normalized data as well as the mean and the standard deviation.
    return None


class BrainCNN(th.nn.Module):
    """Your Brain-CNN model."""

    def __init__(self):
        """Set up the Neural network.

        Use nn.Conv1d, nn.MaxPool1d, nn.Linear and nn.ReLU building blocks.
        See figure 1 of https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.23730
        for architectural inspiration.
        """
        super().__init__()
        # TODO: Implement me!!

    def forward(self, x):
        """Run the forward pass of the network."""
        # TODO: Return the result of the forward pass instead of 0.
        return 0.


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
    # TODO: Compute the correct accuracy.
    accuracy = 0.
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

    # TODO: Set up Network training with validation and a final test-accuracy measurement.
    # Use PyTorch's Adam optimizer.
    # Use the X and y attributes of the set objects to access the EEG measurements
    # and corresponding labels.