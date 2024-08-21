"""Test data loading."""

import os
import sys

sys.path.insert(0, "./src/")

from src.load_eeg import load_train_valid_test


def test_brain_load():
    """Test if dataloader returns the correct matrices."""
    low_cut_hz = 0
    subject_id = 1

    train_filename = os.path.join("./data", "train/{:d}.mat".format(subject_id))
    test_filename = os.path.join("./data", "test/{:d}.mat".format(subject_id))

    # Create the dataset
    train_set, valid_set, test_set = load_train_valid_test(
        train_filename=train_filename,
        test_filename=test_filename,
        low_cut_hz=low_cut_hz,
    )

    assert train_set.X.shape == (287, 44, 1125)
    assert train_set.y.shape == (287,)
    assert valid_set.X.shape == (32, 44, 1125)
    assert valid_set.y.shape == (32,)
    assert test_set.X.shape == (160, 44, 1125)
    assert test_set.y.shape == (160,)
