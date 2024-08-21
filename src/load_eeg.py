"""Load eeg signals."""

import logging
from collections import OrderedDict

import numpy as np

from .util import (
    BBCIDataset,
    create_signal_target_from_raw_mne,
    exponential_running_standardize,
    highpass_cnt,
    mne_apply,
    resample_cnt,
    split_into_two_sets,
)

log = logging.getLogger(__name__)


def load_bbci_data(filename, low_cut_hz, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ["C3", "C4", "C2"]
    # we loaded all sensors to always get same cleaning results
    # independent of sensor selection
    # There is an inbuilt heuristic that tries to use only
    # EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict(
        [
            ("Right Hand", [1]),
            (
                "Left Hand",
                [2],
            ),
            ("Rest", [3]),
            ("Feet", [4]),
        ]
    )
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def, clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info(
        "Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
            np.sum(clean_trial_mask),
            len(set_for_cleaning.X),
            np.mean(clean_trial_mask) * 100,
        )
    )

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = [
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "C3",
        "C4",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "FC3",
        "FCz",
        "FC4",
        "C5",
        "C1",
        "C2",
        "C6",
        "CP3",
        "CPz",
        "CP4",
        "FFC5h",
        "FFC3h",
        "FFC4h",
        "FFC6h",
        "FCC5h",
        "FCC3h",
        "FCC4h",
        "FCC6h",
        "CCP5h",
        "CCP3h",
        "CCP4h",
        "CCP6h",
        "CPP5h",
        "CPP3h",
        "CPP4h",
        "CPP6h",
        "FFC1h",
        "FFC2h",
        "FCC1h",
        "FCC2h",
        "CCP1h",
        "CCP2h",
        "CPP1h",
        "CPP2h",
    ]
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(a, low_cut_hz, cnt.info["sfreq"], filt_order=3, axis=1),
        cnt,
    )
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(
            a.T, factor_new=1e-3, init_block_size=1000, eps=1e-4
        ).T,
        cnt,
    )

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset


def load_train_valid_test(
    train_filename, test_filename, low_cut_hz, valid_set_fraction=0.9, debug=False
):
    log.info("Loading train...")
    full_train_set = load_bbci_data(train_filename, low_cut_hz=low_cut_hz, debug=debug)

    log.info("Loading test...")
    test_set = load_bbci_data(test_filename, low_cut_hz=low_cut_hz, debug=debug)
    if valid_set_fraction < 1.0:
        train_set, valid_set = split_into_two_sets(full_train_set, valid_set_fraction)
    else:
        train_set = full_train_set
        valid_set = None

    log.info("Train set with {:4d} trials".format(len(train_set.X)))
    if valid_set is not None:
        log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    return train_set, valid_set, test_set
