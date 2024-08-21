"""Load data loading utility."""

import logging
import os.path
import re
from collections import Counter, OrderedDict
from copy import deepcopy
from glob import glob

import h5py
import mne
import numpy as np
import pandas as pd
import resampy

log = logging.getLogger(__name__)


def split_into_two_sets(dataset, first_set_fraction=None, n_first_set=None):
    """
    Split set into two sets either by fraction of first set or by number
    of trials in first set.
    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    first_set_fraction: float, optional
        Fraction of trials in first set.
    n_first_set: int, optional
        Number of trials in first set
    Returns
    -------
    first_set, second_set: :class:`.SignalAndTarget`
        The two splitted sets.
    """
    assert (first_set_fraction is None) != (
        n_first_set is None
    ), "Pass either first_set_fraction or n_first_set"
    if n_first_set is None:
        n_first_set = int(round(len(dataset.X) * first_set_fraction))
    assert n_first_set < len(dataset.X)
    first_set = apply_to_X_y(lambda a: a[:n_first_set], dataset)
    second_set = apply_to_X_y(lambda a: a[n_first_set:], dataset)
    return first_set, second_set


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


class BBCIDataset(object):
    """
    Loader class for files created by saving BBCI files in matlab (make
    sure to save with '-v7.3' in matlab, see
    https://de.mathworks.com/help/matlab/import_export/mat-file-versions.html#buk6i87
    )
    Parameters
    ----------
    filename: str
    load_sensor_names: list of str, optional
        Also speeds up loading if you only load some sensors.
        None means load all sensors.
    check_class_names: bool, optional
        check if the class names are part of some known class names at
        Translational NeuroTechnology Lab, AG Ball, Freiburg, Germany.
    """

    def __init__(self, filename, load_sensor_names=None, check_class_names=False):
        self.__dict__.update(locals())
        del self.self

    def load(self):
        cnt = self._load_continuous_signal()
        cnt = self._add_markers(cnt)
        return cnt

    def _load_continuous_signal(self):
        # loads time domain data.
        wanted_chan_inds, wanted_sensor_names = self._determine_sensors()
        fs = self._determine_samplingrate()
        with h5py.File(self.filename, "r") as h5file:
            samples = int(h5file["nfo"]["T"][0, 0])
            cnt_signal_shape = (samples, len(wanted_chan_inds))
            continuous_signal = np.ones(cnt_signal_shape, dtype=np.float32) * np.nan
            for chan_ind_arr, chan_ind_set in enumerate(wanted_chan_inds):
                # + 1 because matlab/this hdf5-naming logic
                # has 1-based indexing
                # i.e ch1,ch2,....
                chan_set_name = "ch" + str(chan_ind_set + 1)
                # first 0 to unpack into vector, before it is 1xN matrix
                chan_signal = h5file[chan_set_name][
                    :
                ].squeeze()  # already load into memory
                continuous_signal[:, chan_ind_arr] = chan_signal
            assert not np.any(np.isnan(continuous_signal)), "No NaNs expected in signal"

        if self.load_sensor_names is None:
            ch_types = ["EEG"] * len(wanted_chan_inds)
        else:
            # Assume we cant know channel type here automatically
            ch_types = ["misc"] * len(wanted_chan_inds)
        info = mne.create_info(
            ch_names=wanted_sensor_names, sfreq=fs, ch_types=ch_types
        )
        cnt = mne.io.RawArray(continuous_signal.T, info)
        return cnt

    def _determine_sensors(self):
        all_sensor_names = self.get_all_sensors(self.filename, pattern=None)
        if self.load_sensor_names is None:
            # if no sensor names given, take all EEG-chans
            eeg_sensor_names = all_sensor_names
            eeg_sensor_names = filter(
                lambda s: not s.startswith("BIP"), eeg_sensor_names
            )
            eeg_sensor_names = filter(lambda s: not s.startswith("E"), eeg_sensor_names)
            eeg_sensor_names = filter(
                lambda s: not s.startswith("Microphone"), eeg_sensor_names
            )
            eeg_sensor_names = filter(
                lambda s: not s.startswith("Breath"), eeg_sensor_names
            )
            eeg_sensor_names = filter(
                lambda s: not s.startswith("GSR"), eeg_sensor_names
            )
            eeg_sensor_names = list(eeg_sensor_names)
            assert (
                len(eeg_sensor_names) == 128
                or len(eeg_sensor_names) == 64
                or len(eeg_sensor_names) == 32
                or len(eeg_sensor_names) == 16
            ), "Recheck this code if you have different sensors..."
            self.load_sensor_names = eeg_sensor_names
        chan_inds = self._determine_chan_inds(all_sensor_names, self.load_sensor_names)
        return chan_inds, self.load_sensor_names

    def _determine_samplingrate(self):
        with h5py.File(self.filename, "r") as h5file:
            fs = h5file["nfo"]["fs"][0, 0]
            assert isinstance(fs, int) or fs.is_integer()
            fs = int(fs)
        return fs

    @staticmethod
    def _determine_chan_inds(all_sensor_names, sensor_names):
        assert sensor_names is not None
        chan_inds = [all_sensor_names.index(s) for s in sensor_names]
        assert len(chan_inds) == len(sensor_names), "All" "sensors should be there."
        assert len(set(chan_inds)) == len(chan_inds), "No" "duplicated sensors wanted."
        return chan_inds

    @staticmethod
    def get_all_sensors(filename, pattern=None):
        """
        Get all sensors that exist in the given file.
        Parameters
        ----------
        filename: str
        pattern: str, optional
            Only return those sensor names that match the given pattern.
        Returns
        -------
        sensor_names: list of str
            Sensor names that match the pattern or all sensor names in the file.
        """
        with h5py.File(filename, "r") as h5file:
            clab_set = h5file["nfo"]["clab"][:].squeeze()
            all_sensor_names = [
                "".join(chr(c[0]) for c in h5file[obj_ref]) for obj_ref in clab_set
            ]
            if pattern is not None:
                all_sensor_names = filter(
                    lambda sname: re.search(pattern, sname), all_sensor_names
                )
        return all_sensor_names

    def _add_markers(self, cnt):
        with h5py.File(self.filename, "r") as h5file:
            event_times_in_ms = h5file["mrk"]["time"][:].squeeze()
            event_classes = h5file["mrk"]["event"]["desc"][:].squeeze().astype(np.int64)

            # Check whether class names known and correct order
            class_name_set = h5file["nfo"]["className"][:].squeeze()
            all_class_names = [
                "".join(chr(c[0]) for c in h5file[obj_ref])
                for obj_ref in class_name_set
            ]

            if self.check_class_names:
                _check_class_names(all_class_names, event_times_in_ms, event_classes)

        event_times_in_samples = event_times_in_ms * cnt.info["sfreq"] / 1000.0
        event_times_in_samples = np.uint32(np.round(event_times_in_samples))

        # Check if there are markers at the same time
        previous_i_sample = -1
        for i_event, (i_sample, id_class) in enumerate(
            zip(event_times_in_samples, event_classes)
        ):
            if i_sample == previous_i_sample:
                log.warning(
                    "Same sample has at least two markers.\n"
                    "{:d}: ({:.0f} and {:.0f}).\n".format(
                        i_sample, event_classes[i_event - 1], event_classes[i_event]
                    )
                    + "Marker codes will be summed."
                )
            previous_i_sample = i_sample

        # Now create stim chan
        stim_chan = np.zeros_like(cnt.get_data()[0])
        for i_sample, id_class in zip(event_times_in_samples, event_classes):
            stim_chan[i_sample] += id_class
        info = mne.create_info(
            ch_names=["STI 014"], sfreq=cnt.info["sfreq"], ch_types=["stim"]
        )
        stim_cnt = mne.io.RawArray(stim_chan[None], info, verbose="WARNING")
        cnt = cnt.add_channels([stim_cnt])
        event_arr = [
            event_times_in_samples,
            [0] * len(event_times_in_samples),
            event_classes,
        ]
        cnt.info["events"] = np.array(event_arr).T
        return cnt


def _check_class_names(all_class_names, event_times_in_ms, event_classes):
    """
    Checks if the class names are part of some known class names used in
    translational neurotechnology lab, AG Ball, Freiburg.
    Logs warning in case class names are not known.
    Parameters
    ----------
    all_class_names: list of str
    event_times_in_ms: list of number
    event_classes: list of number
    """
    if all_class_names == ["Right Hand", "Left Hand", "Rest", "Feet"]:
        pass
    elif (
        (
            all_class_names
            == [
                "1",
                "10",
                "11",
                "111",
                "12",
                "13",
                "150",
                "2",
                "20",
                "22",
                "3",
                "30",
                "33",
                "4",
                "40",
                "44",
                "99",
            ]
        )
        or (
            all_class_names
            == [
                "1",
                "10",
                "11",
                "12",
                "13",
                "150",
                "2",
                "20",
                "22",
                "3",
                "30",
                "33",
                "4",
                "40",
                "44",
                "99",
            ]
        )
        or (all_class_names == ["1", "2", "3", "4"])
    ):
        pass  # Semantic classes
    elif all_class_names == ["Rest", "Feet", "Left Hand", "Right Hand"]:
        # Have to swap from
        # ['Rest', 'Feet', 'Left Hand', 'Right Hand']
        # to
        # ['Right Hand', 'Left Hand', 'Rest', 'Feet']
        right_mask = event_classes == 4
        left_mask = event_classes == 3
        rest_mask = event_classes == 1
        feet_mask = event_classes == 2
        event_classes[right_mask] = 1
        event_classes[left_mask] = 2
        event_classes[rest_mask] = 3
        event_classes[feet_mask] = 4
        log.warn(
            "Swapped  class names {:s}... might cause problems...".format(
                all_class_names
            )
        )
    elif all_class_names == [
        "Right Hand Start",
        "Left Hand Start",
        "Rest Start",
        "Feet Start",
        "Right Hand End",
        "Left Hand End",
        "Rest End",
        "Feet End",
    ]:
        pass
    elif all_class_names == [
        "Right Hand",
        "Left Hand",
        "Rest",
        "Feet",
        "Face",
        "Navigation",
        "Music",
        "Rotation",
        "Subtraction",
        "Words",
    ]:
        pass  # robot hall 10 class decoding
    elif all_class_names == [
        "RightHand",
        "Feet",
        "Rotation",
        "Words",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "RightHand_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Feet_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Rotation_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Words_End",
    ] or all_class_names == [
        "RightHand",
        "Feet",
        "Rotation",
        "Words",
        "Rest",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "RightHand_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Feet_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Rotation_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Words_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Rest_End",
    ]:
        pass  # weird stuff when we recorded cursor in robot hall
        # on 2016-09-14 and 2016-09-16 :D

    elif all_class_names == [
        "0004",
        "0016",
        "0032",
        "0056",
        "0064",
        "0088",
        "0095",
        "0120",
    ]:
        pass
    elif all_class_names == ["0004", "0056", "0088", "0120"]:
        pass
    elif all_class_names == [
        "0004",
        "0016",
        "0032",
        "0048",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0120",
    ]:
        pass
    elif all_class_names == ["0004", "0016", "0056", "0088", "0120", "__"]:
        pass
    elif all_class_names == ["0004", "0056", "0088", "0120", "__"]:
        pass
    elif all_class_names == [
        "0004",
        "0032",
        "0048",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0120",
        "__",
    ]:
        pass
    elif all_class_names == ["0004", "0056", "0080", "0088", "0096", "0120", "__"]:
        pass
    elif all_class_names == [
        "0004",
        "0032",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0120",
    ]:
        pass
    elif all_class_names == [
        "0004",
        "0032",
        "0048",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0120",
    ]:
        pass
    elif all_class_names == [
        "0004",
        "0016",
        "0032",
        "0048",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0096",
        "0120",
    ]:
        pass
    elif all_class_names == ["4", "16", "32", "56", "64", "88", "95", "120"]:
        pass
    elif all_class_names == ["4", "56", "88", "120"]:
        pass
    elif all_class_names == [
        "4",
        "16",
        "32",
        "48",
        "56",
        "64",
        "80",
        "88",
        "95",
        "120",
    ]:
        pass
    elif all_class_names == ["0", "4", "56", "88", "120"]:
        pass
    elif all_class_names == ["0", "4", "16", "56", "88", "120"]:
        pass
    elif all_class_names == ["0", "4", "32", "48", "56", "64", "80", "88", "95", "120"]:
        pass
    elif all_class_names == ["0", "4", "56", "80", "88", "96", "120"]:
        pass
    elif all_class_names == ["4", "32", "56", "64", "80", "88", "95", "120"]:
        pass
    elif all_class_names == ["One", "Two", "Three", "Four"]:
        pass
    elif all_class_names == ["1", "10", "11", "12", "2", "20", "3", "30", "4", "40"]:
        pass
    elif all_class_names == ["1", "10", "12", "13", "2", "20", "3", "30", "4", "40"]:
        pass
    elif all_class_names == ["1", "10", "13", "2", "20", "3", "30", "4", "40", "99"]:
        pass
    elif all_class_names == [
        "1",
        "10",
        "11",
        "14",
        "18",
        "20",
        "21",
        "24",
        "251",
        "252",
        "28",
        "30",
        "4",
        "8",
    ]:
        pass
    elif all_class_names == [
        "1",
        "10",
        "11",
        "14",
        "18",
        "20",
        "21",
        "24",
        "252",
        "253",
        "28",
        "30",
        "4",
        "8",
    ]:
        pass
    elif len(event_times_in_ms) == len(all_class_names):
        pass  # weird neuroone(?) logic where class names have event classes
    elif all_class_names == [
        "Right_hand_stimulus_onset",
        "Feet_stimulus_onset",
        "Rotation_stimulus_onset",
        "Words_stimulus_onset",
        "Right_hand_stimulus_offset",
        "Feet_stimulus_offset",
        "Rotation_stimulus_offset",
        "Words_stimulus_offset",
    ]:
        pass
    else:
        # remove this whole if else stuffs?
        log.warn("Unknown class names {:s}".format(all_class_names))


def load_bbci_sets_from_folder(folder, runs="all"):
    """
    Load bbci datasets from files in given folder.
    Parameters
    ----------
    folder: str
        Folder with .BBCI.mat files inside
    runs: list of int
        If you only want to load specific runs.
        Assumes filenames with such kind of part: S001R02 for Run 2.
        Tries to match this regex: ``'S[0-9]{3,3}R[0-9]{2,2}_'``.
    Returns
    -------
    """
    bbci_mat_files = sorted(glob(os.path.join(folder, "*.BBCI.mat")))
    if runs != "all":
        file_run_numbers = [
            int(re.search("S[0-9]{3,3}R[0-9]{2,2}_", f).group()[5:7])
            for f in bbci_mat_files
        ]
        indices = [file_run_numbers.index(num) for num in runs]

        wanted_files = np.array(bbci_mat_files)[indices]
    else:
        wanted_files = bbci_mat_files
    cnts = []
    for f in wanted_files:
        log.info("Loading {:s}".format(f))
        cnts.append(BBCIDataset(f).load())
    return cnts


def create_signal_target_from_raw_mne(
    raw,
    name_to_start_codes,
    epoch_ival_ms,
    name_to_stop_codes=None,
    prepad_trials_to_n_samples=None,
    one_hot_labels=False,
    one_label_per_trial=True,
):
    """
    Create SignalTarget set from given `mne.io.RawArray`.
    Parameters
    ----------
    raw: `mne.io.RawArray`
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. In case only `name_to_codes` given,
        represents start offset and stop offset from start markers. In case
        `name_to_stop_codes` given, represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.
    prepad_trials_to_n_samples: int
        Pad trials that would be too short with the signal before it (only
        valid if name_to_stop_codes is not None).
    one_hot_labels: bool, optional
        Whether to have the labels in a one-hot format, e.g. [0,0,1] or to
        have them just as an int, e.g. 2
    one_label_per_trial: bool, optional
        Whether to have a timeseries of labels or just a single label per trial.
    Returns
    -------
    dataset: :class:`.SignalAndTarget`
        Dataset with `X` as the trial signals and `y` as the trial labels.
    """
    data = raw.get_data()
    events = np.array([raw.info["events"][:, 0], raw.info["events"][:, 2]]).T
    fs = raw.info["sfreq"]
    return create_signal_target(
        data,
        events,
        fs,
        name_to_start_codes,
        epoch_ival_ms,
        name_to_stop_codes=name_to_stop_codes,
        prepad_trials_to_n_samples=prepad_trials_to_n_samples,
        one_hot_labels=one_hot_labels,
        one_label_per_trial=one_label_per_trial,
    )


def create_signal_target(
    data,
    events,
    fs,
    name_to_start_codes,
    epoch_ival_ms,
    name_to_stop_codes=None,
    prepad_trials_to_n_samples=None,
    one_hot_labels=False,
    one_label_per_trial=True,
):
    """
    Create SignalTarget set given continuous data.
    Parameters
    ----------
    data: 2d-array of number
        The continuous recorded data. Channels x times order.
    events: 2d-array
        Dimensions: Number of events, 2. For each event, should contain sample
        index and marker code.
    fs: number
        Sampling rate.
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. In case only `name_to_codes` given,
        represents start offset and stop offset from start markers. In case
        `name_to_stop_codes` given, represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.
    prepad_trials_to_n_samples: int, optional
        Pad trials that would be too short with the signal before it (only
        valid if name_to_stop_codes is not None).
    one_hot_labels: bool, optional
        Whether to have the labels in a one-hot format, e.g. [0,0,1] or to
        have them just as an int, e.g. 2
    one_label_per_trial: bool, optional
        Whether to have a timeseries of labels or just a single label per trial.
    Returns
    -------
    dataset: :class:`.SignalAndTarget`
        Dataset with `X` as the trial signals and `y` as the trial labels.
    """
    if name_to_stop_codes is None:
        return _create_signal_target_from_start_and_ival(
            data,
            events,
            fs,
            name_to_start_codes,
            epoch_ival_ms,
            one_hot_labels=one_hot_labels,
            one_label_per_trial=one_label_per_trial,
        )
    else:
        return _create_signal_target_from_start_and_stop(
            data,
            events,
            fs,
            name_to_start_codes,
            epoch_ival_ms,
            name_to_stop_codes,
            prepad_trials_to_n_samples,
            one_hot_labels=one_hot_labels,
            one_label_per_trial=one_label_per_trial,
        )


def _to_mrk_code_to_name_and_y(name_to_codes):
    # Create mapping from marker code to class name and y=classindex
    mrk_code_to_name_and_y = {}
    for i_class, class_name in enumerate(name_to_codes):
        codes = name_to_codes[class_name]
        if hasattr(codes, "__len__"):
            for code in codes:
                assert code not in mrk_code_to_name_and_y
                mrk_code_to_name_and_y[code] = (class_name, i_class)
        else:
            assert codes not in mrk_code_to_name_and_y
            mrk_code_to_name_and_y[codes] = (class_name, i_class)
    return mrk_code_to_name_and_y


def _create_signal_target_from_start_and_ival(
    data, events, fs, name_to_codes, epoch_ival_ms, one_hot_labels, one_label_per_trial
):
    cnt_y, i_start_stops = _create_cnt_y_and_trial_bounds_from_start_and_ival(
        data.shape[1], events, fs, name_to_codes, epoch_ival_ms
    )
    signal_target = _create_signal_target_from_cnt_y_start_stops(
        data,
        cnt_y,
        i_start_stops,
        prepad_trials_to_n_samples=None,
        one_hot_labels=one_hot_labels,
        one_label_per_trial=one_label_per_trial,
    )
    # make into arrray as all should have same dimensions
    signal_target.X = np.array(signal_target.X, dtype=np.float32)
    signal_target.y = np.array(signal_target.y, dtype=np.int64)
    return signal_target


def _create_cnt_y_and_trial_bounds_from_start_and_ival(
    n_samples, events, fs, name_to_start_codes, epoch_ival_ms
):
    ival_in_samples = ms_to_samples(np.array(epoch_ival_ms), fs)
    start_offset = np.int32(np.round(ival_in_samples[0]))
    # we will use ceil but exclusive...
    stop_offset = np.int32(np.ceil(ival_in_samples[1]))
    mrk_code_to_name_and_y = _to_mrk_code_to_name_and_y(name_to_start_codes)
    class_to_n_trials = Counter()
    n_classes = len(name_to_start_codes)
    cnt_y = np.zeros((n_samples, n_classes), dtype=np.int64)
    i_start_stops = []
    for i_sample, mrk_code in zip(events[:, 0], events[:, 1]):
        start_sample = int(i_sample) + start_offset
        stop_sample = int(i_sample) + stop_offset
        if mrk_code in mrk_code_to_name_and_y:
            if start_sample < 0:
                log.warning(
                    "Ignore trial with marker code {:d}, would start at "
                    "sample {:d}".format(mrk_code, start_sample)
                )
                continue
            if stop_sample > n_samples:
                log.warning(
                    "Ignore trial with marker code {:d}, would end at "
                    "sample {:d} of {:d}".format(
                        mrk_code, stop_sample - 1, n_samples - 1
                    )
                )
                continue

            name, this_y = mrk_code_to_name_and_y[mrk_code]
            i_start_stops.append((start_sample, stop_sample))
            cnt_y[start_sample:stop_sample, this_y] = 1
            class_to_n_trials[name] += 1
    log.info("Trial per class:\n{:s}".format(str(class_to_n_trials)))
    return cnt_y, i_start_stops


def _create_signal_target_from_start_and_stop(
    data,
    events,
    fs,
    name_to_start_codes,
    epoch_ival_ms,
    name_to_stop_codes,
    prepad_trials_to_n_samples,
    one_hot_labels,
    one_label_per_trial,
):
    assert np.array_equal(
        list(name_to_start_codes.keys()), list(name_to_stop_codes.keys())
    )
    cnt_y, i_start_stops = _create_cnt_y_and_trial_bounds_from_start_stop(
        data.shape[1],
        events,
        fs,
        name_to_start_codes,
        epoch_ival_ms,
        name_to_stop_codes,
    )
    signal_target = _create_signal_target_from_cnt_y_start_stops(
        data,
        cnt_y,
        i_start_stops,
        prepad_trials_to_n_samples=prepad_trials_to_n_samples,
        one_hot_labels=one_hot_labels,
        one_label_per_trial=one_label_per_trial,
    )
    return signal_target


def _create_cnt_y_and_trial_bounds_from_start_stop(
    n_samples, events, fs, name_to_start_codes, epoch_ival_ms, name_to_stop_codes
):
    """
    Create a one-hot-encoded continuous marker array (cnt_y).
    Parameters
    ----------
    n_samples: int
        Number of samples=timesteps in the recorded data.
    events: 2d-array
        Dimensions: Number of events, 2. For each event, should contain sample
        index and marker code.
    fs: number
        Sampling rate.
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. In case only `name_to_codes` given,
        represents start offset and stop offset from start markers. In case
        `name_to_stop_codes` given, represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.
    Returns
    -------
    cnt_y: 2d-array
        Timeseries of one-hot-labels, time x classes.
    trial_bounds: list of (int,int)
        List of (trial_start, trial_stop) tuples.
    """
    assert np.array_equal(
        list(name_to_start_codes.keys()), list(name_to_stop_codes.keys())
    )
    events = np.asarray(events)
    ival_in_samples = ms_to_samples(np.array(epoch_ival_ms), fs)
    start_offset = np.int32(np.round(ival_in_samples[0]))
    # we will use ceil but exclusive...
    stop_offset = np.int32(np.ceil(ival_in_samples[1]))
    start_code_to_name_and_y = _to_mrk_code_to_name_and_y(name_to_start_codes)
    # Ensure all stop marker codes are iterables
    for name in name_to_stop_codes:
        codes = name_to_stop_codes[name]
        if not hasattr(codes, "__len__"):
            name_to_stop_codes[name] = [codes]
    all_stop_codes = np.concatenate(list(name_to_stop_codes.values())).astype(np.int64)
    class_to_n_trials = Counter()
    n_classes = len(name_to_start_codes)
    cnt_y = np.zeros((n_samples, n_classes), dtype=np.int64)

    event_samples = events[:, 0]
    event_codes = events[:, 1]
    i_start_stops = []
    i_event = 0
    first_start_code_found = False
    while i_event < len(events):
        while i_event < len(events) and (
            event_codes[i_event] not in start_code_to_name_and_y
        ):
            i_event += 1
        if i_event < len(events):
            start_sample = event_samples[i_event]
            start_code = event_codes[i_event]
            start_name = start_code_to_name_and_y[start_code][0]
            start_y = start_code_to_name_and_y[start_code][1]
            i_event += 1
            first_start_code_found = True
            waiting_for_end_code = True

            while i_event < len(events) and (
                event_codes[i_event] not in all_stop_codes
            ):
                if event_codes[i_event] in start_code_to_name_and_y:
                    log.warning(
                        "New start marker  {:.0f} at {:.0f} samples found, "
                        "no end marker for earlier start marker {:.0f} "
                        "at {:.0f} samples found.".format(
                            event_codes[i_event],
                            event_samples[i_event],
                            start_code,
                            start_sample,
                        )
                    )
                    start_sample = event_samples[i_event]
                    start_name = start_code_to_name_and_y[start_code][0]
                    start_code = event_codes[i_event]
                    start_y = start_code_to_name_and_y[start_code][1]
                i_event += 1
        if i_event == len(events):
            if waiting_for_end_code:
                log.warning(
                    (
                        "No end marker for start marker code {:.0f} "
                        "at sample {:.0f} found."
                    ).format(start_code, start_sample)
                )
            elif not first_start_code_found:
                log.warning("No markers found at all.")
            break
        stop_sample = event_samples[i_event]
        stop_code = event_codes[i_event]
        assert stop_code in name_to_stop_codes[start_name]
        i_start = int(start_sample) + start_offset
        i_stop = int(stop_sample) + stop_offset
        cnt_y[i_start:i_stop, start_y] = 1
        i_start_stops.append((i_start, i_stop))
        class_to_n_trials[start_name] += 1
        waiting_for_end_code = False

    log.info("Trial per class:\n{:s}".format(str(class_to_n_trials)))
    return cnt_y, i_start_stops


def _create_signal_target_from_cnt_y_start_stops(
    data,
    cnt_y,
    i_start_stops,
    prepad_trials_to_n_samples,
    one_hot_labels,
    one_label_per_trial,
):
    if prepad_trials_to_n_samples is not None:
        new_i_start_stops = []
        for i_start, i_stop in i_start_stops:
            if (i_stop - i_start) > prepad_trials_to_n_samples:
                new_i_start_stops.append((i_start, i_stop))
            elif i_stop >= prepad_trials_to_n_samples:
                new_i_start_stops.append((i_stop - prepad_trials_to_n_samples, i_stop))
            else:
                log.warning(
                    "Could not pad trial enough, therefore not "
                    "not using trial from {:d} to {:d}".format(i_start, i_stop)
                )
                continue

    else:
        new_i_start_stops = i_start_stops

    X = []
    y = []
    for i_start, i_stop in new_i_start_stops:
        if i_start < 0:
            log.warning(
                "Trial start too early, therefore not "
                "not using trial from {:d} to {:d}".format(i_start, i_stop)
            )
            continue
        if i_stop > data.shape[1]:
            log.warning(
                "Trial stop too late (past {:d}), therefore not "
                "not using trial from {:d} to {:d}".format(
                    data.shape[1] - 1, i_start, i_stop
                )
            )
            continue
        X.append(data[:, i_start:i_stop].astype(np.float32))
        y.append(cnt_y[i_start:i_stop])

    # take last label always
    if one_label_per_trial:
        new_y = []
        for this_y in y:
            # if destroying one hot later, just set most occuring class to 1
            unique_labels, counts = np.unique(this_y, axis=0, return_counts=True)
            if not one_hot_labels:
                meaned_y = np.mean(this_y, axis=0)
                this_new_y = np.zeros_like(meaned_y)
                this_new_y[np.argmax(meaned_y)] = 1
            else:
                # take most frequency occurring label combination
                this_new_y = unique_labels[np.argmax(counts)]

            if len(unique_labels) > 1:
                log.warning(
                    "Different labels within one trial: {:s},"
                    "setting single trial label to  {:s}".format(
                        str(unique_labels), str(this_new_y)
                    )
                )
            new_y.append(this_new_y)
        y = new_y
    if not one_hot_labels:
        # change from one-hot-encoding to regular encoding
        # with -1 as indication none of the classes are present
        new_y = []
        for this_y in y:
            if one_label_per_trial:
                if np.sum(this_y) == 0:
                    this_new_y = -1
                else:
                    this_new_y = np.argmax(this_y)
                if np.sum(this_y) > 1:
                    log.warning(
                        "Have multiple active classes and will convert to "
                        "lowest class"
                    )
            else:
                if np.max(np.sum(this_y, axis=1)) > 1:
                    log.warning(
                        "Have multiple active classes and will convert to "
                        "lowest class"
                    )
                this_new_y = np.argmax(this_y, axis=1)
                this_new_y[np.sum(this_y, axis=1) == 0] = -1
            new_y.append(this_new_y)
        y = new_y
    if one_label_per_trial:
        y = np.array(y, dtype=np.int64)

    return SignalAndTarget(X, y)


def create_signal_target_with_breaks_from_mne(
    cnt,
    name_to_start_codes,
    trial_epoch_ival_ms,
    name_to_stop_codes,
    min_break_length_ms,
    max_break_length_ms,
    break_epoch_ival_ms,
    prepad_trials_to_n_samples=None,
):
    """
    Create SignalTarget set from given `mne.io.RawArray`.
    Parameters
    ----------
    cnt: `mne.io.RawArray`
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    trial_epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. Represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.
    min_break_length_ms: number
        Breaks below this length are excluded.
    max_break_length_ms: number
        Breaks above this length are excluded.
    break_epoch_ival_ms: number
        Break ival, offset from trial end to start of the break in ms and
        offset from trial start to end of break in ms.
    prepad_trials_to_n_samples: int
        Pad trials that would be too short with the signal before it (only
        valid if name_to_stop_codes is not None).
    Returns
    -------
    dataset: :class:`.SignalAndTarget`
        Dataset with `X` as the trial signals and `y` as the trial labels.
        Labels as timeseries and of integers, i.e., not one-hot encoded.
    """
    assert "Break" not in name_to_start_codes
    # Create new marker codes for start and stop of breaks
    # Use marker codes that did not exist in the given marker codes...
    all_start_codes = np.concatenate(
        [np.atleast_1d(vals) for vals in name_to_start_codes.values()]
    )
    all_stop_codes = np.concatenate(
        [np.atleast_1d(vals) for vals in name_to_stop_codes.values()]
    )
    break_start_code = -1
    while break_start_code in np.concatenate((all_start_codes, all_stop_codes)):
        break_start_code -= 1
    break_stop_code = break_start_code - 1
    while break_stop_code in np.concatenate((all_start_codes, all_stop_codes)):
        break_stop_code -= 1

    events = cnt.info["events"][:, [0, 2]]
    # later trial segment ival will be added when creating set
    # so remove it here
    break_epoch_ival_ms = np.array(break_epoch_ival_ms) - (
        np.array(trial_epoch_ival_ms)
    )
    events_with_breaks = add_breaks(
        events,
        cnt.info["sfreq"],
        break_start_code,
        break_stop_code,
        name_to_start_codes,
        name_to_stop_codes,
        min_break_length_ms=min_break_length_ms,
        max_break_length_ms=max_break_length_ms,
        break_start_offset_ms=break_epoch_ival_ms[0],
        break_stop_offset_ms=break_epoch_ival_ms[1],
    )

    name_to_start_codes_with_breaks = deepcopy(name_to_start_codes)
    name_to_start_codes_with_breaks["Break"] = break_start_code
    name_to_stop_codes_with_breaks = deepcopy(name_to_stop_codes)
    name_to_stop_codes_with_breaks["Break"] = break_stop_code

    data = cnt.get_data()
    fs = cnt.info["sfreq"]
    signal_target = create_signal_target(
        data,
        events_with_breaks,
        fs,
        name_to_start_codes_with_breaks,
        trial_epoch_ival_ms,
        name_to_stop_codes_with_breaks,
        prepad_trials_to_n_samples=prepad_trials_to_n_samples,
        one_hot_labels=False,
        one_label_per_trial=False,
    )

    return signal_target


def add_breaks(
    events,
    fs,
    break_start_code,
    break_stop_code,
    name_to_start_codes,
    name_to_stop_codes,
    min_break_length_ms=None,
    max_break_length_ms=None,
    break_start_offset_ms=None,
    break_stop_offset_ms=None,
):
    """
    Add break events to given events.
    Parameters
    ----------
    events: 2d-array
        Dimensions: Number of events, 2. For each event, should contain sample
        index and marker code.
    fs: number
        Sampling rate.
    break_start_code: int
        Marker code that will be used for break start markers.
    break_stop_code: int
        Marker code that will be used for break stop markers.
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to start marker code or
        start marker codes.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
    min_break_length_ms: number, optional
        Minimum length in milliseconds a break should have to be included.
    max_break_length_ms: number, optional
        Maximum length in milliseconds a break can have to be included.
    break_start_offset_ms: number, optional
        What offset from trial end to start of the break in ms.
    break_stop_offset_ms: number, optional
        What offset from next trial start end to previous break end in ms.
    Returns
    -------
    events: 2d-array
        Events with break start and stop markers.
    """
    min_samples = (
        None if min_break_length_ms is None else ms_to_samples(min_break_length_ms, fs)
    )
    max_samples = (
        None if max_break_length_ms is None else ms_to_samples(max_break_length_ms, fs)
    )
    orig_events = events
    break_starts, break_stops = _extract_break_start_stop_ms(
        events, name_to_start_codes, name_to_stop_codes
    )

    break_durations = break_stops - break_starts
    valid_mask = np.array([True] * len(break_starts))
    if min_samples is not None:
        valid_mask[break_durations < min_samples] = False
    if max_samples is not None:
        valid_mask[break_durations > max_samples] = False
    if sum(valid_mask) == 0:
        return deepcopy(events)
    break_starts = break_starts[valid_mask]
    break_stops = break_stops[valid_mask]
    if break_start_offset_ms is not None:
        break_starts += int(round(ms_to_samples(break_start_offset_ms, fs)))
    if break_stop_offset_ms is not None:
        break_stops += int(round(ms_to_samples(break_stop_offset_ms, fs)))
    break_events = np.zeros((len(break_starts) * 2, 2))
    break_events[0::2, 0] = break_starts
    break_events[1::2, 0] = break_stops
    break_events[0::2, 1] = break_start_code
    break_events[1::2, 1] = break_stop_code

    new_events = np.concatenate((orig_events, break_events))
    # sort events
    sort_order = np.argsort(new_events[:, 0], kind="mergesort")
    new_events = new_events[sort_order]
    return new_events


def _extract_break_start_stop_ms(events, name_to_start_codes, name_to_stop_codes):
    assert len(events[0]) == 2, "expect only 2dimensional event array here"
    start_code_to_name_and_y = _to_mrk_code_to_name_and_y(name_to_start_codes)
    # Ensure all stop marker codes are iterables
    for name in name_to_stop_codes:
        codes = name_to_stop_codes[name]
        if not hasattr(codes, "__len__"):
            name_to_stop_codes[name] = [codes]
    all_stop_codes = np.concatenate(list(name_to_stop_codes.values())).astype(np.int32)
    event_samples = events[:, 0]
    event_codes = events[:, 1]

    break_starts = []
    break_stops = []
    i_event = 0
    while i_event < len(events):
        while (i_event < len(events)) and (event_codes[i_event] not in all_stop_codes):
            i_event += 1
        if i_event < len(events):
            # one sample after start
            stop_sample = event_samples[i_event]
            stop_code = event_codes[i_event]
            i_event += 1
            while (i_event < len(events)) and (
                event_codes[i_event] not in start_code_to_name_and_y
            ):
                if event_codes[i_event] in all_stop_codes:
                    log.warning(
                        "New end marker  {:.0f} at {:.0f} samples found, "
                        "no start marker for earlier end marker {:.0f} "
                        "at {:.0f} samples found.".format(
                            event_codes[i_event],
                            event_samples[i_event],
                            stop_code,
                            stop_sample,
                        )
                    )
                    stop_sample = event_samples[i_event]
                    stop_code = event_codes[i_event]
                i_event += 1

            if i_event == len(events):
                break

            start_sample = event_samples[i_event]
            start_code = event_codes[i_event]
            assert start_code in start_code_to_name_and_y
            # let's start one after stop of the trial and stop one efore
            # start of the trial to ensure that markers will be
            # in right order
            break_starts.append(stop_sample + 1)
            break_stops.append(start_sample - 1)
    return np.array(break_starts), np.array(break_stops)


class SignalAndTarget(object):
    """
    Simple data container class.
    Parameters
    ----------
    X: 3darray or list of 2darrays
        The input signal per trial.
    y: 1darray or list
        Labels for each trial.
    """

    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y


def ms_to_samples(ms, fs):
    """
    Compute milliseconds to number of samples.
    Parameters
    ----------
    ms: number
        Milliseconds
    fs: number
        Sampling rate
    Returns
    -------
    n_samples: int
        Number of samples
    """
    return ms * fs / 1000.0


def samples_to_ms(n_samples, fs):
    """
    Compute milliseconds to number of samples.
    Parameters
    ----------
    n_samples: number
        Number of samples
    fs: number
        Sampling rate
    Returns
    -------
    milliseconds: int
    """
    return n_samples * 1000.0 / fs


def apply_to_X_y(fn, *sets):
    """
    Apply a function to all `X` and `y` attributes of all given sets.
    Applies function to list of X arrays and to list of y arrays separately.
    Parameters
    ----------
    fn: function
        Function to apply
    sets: :class:`.SignalAndTarget` objects
    Returns
    -------
    result_set: :class:`.SignalAndTarget`
        Dataset with X and y as the result of the
        application of the function.
    """
    X = fn(*[s.X for s in sets])
    y = fn(*[s.y for s in sets])
    return SignalAndTarget(X, y)


def resample_cnt(cnt, new_fs):
    """
    Resample continuous recording using `resampy`.
    Parameters
    ----------
    cnt: `mne.io.RawArray`
    new_fs: float
        New sampling rate.
    Returns
    -------
    resampled: `mne.io.RawArray`
        Resampled object.
    """
    if new_fs == cnt.info["sfreq"]:
        log.info("Just copying data, no resampling, since new sampling rate same.")
        return deepcopy(cnt)
    log.warning("This is not causal, uses future data....")
    log.info("Resampling from {:f} to {:f} Hz.".format(cnt.info["sfreq"], new_fs))

    data = cnt.get_data().T

    new_data = resampy.resample(
        data, cnt.info["sfreq"], new_fs, axis=0, filter="kaiser_fast"
    ).T
    old_fs = cnt.info["sfreq"]
    new_info = deepcopy(cnt.info)
    new_info["sfreq"] = new_fs
    events = new_info["events"]
    event_samples_old = cnt.info["events"][:, 0]
    event_samples = event_samples_old * new_fs / float(old_fs)
    events[:, 0] = event_samples
    return mne.io.RawArray(new_data, new_info)


def concatenate_raws_with_events(raws):
    """
    Concatenates `mne.io.RawArray` objects, respects `info['events']` attributes
    and concatenates them correctly. Also does not modify `raws[0]` inplace
    as the :func:`concatenate_raws` function of MNE does.
    Parameters
    ----------
    raws: list of `mne.io.RawArray`
    Returns
    -------
    concatenated_raw: `mne.io.RawArray`
    """
    # prevent in-place modification of raws[0]
    raws[0] = deepcopy(raws[0])
    event_lists = [r.info["events"] for r in raws]
    new_raw, new_events = concatenate_raws(raws, events_list=event_lists)
    new_raw.info["events"] = new_events
    return new_raw


def mne_apply(func, raw, verbose="WARNING"):
    """
    Apply function to data of `mne.io.RawArray`.
    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.
    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.
    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)


def common_average_reference_cnt(
    cnt,
):
    """
    Common average reference, subtract average over electrodes at each timestep.
    Parameters
    ----------
    cnt: `mne.io.RawArray`
    Returns
    -------
    car_cnt: cnt: `mne.io.RawArray`
        Same data after common average reference.
    """
    return mne_apply(lambda a: a - np.mean(a, axis=0, keepdim=True), cnt)


def exponential_running_standardize(
    data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    """
    Perform exponential running standardization.

    Compute the exponental running mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential running variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.
    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.
    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(data[0:init_block_size], axis=other_axis, keepdims=True)
        init_std = np.std(data[0:init_block_size], axis=other_axis, keepdims=True)
        init_block_standardized = (data[0:init_block_size] - init_mean) / np.maximum(
            eps, init_std
        )
        standardized[0:init_block_size] = init_block_standardized
    return standardized


def exponential_running_demean(data, factor_new=0.001, init_block_size=None):
    """
    Perform exponential running demeanining.
    Compute the exponental running mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    Deman the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t)`.
    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Demean data before to this index with regular demeaning.
    Returns
    -------
    demeaned: 2darray (time, channels)
        Demeaned data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(data[0:init_block_size], axis=other_axis, keepdims=True)
        demeaned[0:init_block_size] = data[0:init_block_size] - init_mean
    return demeaned


def highpass_cnt(data, low_cut_hz, fs, filt_order=3, axis=0):
    """
     Highpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    fs: float
    filt_order: int
    Returns
    -------
    highpassed_data: 2d-array
        Data after applying highpass filter.
    """
    if (low_cut_hz is None) or (low_cut_hz == 0):
        log.info("Not doing any highpass, since low 0 or None")
        return data.copy()
    b, a = scipy.signal.butter(filt_order, low_cut_hz / (fs / 2.0), btype="highpass")
    assert filter_is_stable(a)
    data_highpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_highpassed


def lowpass_cnt(data, high_cut_hz, fs, filt_order=3, axis=0):
    """
     Lowpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    high_cut_hz: float
    fs: float
    filt_order: int
    Returns
    -------
    lowpassed_data: 2d-array
        Data after applying lowpass filter.
    """
    if (high_cut_hz is None) or (high_cut_hz == fs / 2.0):
        log.info("Not doing any lowpass, since high cut hz is None or nyquist freq.")
        return data.copy()
    b, a = scipy.signal.butter(filt_order, high_cut_hz / (fs / 2.0), btype="lowpass")
    assert filter_is_stable(a)
    data_lowpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_lowpassed


def bandpass_cnt(
    data, low_cut_hz, high_cut_hz, fs, filt_order=3, axis=0, filtfilt=False
):
    """
     Bandpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    high_cut_hz: float
    fs: float
    filt_order: int
    filtfilt: bool
        Whether to use filtfilt instead of lfilter
    Returns
    -------
    bandpassed_data: 2d-array
        Data after applying bandpass filter.
    """
    if (low_cut_hz == 0 or low_cut_hz is None) and (
        high_cut_hz == None or high_cut_hz == fs / 2.0
    ):
        log.info(
            "Not doing any bandpass, since low 0 or None and "
            "high None or nyquist frequency"
        )
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        log.info("Using lowpass filter since low cut hz is 0 or None")
        return lowpass_cnt(data, high_cut_hz, fs, filt_order=filt_order, axis=axis)
    if high_cut_hz == None or high_cut_hz == (fs / 2.0):
        log.info("Using highpass filter since high cut hz is None or nyquist freq")
        return highpass_cnt(data, low_cut_hz, fs, filt_order=filt_order, axis=axis)

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype="bandpass")
    assert filter_is_stable(a), "Filter should be stable..."
    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed


def filter_is_stable(a):
    """
    Check if filter coefficients of IIR filter are stable.
    Parameters
    ----------
    a: list or 1darray of number
        Denominator filter coefficients a.
    Returns
    -------
    is_stable: bool
        Filter is stable or not.
    Notes
    ----
    Filter is stable if absolute value of all  roots is smaller than 1,
    see [1]_.
    References
    ----------
    .. [1] HYRY, "SciPy 'lfilter' returns only NaNs" StackOverflow,
       http://stackoverflow.com/a/8812737/1469195
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a))
    )
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a)) < 1)
