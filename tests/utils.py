import mne
import numpy as np

from braindecode.datasets import BaseDataset, BaseConcatDataset


def create_fake_concat_ds():
    ds = create_fake_base_ds()
    concat_ds = BaseConcatDataset([ds])
    return concat_ds


def create_fake_base_ds():
    np.random.seed(20210823)
    info = mne.create_info(['O1', 'O2'], sfreq=100, ch_types='eeg')
    signals = np.random.rand(2 * 100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    return ds


def create_fake_feature_df():
    pass
