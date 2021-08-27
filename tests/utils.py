import mne
import numpy as np
import pandas as pd

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
    df = pd.DataFrame(
        data=[
            [4, 0, 0, .2, .89],
            [4, 0, 1, .4, .73],
            [2, 1, 0, 13.2, .08],
            [2, 1, 1, 13.2, .08],
        ],
        columns=pd.MultiIndex.from_tuples([
            ('Description', 'Target', '', ''),
            ('Description', 'Trial', '', ''),
            ('Description', 'Window', '', ''),
            ('Time', 'line_length', '4-8', 'O1'),
            ('Phase', 'phase_locking_value', '8-13', 'O1-P5'),
        ], names=['Domain', 'Feature', 'Channel', 'Frequency']),
    )
    return df


def _df_n_series():
    feature_col = ('Domain', 'Feature', 'Channel', 'Frequency_band')
    multiindex = pd.MultiIndex.from_tuples([
        ('Description', 'Trial', '', ''),
        ('Description', 'Window', '', ''),
        ('Description', 'Target', '', ''),
    ], names=feature_col)
    d = pd.DataFrame([
        [0, 1, 2],
        [0, 0, 0],
        [0, 1, 0],
    ], columns=multiindex)
    f_name = ('D', 'F', 'C', 'Fr')
    f = pd.Series([0.33, 1.1, -.4], name=f_name)
    return d, f


def _df():
    columns = [
        ('Description', 'Trial', '', ''),
        ('Description', 'Window', '', ''),
        ('Description', 'Target', '', ''),
        ('data', '', '', ''),
    ]
    df = pd.DataFrame(
        data=[
            [0, 0, 4, .2],
            [0, 1, 4, .4],
            [1, 0, 2, 13.2],
        ],
        columns=columns,
    )
    return df
