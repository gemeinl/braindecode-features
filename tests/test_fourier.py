import mne
import numpy as np
import pytest
from sklearn.pipeline import FeatureUnion

from braindecode_features.utils import _initialize_windowing_fn
from braindecode_features.feature_extraction import (
    _build_transformer_list, _FunctionTransformer)
from braindecode_features.domains.fourier import (
    get_fourier_feature_functions, extract_fourier_features)


def test_extract_fourier_features():
    from braindecode.datasets import BaseDataset, BaseConcatDataset
    np.random.seed(20210823)
    info = mne.create_info(['O1', 'O2'], sfreq=100, ch_types='eeg')
    signals = np.random.rand(2 * 100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    concat_ds = BaseConcatDataset([ds])
    transformers = get_fourier_feature_functions()
    transformers = [_FunctionTransformer(f) for f in transformers]
    transformers = _build_transformer_list(transformers)
    fu = FeatureUnion(
        transformer_list=transformers,
    )
    windowing_fn = _initialize_windowing_fn(
        has_events=False,
        windowing_params={
            'window_size_samples': 100,
            'window_stride_samples': 100,
            'drop_last_window': False,
        }
    )
    features = extract_fourier_features(
        concat_ds=concat_ds,
        frequency_bands=[(4, 8), (8, 13)],
        fu=fu,
        windowing_fn=windowing_fn,
    )
    # 1 example, 9 univariate features, 2 frequency bands, 2 channels
    assert features.shape == (1, 9*2*2 + 3)
    expected_cols = [
        'i_trial', 'i_window_in_trial', 'target',
        'Fourier__power__O2__8-13',
    ]
    assert all([col in features.columns for col in expected_cols])

    with pytest.warns(UserWarning, match='Am supposed to pick bins between '):
        features = extract_fourier_features(
            concat_ds=concat_ds,
            frequency_bands=[(3.9, 8.03)],
            fu=fu,
            windowing_fn=windowing_fn,
        )


def test_get_fourier_feature_functions():
    fourier_feature_funcs = get_fourier_feature_functions()
    assert len(fourier_feature_funcs) == 9

    fourier_feature_funcs = get_fourier_feature_functions(
        exclude=['power', 'peak_frequency'])
    assert len(fourier_feature_funcs) == 7

    fourier_feature_funcs = get_fourier_feature_functions(
        include=['value_range'])
    assert len(fourier_feature_funcs) == 1
