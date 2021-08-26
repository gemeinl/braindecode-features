# TODO: add tests with np.apply_along_axis

import mne
import numpy as np
from sklearn.pipeline import FeatureUnion

from braindecode_features.utils import _initialize_windowing_fn
from braindecode_features.feature_extraction import (
    _build_transformer_list, _FunctionTransformer)
from braindecode_features.domains.time import (
    extract_time_features, get_time_feature_functions)


def test_extract_time_features():
    from braindecode.datasets import BaseDataset, BaseConcatDataset
    np.random.seed(20210823)
    info = mne.create_info(['O1', 'O2'], sfreq=100, ch_types='eeg')
    signals = np.random.rand(2 * 100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    concat_ds = BaseConcatDataset([ds])
    transformers = get_time_feature_functions(
        exclude=['higuchi_fractal_dimension'])
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
    features = extract_time_features(
        concat_ds=concat_ds,
        frequency_bands=[(4, 8), (8, 13)],
        fu=fu,
        windowing_fn=windowing_fn,
    )
    # 1 example, 27 univariate features, 2 frequency bands, 2 channels
    # 2 bivariate features (covariance, phase_locking_value)
    assert features.shape == (1, 27*2*2 + 2*2 + 3)
    expected_cols = [
        'i_trial', 'i_window_in_trial', 'target',
        'Time__phase_locking_value__O1-O2__4-8',
        'Time__covariance__O1-O2__8-13',
    ]
    assert all([col in features.columns for col in expected_cols])


def test_get_time_feature_functions():
    time_feature_functions = get_time_feature_functions()
    assert len(time_feature_functions) == 30

    time_feature_functions = get_time_feature_functions(exclude=['mean'])
    assert len(time_feature_functions) == 29

    time_feature_functions = get_time_feature_functions(
        include=['katz_fractal_dimension', 'line_length'])
    assert len(time_feature_functions) == 2
    assert time_feature_functions[0].__name__ == 'katz_fractal_dimension'
    assert time_feature_functions[1].__name__ == 'line_length'

    # test order
    time_feature_functions = get_time_feature_functions(
        include=['line_length', 'katz_fractal_dimension'])
    assert len(time_feature_functions) == 2
    assert time_feature_functions[0].__name__ == 'katz_fractal_dimension'
    assert time_feature_functions[1].__name__ == 'line_length'
