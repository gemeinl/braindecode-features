import mne
import numpy as np
from sklearn.pipeline import FeatureUnion

from braindecode_features.feature_extraction import (
    _build_transformer_list, _FunctionTransformer)
from braindecode_features.utils import _initialize_windowing_fn
from braindecode_features.domains.cross_frequency import (
    get_cross_frequency_feature_functions, extract_cross_frequency_features)


def test_extract_cross_frequency_features():
    from braindecode.datasets import BaseDataset, BaseConcatDataset
    np.random.seed(20210823)
    info = mne.create_info(['O1', 'O2'], sfreq=100, ch_types='eeg')
    signals = np.random.rand(2 * 100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    concat_ds = BaseConcatDataset([ds])
    transformers = get_cross_frequency_feature_functions()
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
    features = extract_cross_frequency_features(
        concat_ds=concat_ds,
        frequency_bands=[(4, 8), (8, 13)],
        fu=fu,
        windowing_fn=windowing_fn,
    )
    assert features.shape == (1, 5)
    expected_cols = [
        'i_trial', 'i_window_in_trial', 'target',
        'Cross-frequency__cross_frequency_coupling__O1__4-8, 8-13',
        'Cross-frequency__cross_frequency_coupling__O2__4-8, 8-13',
    ]
    assert all([col in features.columns for col in expected_cols])


def test_get_cross_frequency_features_functions():
    cross_frequency_feature_funcs = get_cross_frequency_feature_functions()
    assert len(cross_frequency_feature_funcs) == 1

    cross_frequency_feature_funcs = get_cross_frequency_feature_functions(
        exclude=['cross_frequency_coupling'])
    assert not cross_frequency_feature_funcs
