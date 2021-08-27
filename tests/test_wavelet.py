from sklearn.pipeline import FeatureUnion

from braindecode_features.utils import _initialize_windowing_fn
from braindecode_features.feature_extraction import (
    _build_transformer_list, _FunctionTransformer)
from braindecode_features.domains.wavelet import (
    get_wavelet_feature_functions, extract_wavelet_features)

from .utils import create_fake_concat_ds


def test_extract_wavelet_features():
    concat_ds = create_fake_concat_ds()
    transformers = get_wavelet_feature_functions()
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
    features = extract_wavelet_features(
        concat_ds=concat_ds,
        frequency_bands=[(4, 8), (8, 13)],
        fu=fu,
        windowing_fn=windowing_fn,
    )
    # 1 example, 9 univariate features, 2 frequency bands, 2 channels
    assert features.shape == (1, 9*2*2 + 3)
    expected_cols = [
        'i_trial', 'i_window_in_trial', 'target',
        'Wavelet__variance__O2__4-8',
    ]
    assert all([col in features.columns for col in expected_cols])


def test_get_wavelet_feature_functions():
    wavelet_feature_funcs = get_wavelet_feature_functions()
    assert len(wavelet_feature_funcs) == 9

    wavelet_feature_funcs = get_wavelet_feature_functions(
        exclude=['variance', 'standard_deviation', 'minimum'])
    assert len(wavelet_feature_funcs) == 6

    wavelet_feature_funcs = get_wavelet_feature_functions(
        include=['maximum', 'bounded_variation'])
    assert len(wavelet_feature_funcs) == 2
    assert wavelet_feature_funcs[0].__name__ == 'bounded_variation'
    assert wavelet_feature_funcs[1].__name__ == 'maximum'
