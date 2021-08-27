import numpy as np
import pytest
from sklearn.pipeline import FeatureUnion

from antropy import app_entropy, lziv_complexity, perm_entropy, svd_entropy
from nolds import corr_dim, dfa, hurst_rs, lyap_r

from braindecode_features.utils import _initialize_windowing_fn
from braindecode_features.feature_extraction import (
    _build_transformer_list, _FunctionTransformer)
from braindecode_features.domains.time import (
    extract_time_features, get_time_feature_functions)

from .utils import create_fake_concat_ds


@pytest.fixture
def get_func_and_fake_data():
    def _get_func_and_fake_data(func_name):
        [f] = get_time_feature_functions(include=func_name)
        np.random.seed(20210827)
        data = np.random.rand(2 * 3 * 100).reshape(2, 3, 100)
        return f, data
    return _get_func_and_fake_data


def test_extract_time_features():
    concat_ds = create_fake_concat_ds()
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


def test_approximate_entropy(get_func_and_fake_data):
    f, data = get_func_and_fake_data('approximate_entropy')
    # 3d implementation (loop and apply along axis)
    feats = f(data)
    # 1d calls to original function
    expected_feats = np.array([app_entropy(ch) for d in data for ch in d])
    expected_feats = expected_feats.reshape(2, -1)
    np.testing.assert_allclose(feats, expected_feats, atol=1e-6, rtol=1e-6)


def test_correlation_dimension(get_func_and_fake_data):
    f, data = get_func_and_fake_data('correlation_dimension')
    # 3d implementation (loop and apply along axis)
    feats = f(data)
    # 1d calls to original function
    expected_feats = np.array(
        [corr_dim(ch, emb_dim=len(ch)//10) for d in data for ch in d])
    expected_feats = expected_feats.reshape(2, -1)
    np.testing.assert_allclose(feats, expected_feats, atol=1e-6, rtol=1e-6)


def test_detrended_fluctuation_analysis(get_func_and_fake_data):
    f, data = get_func_and_fake_data('detrended_fluctuation_analysis')
    # 3d implementation (loop and apply along axis)
    feats = f(data)
    # 1d calls to original function
    expected_feats = np.array(
        [dfa(ch) for d in data for ch in d])
    expected_feats = expected_feats.reshape(2, -1)
    np.testing.assert_allclose(feats, expected_feats, atol=1e-6, rtol=1e-6)


def test_higuchi_fractal_dimension(get_func_and_fake_data):
    try:
        from pyeeg import higuchi_fd
    except ModuleNotFoundError:
        pass
    else:
        f, data = get_func_and_fake_data('higuchi_fractal_dimension')
        # 3d implementation (loop and apply along axis)
        feats = f(data)
        # 1d calls to original function
        expected_feats = np.array(
            [higuchi_fd(ch) for d in data for ch in d])
        expected_feats = expected_feats.reshape(2, -1)
        np.testing.assert_allclose(feats, expected_feats, atol=1e-6, rtol=1e-6)


def test_hurst_exponent(get_func_and_fake_data):
    f, data = get_func_and_fake_data('hurst_exponent')
    # 3d implementation (loop and apply along axis)
    feats = f(data)
    # 1d calls to original function
    expected_feats = np.array(
        [hurst_rs(ch, fit='poly') for d in data for ch in d])
    expected_feats = expected_feats.reshape(2, -1)
    np.testing.assert_allclose(feats, expected_feats, atol=1e-6, rtol=1e-6)


def test_lyauponov_exponent(get_func_and_fake_data):
    f, data = get_func_and_fake_data('lyauponov_exponent')
    # 3d implementation (loop and apply along axis)
    feats = f(data)
    # 1d calls to original function
    expected_feats = np.array(
        [lyap_r(ch, fit='poly') for d in data for ch in d])
    expected_feats = expected_feats.reshape(2, -1)
    np.testing.assert_allclose(feats, expected_feats, atol=1e-6, rtol=1e-6)


def test_lziv_complexity(get_func_and_fake_data):
    f, data = get_func_and_fake_data('lziv_complexity')
    # 3d implementation (loop and apply along axis)
    feats = f(data)
    # 1d calls to original function
    expected_feats = np.array(
        [lziv_complexity(ch) for d in data for ch in d])
    expected_feats = expected_feats.reshape(2, -1)
    np.testing.assert_allclose(feats, expected_feats, atol=1e-6, rtol=1e-6)


def test_permutation_entropy(get_func_and_fake_data):
    f, data = get_func_and_fake_data('permutation_entropy')
    # 3d implementation (loop and apply along axis)
    feats = f(data)
    # 1d calls to original function
    expected_feats = np.array(
        [perm_entropy(ch) for d in data for ch in d])
    expected_feats = expected_feats.reshape(2, -1)
    np.testing.assert_allclose(feats, expected_feats, atol=1e-6, rtol=1e-6)


def test_svd_entropy(get_func_and_fake_data):
    f, data = get_func_and_fake_data('svd_entropy')
    # 3d implementation (loop and apply along axis)
    feats = f(data)
    # 1d calls to original function
    expected_feats = np.array(
        [svd_entropy(ch) for d in data for ch in d])
    expected_feats = expected_feats.reshape(2, -1)
    np.testing.assert_allclose(feats, expected_feats, atol=1e-6, rtol=1e-6)
