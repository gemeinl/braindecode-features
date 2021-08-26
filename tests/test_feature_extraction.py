from functools import partial

import mne
import numpy as np
import pandas as pd

from braindecode_features.feature_extraction import (
    _get_feature_functions,
    _get_extraction_routines, _FunctionTransformer, _finalize_df, _merge_dfs,
    _params_to_domain_params, _build_transformer_list, extract_ds_features
)


def test_extract_ds_features():
    from braindecode.datasets import BaseDataset, BaseConcatDataset
    np.random.seed(20210823)
    channel_names = ['O1', 'O2']
    info = mne.create_info(channel_names, sfreq=100, ch_types='eeg')
    signals = np.random.rand(2*100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    concat_ds = BaseConcatDataset([ds])
    features = extract_ds_features(
        ds=concat_ds,
        frequency_bands=[(8, 13)],
        windowing_params={
            'window_size_samples': 100,
            'window_stride_samples': 100,
            'drop_last_window': False,
        },
        params={'Time__higuchi_fractal_dimension__kmax': 3},
        n_jobs=2,
    )
    assert isinstance(features, pd.DataFrame)
    assert features.ndim == 2


def test_build_transformer_list():
    from braindecode_features.feature_extraction import _FunctionTransformer
    transformers = [_FunctionTransformer(np.mean), _FunctionTransformer(np.var)]
    transformers = _build_transformer_list(transformers)
    assert len(transformers) == 2
    assert all([isinstance(k, str) for (k, v) in transformers])
    assert all([isinstance(v, _FunctionTransformer) for (k, v) in transformers])


def test_params_to_domain_params():
    params = {
        'Time__mean__axis': -1,
        'Time__median__hi': None,
        'Fourier__power__test': 'this',
    }
    expected_domain_params = {
        'Time': {'mean__kw_args': {'axis': -1},
                 'median__kw_args': {'hi': None}},
        'Fourier': {'power__kw_args': {'test': 'this'}},
    }
    domain_params = _params_to_domain_params(params)
    assert expected_domain_params == domain_params


def test_merge_dfs():
    df = pd.DataFrame([
        [0, 1, 2, .1],
        [0, 0, 0, -3],
        [0, 1, 0, 9],
    ], columns=['i_trial', 'i_window_in_trial', 'target',
                'Time__fake_feature__Ch1__0-0'])
    df2 = pd.DataFrame([
        [0, 1, 2, .5],
        [0, 0, 0, 12],
        [0, 1, 0, -.3],
    ], columns=['i_trial', 'i_window_in_trial', 'target',
                'Fourier__another_fake_feature__Ch1__4-8'])
    merged_df = _merge_dfs([df, df2], ['i_trial', 'i_window_in_trial', 'target'])
    assert merged_df.shape == (3, 5)
    assert 'Fourier__another_fake_feature__Ch1__4-8' in merged_df.columns
    assert 'Time__fake_feature__Ch1__0-0' in merged_df.columns


def test_finalize_df():
    df = pd.DataFrame([
        [0, 1, 2, .1],
        [0, 0, 0, -3],
        [0, 1, 0, 9],
    ], columns=['i_trial', 'i_window_in_trial', 'target',
                'Time__fake_feature__Ch1__0-0'])
    final_df = _finalize_df(dfs=[df])
    assert ('Description', 'Trial', '', '') in final_df.columns
    assert ('Description', 'Window', '', '') in final_df.columns
    assert ('Description', 'Target', '', '') in final_df.columns
    assert ('Time', 'fake_feature', 'Ch1', '0-0') in final_df.columns
    df2 = pd.DataFrame([
        [0, 1, 2, .5],
        [0, 0, 0, 12],
        [0, 1, 0, -.3],
    ], columns=['i_trial', 'i_window_in_trial', 'target',
                'Fourier__another_fake_feature__Ch1__4-8'])
    final_df = _finalize_df(dfs=[df, df2])
    assert ('Fourier', 'another_fake_feature', 'Ch1', '4-8') in final_df.columns


def test_FunctionTransformer():
    transformer = _FunctionTransformer(partial(np.mean, axis=-1))
    assert not hasattr(transformer, 'chs_out_')
    data = np.ones((2, 4, 100))
    features = transformer.fit_transform(data)
    assert features.shape == (2, 4)
    assert hasattr(transformer, 'chs_in_')
    assert hasattr(transformer, 'chs_out_')


def test_get_feature_functions():
    feature_functions = _get_feature_functions(domain=None)
    assert len(feature_functions) == 4
    expected_domains = ['Time', 'Fourier', 'Wavelet', 'Cross-frequency']
    assert all([d in feature_functions.keys() for d in expected_domains])

    feature_functions = _get_feature_functions(domain='Time')
    assert len(feature_functions) == 1
    assert 'Time' in feature_functions.keys()


def test_get_extraction_routines():
    extraction_routines = _get_extraction_routines(domain=None)
    assert len(extraction_routines) == 4
    expected_domains = ['Time', 'Fourier', 'Wavelet', 'Cross-frequency']
    assert all([d in extraction_routines.keys() for d in expected_domains])

    extraction_routines = _get_extraction_routines(domain='Time')
    assert len(extraction_routines) == 1
    assert 'Time' in extraction_routines.keys()
