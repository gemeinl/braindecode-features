import os
import tempfile
from functools import partial

import mne
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion

from braindecode_features.feature_extraction import (
    _build_transformer_list, _FunctionTransformer)
from braindecode_features.utils import (
    filter_df, drop_window, add_description, _window, _read_and_aggregate,
    _initialize_windowing_fn, _get_unfiltered_chs, _generate_feature_names,
    _find_col, _filter_and_window, _filter, _concat_ds_and_window,
    _check_df_consistency, _aggregate_windows, _select_funcs)


def test_check_df_consistency():
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
    # check df is fine
    _check_df_consistency(d)
    f_name = ('D', 'F', 'C', 'Fr')
    f = pd.Series([0.33, 1.1, -.4], name=f_name)
    df = pd.concat([d, f], axis=1)
    _check_df_consistency(df)

    # check df fails with None value
    f = pd.Series([0.33, None, -.4], name=f_name)
    df = pd.concat([d, f], axis=1)
    with pytest.raises(AssertionError):
        _check_df_consistency(df)

    # check df fails with NaN value
    f = pd.Series([0.33, 1.1, np.nan], name=f_name)
    df = pd.concat([d, f], axis=1)
    with pytest.raises(AssertionError):
        _check_df_consistency(df)

    # check df fails with inf value
    f = pd.Series([np.inf, 1.1, -.4], name=f_name)
    df = pd.concat([d, f], axis=1)
    with pytest.raises(AssertionError):
        _check_df_consistency(df)


def test_concat_ds_and_window():
    from braindecode.datasets import BaseDataset
    np.random.seed(20210823)
    channel_names = ['O1', 'O2']
    info = mne.create_info(channel_names, sfreq=100)
    band_channels = ['O1']
    signals = np.random.rand(2*100).reshape(2, 100)
    new_signals = np.random.rand(2*100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    concat_windows_ds = _concat_ds_and_window(
        ds=ds,
        data=new_signals,
        windowing_fn=_initialize_windowing_fn(
            has_events=False,
            windowing_params={
                'window_size_samples': 100,
                'window_stride_samples': 100,
                'drop_last_window': False,
            },
        ),
        band_channels=band_channels,
    )
    assert len(concat_windows_ds) == 1
    for x, y, ind in concat_windows_ds:
        assert x.shape == (1, 100)
        ids = [channel_names.index(ch) for ch in band_channels]
        np.testing.assert_allclose(
            new_signals[ids], x, rtol=1e-6, atol=1e-6)
        break


def test_filter():
    from braindecode.datasets import BaseDataset, BaseConcatDataset
    np.random.seed(20210823)
    channel_names = ['O1', 'O2']
    info = mne.create_info(channel_names, sfreq=100, ch_types='eeg')
    signals = np.random.rand(2 * 100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    concat_ds = BaseConcatDataset([ds])
    filtered_ds = _filter(concat_ds, frequency_bands=[(0, 4), (4, 8)])
    assert filtered_ds.datasets[0].raw.ch_names == [
        'O1', 'O1_0-4', 'O1_4-8', 'O2', 'O2_0-4', 'O2_4-8']


def test_filter_and_window():
    from braindecode.datasets import BaseDataset, BaseConcatDataset
    np.random.seed(20210823)
    channel_names = ['O1', 'O2']
    info = mne.create_info(channel_names, sfreq=100, ch_types='eeg')
    signals = np.random.rand(2 * 100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    concat_ds = BaseConcatDataset([ds])
    concat_windows_ds = _filter_and_window(
        ds=concat_ds,
        frequency_bands=[(0, 4), (4, 8)],
        windowing_fn=_initialize_windowing_fn(
            has_events=False,
            windowing_params={
                'window_size_samples': 100,
                'window_stride_samples': 100,
                'drop_last_window': False,
            },
        ),
    )
    assert hasattr(concat_windows_ds.datasets[0], 'windows')
    assert concat_windows_ds.datasets[0].windows.ch_names == [
        'O1', 'O1_0-4', 'O1_4-8', 'O2', 'O2_0-4', 'O2_4-8']


""""
def test_filter_df():
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
    df1 = pd.concat([d, f], axis=1)
    f_name = ('D2', 'F2', 'C2', 'Fr2')
    f = pd.Series([0.33, 1.1, -.4], name=f_name)
    df2 = pd.concat([df1, f], axis=1)
    filtered_df = filter_df(
        df=df2,
        query='D',
        exact_match=False,
        level_to_consider=None,
    )
    assert pd.testing.assert_frame_equal(df2, filtered_df)
    filtered_df = filter_df(
        df=df2,
        query='D',
        exact_match=True,
        level_to_consider=None,
    )
    assert pd.testing.assert_frame_equal(df1, filtered_df)
    filtered_df = filter_df(
        df=df2,
        query='D',
        exact_match=False,
        level_to_consider=0,
    )
    assert pd.testing.assert_frame_equal(df1, filtered_df)
    filtered_df = filter_df(
        df=df2,
        query=['D'],
        exact_match=False,
        level_to_consider=0,
    )
    assert pd.testing.assert_frame_equal(df1, filtered_df)
"""


def test_generate_feature_names():
    ch_names = ['O1', 'O2']
    transformers = [np.mean, np.var]
    transformers = [_FunctionTransformer(t) for t in transformers]
    transformers = _build_transformer_list(transformers)
    fu = FeatureUnion(transformer_list=transformers)
    fu.set_params(
        **{'mean__kw_args': {'axis': -1}, 'var__kw_args': {'axis': -1}})
    with pytest.raises(AssertionError, match='Call transform on your data fir'):
        names = _generate_feature_names(
            fu=fu,
            ch_names=ch_names,
        )
    features = fu.fit_transform(np.ones((1, 2, 100)))
    names = _generate_feature_names(
        fu=fu,
        ch_names=ch_names,
    )
    assert names == ['mean__O1', 'mean__O2', 'var__O1', 'var__O2']
    # domains are added by extraction routines
    # frequency bands are added by extraction routines
    # the feature functions and the feature union have no knowledge about them


def test_get_unfiltered_chs():
    from braindecode.datasets import BaseDataset
    np.random.seed(20210823)
    info = mne.create_info(['O1', 'O1_4-8'], sfreq=100)
    signals = np.random.rand(2*100).reshape(2, 100)
    new_signals = np.random.rand(2*100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    unfiltered_chs = _get_unfiltered_chs(
        ds=ds,
        frequency_bands=[(4, 8)],
    )
    assert unfiltered_chs == ['O1']


def test_initialize_windowing_fn():
    windowing_fn = _initialize_windowing_fn(
        has_events=False,
        windowing_params={
            'window_size_samples': 100,
            'window_stride_samples': 100,
        }
    )
    assert windowing_fn.func.__name__ == 'create_fixed_length_windows'
    assert windowing_fn.keywords['window_size_samples'] == 100
    assert windowing_fn.keywords['window_stride_samples'] == 100

    windowing_fn = _initialize_windowing_fn(
        has_events=True,
        windowing_params={},
    )
    assert windowing_fn.func.__name__ == 'create_windows_from_events'


def test_read_and_aggregate():
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
    df1 = pd.concat([d, f], axis=1)
    path = tempfile.mkdtemp()
    file_path = os.path.join(path, 'test.hdf')
    df1.to_hdf(file_path, 'data')
    read_df = _read_and_aggregate(
        path=file_path,
        agg_func=None,
    )
    pd.testing.assert_frame_equal(df1, read_df)


def test_window():
    from braindecode.datasets import BaseDataset, BaseConcatDataset
    np.random.seed(20210823)
    info = mne.create_info(['O1', 'O2'], sfreq=100)
    signals = np.random.rand(2 * 100).reshape(2, 100)
    raw = mne.io.RawArray(signals, info)
    ds = BaseDataset(raw=raw)
    concat_ds = BaseConcatDataset([ds])
    windows_ds = _window(
        ds=concat_ds,
        windowing_fn=_initialize_windowing_fn(
            has_events=False,
            windowing_params={
                'window_size_samples': 100,
                'window_stride_samples': 100,
                'drop_last_window': False,
            },
        )
    )
    assert hasattr(windows_ds.datasets[0], 'windows')
    for x, y, ind in windows_ds:
        assert x.shape == (2, 100)
        break


def test_filter_df():
    columns = [
        ('Description', 'Trial', '', ''),
        ('Description', 'Window', '', ''),
        ('Description', 'Target', '', ''),
        ('Time', 'line_length', '4-8', 'O1'),
        ('Phase', 'phase_locking_value', '8-13', 'O1-P5'),
    ]
    df = pd.DataFrame(
        data=[
            [0, 4, 0, .2, .89],
            [0, 4, 1, .4, .73],
            [1, 2, 0, 13.2, .08],
        ],
        columns=pd.MultiIndex.from_tuples(columns),
    )
    pd.testing.assert_frame_equal(filter_df(df, '8'), df)
    pd.testing.assert_frame_equal(filter_df(df, 'l'), df)
    pd.testing.assert_frame_equal(filter_df(df, 'h'), df)
    pd.testing.assert_frame_equal(
        filter_df(df, 'h', level_to_consider=0),
        df[[_find_col(df, c) for c in ['Trial', 'Window', 'Target', 'Phase']]])


def test_find_col():
    columns = pd.MultiIndex.from_tuples([
        ('Test', '', '4'),
        ('Another', 'Test', ''),
    ],
        names=['C1', 'C2', 'C3']
    )
    found_col = _find_col(columns, 'Another')
    assert found_col == ('Another', 'Test', '')
    with pytest.raises(AssertionError):
        _find_col(columns, 'other')
    with pytest.raises(AssertionError):
        _find_col(columns, 'Test')


def test_aggregate_windows():
    columns = ['Trial', 'Window', 'Target', 'data']
    df = pd.DataFrame(
        data=[
            [0, 0, 4, .2],
            [0, 1, 4, .4],
            [1, 0, 2, 13.2],
        ],
        columns=columns,
    )    
    expected_df = pd.DataFrame(
        data=[
            [0, 4, .3],
            [1, 2, 13.2],
        ],
        columns=['Trial', 'Target', 'data'],
    )
    avg_df = _aggregate_windows(df, 'mean')
    pd.testing.assert_frame_equal(avg_df, expected_df)

    
def test_add_description():
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
    df = add_description(
        df=df,
        description=[-1, -1, -1],
        name='Test',
    )
    expected_df = pd.DataFrame(
        data=[
            [-1, 0, 0, 4, .2],
            [-1, 0, 1, 4, .4],
            [-1, 1, 0, 2, 13.2],
        ],
        columns=pd.MultiIndex.from_tuples(
            [('Description', 'Test', '', '')] + columns),
    )
    pd.testing.assert_frame_equal(expected_df, df)


def test_drop_window():
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
    df = drop_window(
        df=df,
        window_i=0,
    )
    expected_df = pd.DataFrame(
        data=[
            [0, 0, 4, .4],
        ],
        columns=columns,
    )
    pd.testing.assert_frame_equal(expected_df, df)


def test_select_funcs():
    funcs = [np.mean, np.median, np.var]
    selected_funcs = _select_funcs(funcs)
    assert len(selected_funcs) == len(funcs)
    assert funcs == selected_funcs

    selected_funcs = _select_funcs(funcs, include=['var', 'mean'])
    assert len(selected_funcs) == 2
    assert selected_funcs[0].__name__ == 'mean'
    assert selected_funcs[1].__name__ == 'var'

    selected_funcs = _select_funcs(funcs, exclude=['mean', 'median'])
    assert len(selected_funcs) == 1
    assert selected_funcs[0].__name__ == 'var'

    with pytest.raises(ValueError, match='Can either include or exclude speci'):
        _ = _select_funcs(funcs, include=['mean'], exclude=['median'])

    with pytest.raises(ValueError, match='You specified unknown functions'):
        _ = _select_funcs(funcs, include=['test'])

    selected_funcs = _select_funcs(funcs, include='median')
    assert len(selected_funcs) == 1
    assert selected_funcs[0].__name__ == 'median'
