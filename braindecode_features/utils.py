import re
import logging
from functools import partial

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

        
def filter_df(df, query, exact_match=False, level_to_consider=None):
    """Filter the MultiIndex of a DataFrame wrt 'query'. Thereby, columns required
    for decoding, i.e., 'Target', 'Trial', 'Window' are always preserved.

    Parameters
    ----------
    df: `pd.DataFrame`
        A DataFrame to be filtered.
    query: str
        The query to look for in the columns of the DataFrame.
    exact_match: bool
        Whether the query has to be matched exactly or just be contained.
    level_to_consider: int
        Limit the filtering to look at a specific level of the MultiIndex.

    Returns
    -------
    df: `pd.DataFrame`
        The DataFrame limited to the columns matching the query.
    """
    masks = []
    assert not isinstance(level_to_consider, str), (
        'Has to be int, or list of int')
    if level_to_consider is None:
        levels = range(df.columns.nlevels)
    elif hasattr(level_to_consider, '__iter__'):
        levels = level_to_consider
    else:
        levels = [level_to_consider]
    # if available add target, trial and window to selection 
    info_cols = []
    for info_col in ['Trial', 'Window', 'Target']:
        info_col = _find_col(df.columns, info_col)
        info_cols.append(info_col)
    # go through all the levels in the multiindex and check whether the query
    # matches any value exactly or just contains it
    for i in levels:
        if exact_match:
            mask = query == df.columns[len(info_cols):].get_level_values(i)
        else:
            mask = df.columns[len(info_cols):].get_level_values(i).str.contains(
                query)
        masks.append(mask)
    mask = np.sum(masks, axis=0) > 0

    multiindex = pd.MultiIndex.from_tuples(
        info_cols+df.columns[len(info_cols):][mask].to_list())
    return df[multiindex]


def add_description(df, description, name):
    """Add a custom column with 'name' to the description section of the DataFrame.
    
    Parameters
    ----------
    df: `pd.DatFrame`
        The DataFrame to add the description to.
    description: array-like
        Data of same length as DataFrame.
    name: str
        The name of data to be added.
        
    Returns
    -------
    `pd.DataFrame`
        A DataFrame that includes the column ('Description', name, '', '').
    """
    df_ = pd.DataFrame(
        description, 
        columns=pd.MultiIndex.from_tuples([('Description', name, '', '')])
    )
    return pd.concat([df_, df], axis=1)


def drop_window(df, window_i):
    """Drop all rows of window_i in the feature DataFrame.
    
    Parameters
    ----------
    df: `pd.DataFrame`
        The feature dataframe.
    window_i: int
        The id of the window to be dropped.
    
    Returns
    -------
    df: `pd.DataFrame`
        The feature dataframe, where all rows of window_i where 
        dropped and remaining windows were re-indexed.
    """
    # select all windows that are not window_i
    window_col = _find_col(df, 'Window')
    df = df[df[window_col] != window_i]
    windows = df.pop(window_col)
    # TODO: is it OK to do this?
    # reindex the windows
    windows -= windows.values > window_i
    # insert the updated windows again
    df.insert(1, window_col, windows)
    # reset the index
    df.reset_index(drop=True, inplace=True)
    return df


def _generate_feature_names(fu, ch_names):
    """From the feature names returned by the feature functions through the feature union,
    replace the unknown channels indicated by ids (ch0 or ch0-ch13) with their actual names.
    
    Parameters
    ----------
    fu: FeatureUnion
        Scikit-learn FeatureUnion of FunctionTransformers extracting features.
    ch_names: list
        List of original channel names that will be inserted into the feature names
        returned by the union.
    
    Returns
    -------
    feature_names: list
        List of feature names including channel(s), frequency band(s), feature domain 
        and feature type.
    """
    feature_names = fu.get_feature_names()
    mapping = {f'ch{i}': ch for i, ch in enumerate(ch_names)}
    # loop below taken from mne-features
    for pattern, translation in mapping.items():
        r = re.compile(rf'{pattern}(?=_)|{pattern}\b')
        feature_names = [
            r.sub(string=feature_name, repl=translation)
            for feature_name in feature_names]
    return feature_names


def _get_unfiltered_chs(ds, frequency_bands):
    windows_or_raw = 'windows' if hasattr(ds, 'windows') else 'raw'
    orig_chs = []
    for ch in getattr(ds, windows_or_raw).ch_names:
        if any([ch.endswith('-'.join([str(low), str(high)])) 
                for (low, high) in frequency_bands]):
            continue
        else:
            orig_chs.append(ch)
    return orig_chs


def _aggregate_windows(df, agg_func):
    trial_col = _find_col(df.columns, 'Trial')
    grouped = df.groupby(trial_col)
    df = grouped.agg(agg_func)
    df.reset_index(inplace=True)
    # drop the window column
    if _find_col(df.columns, 'Window'):
        df.drop(_find_col(df.columns, 'Window'), axis=1, inplace=True)
        log.debug('aggregated features over windows. dropped window column')
    # agg changes dtype of target, force it to keep original dtype
    cols = [_find_col(df, col) for col in ['Trial', 'Target']]
    df[cols] = df[cols].astype(np.int64)
    return df


def _read_and_aggregate(path, agg_func):
    df = pd.read_hdf(path, key='data')
    if agg_func is not None:
        df = _aggregate_windows(
            df=df,
            agg_func=agg_func,
        )
    return df


def _find_col(columns, hint):
    """Find a column containing 'hint' in the column name."""
    found_col = [c for c in columns if hint in c]
    assert len(found_col) == 1, (
        f'Please be more precise, found: {found_col}')
    return found_col[0]


def _filter_and_window(ds, frequency_bands, windowing_fn):
    return _window(
        ds=_filter(
            ds=ds,
            frequency_bands=frequency_bands,
        ),
        windowing_fn=windowing_fn,
    )


def _filter(ds, frequency_bands):
    """Filter signals in a BaseConcatDataset of BaseDataset in time domain to given
    frequency ranges."""
    from braindecode.preprocessing import Preprocessor, preprocess, filterbank
    # check whether filtered signals already exist
    frequency_bands_str = ['-'.join([str(b[0]), str(b[1])]) for b in frequency_bands]
    all_band_channels = []
    for frequency_band in frequency_bands_str:
        # pick all channels corresponding to a single frequency band
        all_band_channels.append([ch for ch in ds.datasets[0].raw.ch_names 
                                  if ch.endswith(frequency_band)])
    # If we cannot find all the bands in the channel names annotations, we have to filter.
    if not all(all_band_channels):
        log.debug('Filtering ...')
        preprocess(
            concat_ds=ds,
            preprocessors=[
                Preprocessor(
                    apply_on_array=False,
                    fn=filterbank, 
                    frequency_bands=sorted(frequency_bands, key=lambda b: b[0]), 
                    drop_original_signals=False, 
                )
            ],
        )
    return ds

        
def _window(ds, windowing_fn):
    """Cut braindecode compute windows."""
    log.debug('Windowing ...')
    ds = windowing_fn(
        concat_ds=ds,
    )
    log.debug(f'got {len(ds)} windows')
    return ds


def _initialize_windowing_fn(has_events, windowing_params):
    """Set windowing params to the appropriate windowing function."""
    if windowing_params is None:
        windowing_params = {}
    from braindecode.preprocessing import (
        create_windows_from_events, create_fixed_length_windows)
    if has_events:
        windowing_fn = partial(
            create_windows_from_events,
            **windowing_params,
        )
    else:
        windowing_fn = partial(
            create_fixed_length_windows,
            **windowing_params,
        )
    return windowing_fn


def _concat_ds_and_window(ds, data, windowing_fn, band_channels):
    """Overwrite data in ds.raw with domain transformed data and cut windows."""
    from braindecode.datasets.base import BaseDataset, BaseConcatDataset
    raw = ds.raw.copy()
    raw = raw.pick_channels(band_channels)
    raw._data = data
    target_name = ds.target_name if isinstance(ds.target_name, str) else tuple(ds.target_name)
    concat_ds = BaseConcatDataset([
        BaseDataset(
            raw=raw, 
            description=ds.description,
            target_name=target_name,
        )
    ])
    return _window(
        ds=concat_ds,
        windowing_fn=windowing_fn,
    )


def _check_df_consistency(df):
    """Make sure the feature DataFrames do not contain illegal values like 
    +/- inf, None or NaN."""
    # TODO: their might be a tuple inside the target column which will 
    # TODO: cause checks below to fail
    feature_cols = df.columns[3:]  # TODO: do not hardcode this
    assert not pd.isna(df[feature_cols].values).any()
    assert not pd.isnull(df[feature_cols].values).any()
    assert np.abs(df[feature_cols].values).max() < np.inf


from braindecode.datasets import BaseDataset
class FeatureDataset(BaseDataset):
    """
    """
    def __init__(self, feature_df, description=None, target_name=None):
        super().__init__(
            raw=None, 
            description=description,
            target_name=target_name,
            transform=None,
        )
        self.names = feature_df.columns.to_list()
        feature_cols = [col for col in self.names if 'Description' not in col]
        ind_cols = [col for col in self.names if 'Description' in col]
        ind = feature_df[ind_cols]
        self.y = ind[[col for col in ind.columns if 'Target' in col]].to_numpy()
        self.ind_dtypes = ind.dtypes.values
        self.ind = ind[[col for col in ind.columns if 'Target' not in col]].to_numpy()
        self.x = feature_df[feature_cols].to_numpy()
    
    @property
    def feature_df(self):
        df = pd.DataFrame(
            np.concatenate([self.ind, self.y, self.x], axis=1), 
            columns=self.names)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        dtype_map = {df.columns[i]: self.ind_dtypes[i] 
                     for i in range(len(self.ind_dtypes))}
        df = df.astype(dtype_map)
        return df

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.ind[index]
        
    def __len__(self):
        return len(self.x)
