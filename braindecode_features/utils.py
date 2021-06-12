import re

import numpy as np
import pandas as pd


def generate_feature_names(fu, ch_names):
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
    return df


def _get_unfiltered_chs(windows_ds, frequency_bands):
    orig_chs = []
    for ch in windows_ds.windows.ch_names:
        if any([ch.endswith('-'.join([str(low), str(high)])) for (low, high) in frequency_bands]):
            continue
        else:
            orig_chs.append(ch)
    return orig_chs


def _aggregate_windows(df, agg_func):
    trial_col = _find_col(df.columns, 'Trial')
    grouped = df.groupby(trial_col)
    df = grouped.agg(agg_func)
    df.reset_index(inplace=True)
    # set window id to zero when aggregating, will be dropped later
    # keep it at this point for compatibility
    df[_find_col(df.columns, 'Window')] = len(df) * [0]
    # agg changes dtype of target, force it to keep original dtype
    cols = [_find_col(df, col) for col in ['Trial', 'Target', 'Window']]
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
    found_col = [c for c in columns if hint in c]
    assert len(found_col) == 1, (
        f'Please be more precise, found: {found_col}')
    return found_col[0]