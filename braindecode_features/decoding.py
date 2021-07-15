import logging

import pandas as pd
import numpy as np

from .utils import _find_col, _aggregate_windows


log = logging.getLogger(__name__)


def prepare_features(df, agg_func=None, windows_as_examples=True):
    """Prepare a feature DataFrame for decoding, i.e., generate X and y, groups,
    and feature names. Thereby, optionally aggregates features by trials.
    Compute windows can either be used as independent examples, or to be
    appended to the feature vector.

    Parameters
    ----------
    df: `pd.DataFrame`
        The feature DataFrame as returned by `extract_windows_ds_features` and
        `read_features`.
    agg_func: None | str
        Aggregation function supported by `pd.DataFrame.agg()` used to
        optionally aggregate features by trials.        
    windows_as_examples: bool
        Whether to move the window dimension to examples or features. Without 
        effect if 'agg_func' is not None.
        
    Returns
    -------
    X: `np.ndarray`
        The feature matrix (n_examples x n_feautures).
    y: `np.ndarray`
        The targets (n_examples).
    groups: `np.ndarray`
        For every example, holds an id corresponding to the trial it originated 
        from. (Relevant if windows_as_examples to combine compute window 
        predictions to trial predictions.)
    feature_names: `pd.DataFrame`
    """
    trial_col = _find_col(df.columns, 'Trial')
    target_col = _find_col(df.columns, 'Target')
    window_col = _find_col(df.columns, 'Window')
    if agg_func is not None:
        if windows_as_examples:
            log.warning("'windows_as_examples' without effect if 'agg_func' is "
                        "not None.")
        if window_col not in df.columns or len(set(df[window_col])) == 1:
            log.warning("Data was already aggregated.")
        else:
            df = _aggregate_windows(df=df, agg_func=agg_func)
    else:
        if not windows_as_examples:
            df = _examples_from_windows(df)
    # for data and feature names, ignore 'Description' domain
    feature_cols = [col for col in df.columns if 'Description' not in col]
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    groups = df[trial_col].to_numpy()
    # feature names start after 'Description' domain
    feature_names = df.columns[len(df.columns)-len(feature_cols):].to_frame(
        index=False)
    assert len(feature_names) == X.shape[-1]
    assert X.shape[0] == y.shape[0] == groups.shape[0]
    corrs = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    log.info(f"Feature '{'_'.join(feature_names.iloc[corrs.argmax()])}' has "
             f"the highest correlation of any feature with the target at a "
             f"value of {corrs.max():.2f}.")
    if corrs.max() > .5:
        log.warning('Did you accidentally add the target to the feature matrix'
                    '?')
    return X, y, groups, feature_names


def score(score_func, y, y_pred, y_groups=None):
    """Compute the score func on predictions.
    
    Parameters
    ----------
    score_func: callable
        A function that takes y and y_pred and returns a score.
    y: array-like
        Window labels.
    y_pred: array-like
        Window predictions.
    groups: array-like
        Mapping of labels and predictions to groups.
    
    Returns
    -------
    Window and trialwise score.
    """
    y = np.array(y)
    y_pred = np.array(y_pred)
    scores = {'window_' + score_func.__name__: score_func(y, y_pred)}
    if all(np.bincount(y_groups) == 1):
        log.info('Window and trial accuracy will be identical, since there '
                 'is always exactly one window per trial.')
    pred_df = {'y': y, 'pred': y_pred, 'group': y_groups}
    trial_pred, trial_y = [], []
    for n, g in pd.DataFrame(pred_df).groupby('group'):
        trial_pred.append(g.pred.value_counts().idxmax())  # TODO: verify
        assert len(g.y.unique()) == 1
        trial_y.append(g.y.value_counts().idxmax())    
    trial_pred = np.array(trial_pred)
    trial_y = np.array(trial_y)
    scores.update(
        {'trial_' + score_func.__name__: score_func(trial_pred, trial_y)}
    )
    return scores


def _examples_from_windows(df):
    target_col = _find_col(df.columns, 'Target')
    trial_col = _find_col(df.columns, 'Trial')
    window_col = _find_col(df.columns, 'Window')
    # Check if we have variable length trials. If so, determine the minimum
    # number of windows of the shortest trial and use this number of windows
    # from every trial.
    n_windows_per_trial = df.groupby(trial_col).tail(1)[window_col]
    variable_length_trials = len(n_windows_per_trial.unique()) > 1
    n_windows_min = df.groupby(trial_col).tail(1)[window_col].min()
    if variable_length_trials:
        log.warning(f'Found inconsistent numbers of windows. '
                    f'Will use the minimum number of windows '
                    f'({n_windows_min+1}) as maximum.')
    new_df = []
    for group_i, ((_, window_i), g) in enumerate(
        df.groupby([trial_col, window_col])):
        # If trial has excessive windows, skip them
        if window_i > n_windows_min:
            continue
        targets = g.pop(target_col)
        trials = g.pop(trial_col)
        g.pop(window_col)
        # if this is the first window of a trial
        if window_i == 0:
            # if this is not the first trial
            if group_i != 0:
                # it means we just started touching a new trial, so concatenate
                # features already gathered and append to result
                new_df.append(pd.concat(flat, axis=1))
            # this is the very first trial. take its columns required for 
            # decoding and append
            targets.name = tuple(list(targets.name) + [''])
            trials.name = tuple(list(trials.name) + [''])
            flat = [targets.reset_index(drop=True),
                    trials.reset_index(drop=True)]
        # add a level to the multiindex which tells us the id of the window 
        # within each trial
        g.columns = [
            (c[0], c[1], str(window_i), c[2], c[3]) for c in g.columns]
        flat.append(g.reset_index(drop=True))
    # add feature vector of last trial
    new_df.append(pd.concat(flat, axis=1))
    # concatenate all trials and create a proper multiindex
    new_df = pd.concat(new_df, axis=0, ignore_index=True)
    new_df.columns = pd.MultiIndex.from_tuples(
        tuples=new_df.columns,
        names=df.columns.names[:2] + ['Window'] + df.columns.names[2:]
    )
    return new_df
