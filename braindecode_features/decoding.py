import logging

from sklearn.model_selection import KFold, GroupKFold
import pandas as pd
import numpy as np

from .utils import _find_col, _aggregate_windows


log = logging.getLogger(__name__)

def prepare_features(df, agg_func=None, windows_as_examples=False):
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
            log.warning("'windows_as_examples' without effect if 'agg_func' is not None.")
        if window_col not in df.columns or len(set(df[window_col])) == 1:
            log.warning("Data was already aggregated.")
        else:
            df = _aggregate_windows(df=df, agg_func=agg_func)
    else:
        if not windows_as_examples:
            df = _examples_from_windows(df)
    # for data and feature names, ignore first 3 columns (Target, Dataset, Window)
    X = df[df.columns[3:]].to_numpy()
    y = df[target_col].to_numpy()
    groups = df[trial_col].to_numpy()
    feature_names = df.columns[3:].to_frame(index=False)
    assert len(feature_names) == X.shape[-1]
    assert X.shape[0] == y.shape[0] == groups.shape[0]
    return X, y, groups, feature_names


def cross_validate(
    df, clf, subject_id, only_last_fold, agg_func, windows_as_examples, 
    out_path=None):
    """
    Run (cross-)validation on features and targets in df using estimator clf.
    
    Parameters
    ----------
    df: `pd.DataFrame`
        A feature DataFrame.
    clf: sklearn.estimator
        A scikit-learn estimator.
    subject_id: int
    only_last_fold: bool
        Whether to only run the last fold of CV. Corresponds to 80/20 split.
    agg_func: callable
        Function to aggregate trial features, e.g. mean, median...
    windows_as_examples: bool
        Whether to consider compute windows as independent examples.
    out_path: str
        Directory to save 'cv_results.csv' to.
    """
    invalid_cols = [
        (col, ty) for col, ty in df.dtypes.items() if ty not in ['float32', 'int64']]
    if invalid_cols:
        log.error(f'Only integer and float values are allowed to exist in the DataFrame. '
              f'Found {invalid_cols}. Please convert.')
        return
    assert not pd.isna(df).values.any(), 'Found NaN in DataFrame.'
    assert not pd.isnull(df.values.any()), 'Found null in DataFrame.'
    assert df.isin(df.values).values.all(), 'Found inf in DataFrame.'
    results = pd.DataFrame()
    n_splits = 5
    X, y, groups, feature_names = prepare_features(
        df=df,
        agg_func=agg_func,
        windows_as_examples=windows_as_examples,
    )
    if agg_func is not None or not windows_as_examples:
        # preserves order of examples but might split groups
        # therefore don't use when not aggregating and using windows as examples
        cv = KFold(n_splits=n_splits, shuffle=False)
    else:
        # does not preserve order of examples but guarantees not splitting groups
        cv = GroupKFold(n_splits=n_splits)

    #if isinstance(clf, AutoSklearnClassifier) or isinstance(clf, AutoSklearn2Classifier):
    if hasattr(clf, 'refit'):
        # optimize hyperparameters on entire training data
        dataset_name = '_'.join(['subject', str(subject_id)])
        clf = clf.fit(X=X, y=y, dataset_name=dataset_name)
    # perform validation
    infos = []
    for fold_i, (train_is, valid_is) in enumerate(cv.split(X, y, groups)):
        if only_last_fold and fold_i != cv.n_splits - 1:
            continue
        X_train, y_train, groups_train = X[train_is], y[train_is],groups[train_is]
        X_valid, y_valid, groups_valid = X[valid_is], y[valid_is], groups[valid_is]
        log.debug(f'train shapes {X_train.shape}, {y_train.shape}, {groups_train.shape}')
        log.debug(f'valid shapes {X_valid.shape}, {y_valid.shape}, {groups_valid.shape}')
        if hasattr(clf, 'refit'):
            # for autosklearn, refit the ensemble found on the entire training set
            # on the training data of the cv split
            clf = clf.refit(X_train, y_train)
        else:
            clf = clf.fit(X_train, y_train)
        pred = clf.predict(X_valid)
        # compute window and trial accuracy
        window_acc = window_accuracy(y_valid, pred)
        trial_acc = trial_accuracy(y_valid, pred, y_groups=groups_valid)
        if agg_func is not None or not windows_as_examples:
            assert window_acc == trial_acc

        info = pd.DataFrame([pd.Series({
            'subject': subject_id,
            'fold': fold_i,
            'estimator': clf.__class__.__name__,
            'window_accuracy': window_acc, 
            'trial_accuracy': trial_acc,
            'predictions': pred.tolist(),
            'targets': y_valid.tolist(),
            'model': clf.show_models() if hasattr(clf, 'show_models') else str(clf),
            'feature_names': feature_names.to_dict(),
            'windows_as_examples': windows_as_examples,
            'agg_func': agg_func.__name__ if agg_func is not None else agg_func,
        })])
        if out_path is not None:
            out_file = os.path.join(os.path.dirname(os.path.dirname(out_path)), 'cv_results.csv')
            info.to_csv(out_file, mode='a', header=not os.path.exists(out_file))
        cols = ['estimator', 'agg_func', 'windows_as_examples', 'window_accuracy', 'trial_accuracy']
        log.info(info[cols].tail(1))
        infos.append(info)
    return pd.concat(infos)


def window_accuracy(y, y_pred):
    """Compute the accuracy of window predictions.
    
    Parameters
    ----------
    y: array-like
        Window labels.
    y_pred: array-like
        Window predictions.
        
    Returns
    -------
    Window accuracy.
    """
    return (y == y_pred).mean()


def trial_accuracy(y, y_pred, y_groups):
    """Compute the accuracy of window predictions.
    
    Parameters
    ----------
    y: array-like
        Window labels.
    y_pred: array-like
        Window predictions.
    groups: array-like
        Mapping of labels and predictions to groups.
    
    Returns
    -------
    Trial accuracy.
    """
    pred_df = {'y': y, 'pred': y_pred, 'group': y_groups}
    trial_pred, trial_y = [], []
    for n, g in pd.DataFrame(pred_df).groupby('group'):
        trial_pred.append(g.pred.value_counts().idxmax())  # TODO: verify
        assert len(g.y.unique()) == 1
        trial_y.append(g.y.value_counts().idxmax())    
    trial_pred = np.array(trial_pred)
    trial_y = np.array(trial_y)
    return (trial_pred == trial_y).mean()


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
                    f'Will use the minimum number of windows ({n_windows_min+1}) '
                    f'as maximum.')
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
