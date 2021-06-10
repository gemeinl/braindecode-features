import os
import glob

import pandas as pd
from joblib import Parallel, delayed

from .utils import _read_and_aggregate, _find_col


def save_features(df, out_path):
    """Save the feature DataFrame as 'h5' files to out_path one trial at a time.

    Parameters
    ----------
    df: `pd.DataFrame`
        The feature DataFrame as returned by `extract_windows_ds_features`.
    out_path: str
        The path to the root directory in which 'h5' files will be stored.
    """
    # under out_path create a subdirectory wrt subject_id and split_name 
    #out_p = os.path.join(out_path, str(subject_id), split_name)
    #if not os.path.exists(out_p):
    #    os.makedirs(out_p)
    trial_col = _find_col(df.columns, 'Trial')
    for trial, feats in df.groupby(trial_col):
        # store as hdf files, since reading is much faster than csv
        feats.reset_index(inplace=True, drop=True)
        feats.to_hdf(os.path.join(out_path, f'{trial}.h5'), 'data')
        
        
def read_features(path, agg_func=None, columns_as_json=False, n_jobs=1):
    """Read the features of all 'h5' files in 'path'. Optionally aggregates
    features by trials and formats the output columns as JSON.

    Parameters
    ----------
    path: str
        The path from where to load the features.
    agg_func: None | str
        Aggregation function supported by `pd.DataFrame.agg()` used to
        optionally aggregate features by trials.
    columns_as_json: bool
        Whether to format the columns of the output DataFrame as JSON.
    n_jobs: int
        Number of workers to load files in parallel.
        
    Returns
    -------
    dfs: `pd.DataFrame`
        The feature DataFrame.
    """
    # read all 'h5' files in path and sort them ascendingly 
    file_paths = glob.glob(os.path.join(path, '*.h5'))
    file_paths = sorted(
        file_paths,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    dfs = Parallel(n_jobs=n_jobs)(delayed(_read_and_aggregate)(p, agg_func) for p in file_paths)
    dfs = pd.concat(dfs).reset_index(drop=True)
    if isinstance(dfs.columns, pd.MultiIndex) and columns_as_json:
        dfs.columns = multiindex_to_json(dfs.columns)
    return dfs


def save_features_by_trial(df, out_path, subject_id, split_name):
    log.warning("Deprecated. Use 'save_features' instead. 'subject_id' and "
                "'split_name' no longer supported. Create subdirectories "
                "outside of this function.")

