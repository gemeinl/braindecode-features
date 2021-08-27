import os
import tempfile

import pandas as pd

from braindecode_features.serialization import save_features, read_features

from .utils import _df_n_series


def test_save_features():
    d, f = _df_n_series()
    df = pd.concat([d, f], axis=1)
    out_path = tempfile.mkdtemp()
    save_features(
        df=df,
        out_path=out_path,
    )
    assert os.path.exists(os.path.join(out_path, '0.h5'))
    assert len(os.listdir(out_path)) == 1


def test_read_features():
    d, f = _df_n_series()
    df = pd.concat([d, f], axis=1)
    out_path = tempfile.mkdtemp()
    save_features(
        df=df,
        out_path=out_path,
    )
    read_df = read_features(
        path=out_path,
        agg_func=None,
        n_jobs=1,
    )
    pd.testing.assert_frame_equal(df, read_df)
