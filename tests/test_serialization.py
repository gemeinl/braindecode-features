import os
import tempfile

import pandas as pd

from braindecode_features.serialization import save_features, read_features


def test_save_features():
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
    df = pd.concat([d, f], axis=1)
    out_path = tempfile.mkdtemp()
    save_features(
        df=df,
        out_path=out_path,
    )
    assert os.path.exists(os.path.join(out_path, '0.h5'))
    assert len(os.listdir(out_path)) == 1


def test_read_features():
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
