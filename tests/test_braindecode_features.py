import pandas as pd

from braindecode_features import filter_df
from braindecode_features.utils import _find_col


def test_filter_df():
    columns=[
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
