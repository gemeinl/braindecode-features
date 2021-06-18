import pandas as pd

from braindecode_features.decoding import _examples_from_windows

    
def test_examples_from_windows():
    df = pd.DataFrame(
        data=[
            [4, 0, 0, .2, .89],
            [4, 0, 1, .4, .73],
            [2, 1, 0, 13.2, .08],
            [2, 1, 1, 13.2, .08],
        ],
        columns=pd.MultiIndex.from_tuples([
            ('Target', '', '', ''), 
            ('Trial', '', '' ,''),
            ('Window', '', '', ''),
            ('Time', 'line_length', '4-8', 'O1'),
            ('Phase', 'phase_locking_value', '8-13', 'O1-P5'),
        ], names=['Domain', 'Feature', 'Channel', 'Frequency']),
    )
    new_df = _examples_from_windows(df)
    expected_df = pd.DataFrame(
        data=[
            [4, 0, .2, .89, .4, .73],
            [2, 1, 13.2, .08, 13.2, .08],
        ],
        columns=pd.MultiIndex.from_tuples([
            ('Target', '', '', '', ''), 
            ('Trial', '', '' ,'', ''),
            ('Time', 'line_length', '0', '4-8', 'O1'),
            ('Phase', 'phase_locking_value', '0', '8-13', 'O1-P5'),
            ('Time', 'line_length', '1', '4-8', 'O1'),
            ('Phase', 'phase_locking_value', '1', '8-13', 'O1-P5'),
        ], names=['Domain', 'Feature']+['Window']+['Channel', 'Frequency']),
    )
    pd.testing.assert_frame_equal(new_df, expected_df)
