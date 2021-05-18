import pandas as pd
import pytest

from mne_features.braindecode_features import (
    multiindex_to_json, json_to_multiindex, filter_df, NAMES,
    _create_explicit_multiindex, _find_col, _find_col,
    _get_trial_window_mapping, _examples_from_windows, _aggregate_windows
)


def test_multiindex_to_json():
    multiindex = pd.MultiIndex.from_tuples([
        ('Test', '', '4'),
        ('Another', 'Test', ''),
    ], names=['C1', 'C2', 'C3']
    )
    json = multiindex_to_json(multiindex)
    assert json == ['{"C1": "Test", "C2": "", "C3": "4"}',
                    '{"C1": "Another", "C2": "Test", "C3": ""}']
    assert multiindex.equals(json_to_multiindex(multiindex_to_json(multiindex)))

    
def test_json_to_multiindex():
    json = ['{"C1": "Test", "C2": "", "C3": "4"}',
            '{"C1": "Another", "C2": "Test", "C3": ""}']
    multiindex = json_to_multiindex(json)
    assert multiindex.equals(pd.MultiIndex.from_tuples([
        ('Test', '', '4'),
        ('Another', 'Test', ''),
    ],
        names=['C1', 'C2', 'C3']
    ))
    assert json == multiindex_to_json(json_to_multiindex(json))

    
def test_create_explicit_multiindex():
    columns = [
        ('power', 'DFT/C1_4-8'),
        ('power', 'DFT/O2_4-8-T4_4-8'),
        ('line_length', 'Time/Fpz_4-8_8-13'),
    ]
    multiindex = _create_explicit_multiindex(columns)
    assert multiindex.equals(pd.MultiIndex.from_tuples([
        ('DFT', 'power', '4-8', 'C1'),
        ('DFT', 'power', '4-8', 'O2, T4'),
        ('Time', 'line_length', '4-8, 8-13', 'Fpz')
    ],
        names=['Domain', 'Feature', 'Frequency', 'Channel']
    ))

    
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
    columns=['Trial', 'Target', 'Window', 'data']
    df = pd.DataFrame(
        data=[
            [0, 4, 0, .2],
            [0, 4, 1, .4],
            [1, 2, 0, 13.2],
        ],
        columns=columns,
    )    
    expected_df = pd.DataFrame(
        data=[
            [0, 4, 0, .3],
            [1, 2, 0, 13.2],
        ],
        columns=columns,
    )
    avg_df = _aggregate_windows(df, 'mean')
    pd.testing.assert_frame_equal(avg_df, expected_df)
    
    
def test_filter_df():
    columns=[
        ('Target', '', '', ''), 
        ('Trial', '', '' ,''),
        ('Window', '', '', ''),
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
        df[[_find_col(df, c) for c in ['Target', 'Trial', 'Window', 'Phase']]])  
    
    
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
        ], names=NAMES),
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
        ], names=list(NAMES[:2])+['Window']+list(NAMES[2:])),
    )
    pd.testing.assert_frame_equal(new_df, expected_df)
    