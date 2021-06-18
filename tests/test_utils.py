import pandas as pd
import pytest

from braindecode_features.utils import _find_col, _aggregate_windows, drop_window, add_description

    
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
    columns=['Trial', 'Window', 'Target', 'data']
    df = pd.DataFrame(
        data=[
            [0, 0, 4, .2],
            [0, 1, 4, .4],
            [1, 0, 2, 13.2],
        ],
        columns=columns,
    )    
    expected_df = pd.DataFrame(
        data=[
            [0, 4, .3],
            [1, 2, 13.2],
        ],
        columns=['Trial', 'Target', 'data'],
    )
    avg_df = _aggregate_windows(df, 'mean')
    pd.testing.assert_frame_equal(avg_df, expected_df)

    
def test_add_description():
    columns=[('Description', 'Trial', '', ''),
             ('Description', 'Window', '', ''), 
             ('Description', 'Target', '', ''),
             ('data', '', '', ''),]
    df = pd.DataFrame(
        data=[
            [0, 0, 4, .2],
            [0, 1, 4, .4],
            [1, 0, 2, 13.2],
        ],
        columns=columns,
    )    
    df = add_description(
        df=df,
        description=[-1, -1, -1],
        name='Test',
    )
    expected_df = pd.DataFrame(
        data=[
            [-1, 0, 0, 4, .2],
            [-1, 0, 1, 4, .4],
            [-1, 1, 0, 2, 13.2],
        ],
        columns=pd.MultiIndex.from_tuples(
            [('Description', 'Test', '', '')] + columns),
    )
    pd.testing.assert_frame_equal(expected_df, df)


def test_drop_window():
    columns=[('Description', 'Trial', '', ''),
             ('Description', 'Window', '', ''), 
             ('Description', 'Target', '', ''),
             ('data', '', '', ''),]
    df = pd.DataFrame(
        data=[
            [0, 0, 4, .2],
            [0, 1, 4, .4],
            [1, 0, 2, 13.2],
        ],
        columns=columns,
    )
    df = drop_window(
        df=df,
        window_i=0,
    )
    expected_df = pd.DataFrame(
        data=[
            [0, 0, 4, .4],
        ],
        columns=columns,
    )
    pd.testing.assert_frame_equal(expected_df, df)
