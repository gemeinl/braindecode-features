import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from braindecode_features.decoding import (
    _examples_from_windows, score, prepare_features)

from .utils import create_fake_feature_df

    
def test_examples_from_windows():
    df = create_fake_feature_df()
    new_df = _examples_from_windows(df)
    expected_df = pd.DataFrame(
        data=[
            [4, 0, .2, .89, .4, .73],
            [2, 1, 13.2, .08, 13.2, .08],
        ],
        columns=pd.MultiIndex.from_tuples([
            ('Description', 'Target', '', '', ''),
            ('Description', 'Trial', '', '', ''),
            ('Time', 'line_length', '0', '4-8', 'O1'),
            ('Phase', 'phase_locking_value', '0', '8-13', 'O1-P5'),
            ('Time', 'line_length', '1', '4-8', 'O1'),
            ('Phase', 'phase_locking_value', '1', '8-13', 'O1-P5'),
        ], names=['Domain', 'Feature']+['Window']+['Channel', 'Frequency']),
    )
    assert list(expected_df.columns) == list(new_df.columns)
    pd.testing.assert_frame_equal(new_df, expected_df)


def test_score():
    y = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 1]
    results = score(
        score_func=accuracy_score,
        y=y,
        y_pred=y_pred,
        y_groups=None,
    )
    assert len(results) == 1
    assert results['window_accuracy_score'] == 4/5

    # multiple score funcs with the same name
    with pytest.warns(UserWarning, match="Score 'window_accuracy_score' alrea"):
        results = score(
            score_func=[accuracy_score, accuracy_score],
            y=y,
            y_pred=y_pred,
            y_groups=None,
        )
        assert len(results) == 1
        assert results['window_accuracy_score'] == 4/5

    # test groups
    results = score(
        score_func=accuracy_score,
        y=y,
        y_pred=y_pred,
        y_groups=[0, 0, 1, 2, 2],
    )
    assert len(results) == 2
    assert results['window_accuracy_score'] == 4/5
    assert results['trial_accuracy_score'] == 2/3

    # test groups tie break true label 0
    results = score(
        score_func=accuracy_score,
        y=y,
        y_pred=y_pred,
        y_groups=[0, 1, 1, 2, 2],
    )
    assert len(results) == 2
    assert results['window_accuracy_score'] == 4/5
    assert results['trial_accuracy_score'] == 1

    # test groups tie break true label 1
    y = [0, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1]
    results = score(
        score_func=accuracy_score,
        y=y,
        y_pred=y_pred,
        y_groups=[0, 1, 1, 2, 2],
    )
    assert len(results) == 2
    assert results['window_accuracy_score'] == 4/5
    assert results['trial_accuracy_score'] == 2/3
    # -> will tie break to lower class id


def test_prepare_features():
    df = create_fake_feature_df()
    # agg_func = None, windows_as_examples = False
    X, y, groups, feature_names = prepare_features(
        df=df,
        agg_func=None,
        windows_as_examples=False,
    )
    assert X.shape == (2, 4)
    np.testing.assert_array_equal(y, [4, 2])

    # agg_func = None, windows_as_examples = True
    X, y, groups, feature_names = prepare_features(
        df=df,
        agg_func=None,
        windows_as_examples=True,
    )
    assert X.shape == (4, 2)
    np.testing.assert_array_equal(y, [4, 4, 2, 2])

    # agg_func = 'mean', windows_as_examples = True
    X, y, groups, feature_names = prepare_features(
        df=df,
        agg_func='mean',
        windows_as_examples=False,
    )
    assert X.shape == (2, 2)
    np.testing.assert_array_equal(y, [4, 2])

    # agg_func = 'mean', windows_as_examples = False
    with pytest.warns(UserWarning, match="'windows_as_examples' without effec"):
        X, y, groups, feature_names = prepare_features(
            df=df,
            agg_func='mean',
            windows_as_examples=True,
        )
        assert X.shape == (2, 2)
        np.testing.assert_array_equal(y, [4, 2])

    # test raise warning when target in features
    d = pd.DataFrame(
        [4, 4, 2, 2],
        columns=pd.MultiIndex.from_tuples([
            ('TargetFeature ', 'for', 'test', 'case')
        ], names=['Domain', 'Feature', 'Channel', 'Frequency']),
    )
    df = pd.concat([df, d], axis=1)
    with pytest.warns(UserWarning, match='Did you accidentally add the target'):
        X, y, groups, feature_names = prepare_features(
            df=df,
            agg_func=None,
            windows_as_examples=True,
        )
