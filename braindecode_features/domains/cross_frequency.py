import logging

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from braindecode_features.utils import generate_feature_names, _get_unfiltered_chs


log = logging.getLogger(__name__)


def get_cross_frequency_feature_functions():
    """Get feature functions of the cross-frequency domain."""
    # TODO klassisch: theta zu gamma band
    def cross_frequency_coupling(data1_n_data2):
        # loosely following https://mark-kramer.github.io/Case-Studies-Python/07.html
        data1, data2 = data1_n_data2
        instantaneous_phases = np.angle(hilbert(data1, axis=-1))
        amplitudes = np.abs(hilbert(data2, axis=-1))
        phase_bins = np.arange(-np.pi, np.pi, 0.1)
        a_mean = []
        for i in range(phase_bins.size-1):
            mask = (instantaneous_phases >= phase_bins[i]) & (instantaneous_phases < phase_bins[i+1])
            ma = np.ma.masked_array(amplitudes, ~mask)
            a_mean.append(np.mean(ma, axis=-1))
            #p_mean.append(np.mean([phase_bins[i], phase_bins[i+1]]))
        a_mean = np.array(a_mean)
        # as a statistic we take the difference of highest and lowest coupling
        return np.ptp(a_mean, axis=0)

    funcs = [cross_frequency_coupling]
    return funcs


def extract_cross_frequency_features(windows_ds, frequency_bands, fu):
    """Extract wavelet transform features. Therefore, iterate all the datasets. 
    Use windows of pairs of prefiltered signals and compute features. 
    
    Parameters
    ----------
    windows_ds: BaseConcatDataset of WindowsDataset
        Braindecode dataset to be used for feature extraction.
    frequency_bands: list(tuple)
        A list of frequency bands of prefiltered signals.
    fu: FeatureUnion
        Scikit-learn FeatureUnion of FunctionTransformers extracting features.
        
    Returns
    -------
    cross_frequency_df: DataFrame
        The cross frequency domain feature DataFrame including target information
        and feature name annotations.
    """
    # TODO: improve this, sometimes a band is contained in the other which probably
    # does not make too much sense
    # create all possible bands from all freq band limits in the form (low, high)
    all_possible_bands = [(frequency_bands[band_i], frequency_bands[band_j])
                          for band_i in range(len(frequency_bands)) 
                          for band_j in range(band_i+1, len(frequency_bands))]
    cross_frequency_df = []
    for ds_i, ds in enumerate(windows_ds.datasets):
        # get names of unfiltered channels
        sensors = _get_unfiltered_chs(ds, frequency_bands)
        f, feature_names = [], []
        for band1, band2 in all_possible_bands:
            # get the data of the low frequency band
            band1 = '-'.join([str(band1[0]), str(band1[1])])
            chs1 = [ch for ch in ds.windows.ch_names if ch.endswith(band1)]
            data1 = ds.windows.get_data(picks=chs1)
            # get the data of the high frequency band
            band2 = '-'.join([str(band2[0]), str(band2[1])])
            chs2 = [ch for ch in ds.windows.ch_names if ch.endswith(band2)]
            data2 = ds.windows.get_data(picks=chs2)
            data = np.array([data1, data2])
            # call all features in the union
            f.append(fu.fit_transform(data).astype(np.float32))
            # first, manually add the frequency bands to the used channel names
            # then generate feature names
            feature_names.append(generate_feature_names(
                fu, ['__'.join([ch, ', '.join([band1, band2])]) for ch in sensors]
            ))
        # concatenate frequency band feature and names in the identical way
        f = np.concatenate(f, axis=-1)
        feature_names = np.concatenate(feature_names, axis=-1)
        feature_names = ['__'.join(['Cross-frequency', name]) for name in feature_names]
        cross_frequency_df.append(
            pd.concat([
                # add dataset_id to be able to backtrack
                # pd.DataFrame({'Dataset': len(ds) * [ds_i]}),  # equivalent to Trial?
                # add trial and target info to features
                ds.windows.metadata[['i_window_in_trial', 'target']], 
                # create a dataframe of feature values and feature names
                pd.DataFrame(f, columns=feature_names)
            ], axis=1)
        )
    # concatenate all datasets
    cross_frequency_df = pd.concat(cross_frequency_df, axis=0)
    # generate an additional (redundant) trial column
    series = ((cross_frequency_df['i_window_in_trial'] == 0).cumsum() - 1)
    series.name = 'i_trial'
    cross_frequency_df = pd.concat([series, cross_frequency_df], axis=1)
    assert not pd.isna(cross_frequency_df.values).any()
    assert not pd.isnull(cross_frequency_df.values).any()
    assert np.abs(cross_frequency_df.values).max() < np.inf
    return cross_frequency_df
