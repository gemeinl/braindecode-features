import logging

import numpy as np
import pandas as pd

from braindecode_features.utils import generate_feature_names, _get_unfiltered_chs


log = logging.getLogger(__name__)


def get_fourier_feature_functions():
    """Get feature functions of the Fourier domain."""
    # DFT
    # TODO: add spectral entropy?
    def maximum(transform): return np.max(np.abs(transform), axis=-1)
    def mean(transform): return np.mean(np.abs(transform), axis=-1)
    def median(transform): return np.median(np.abs(transform), axis=-1)
    def minimum(transform): return np.min(np.abs(transform), axis=-1)
    def peak_frequency(transform):
        amplitudes = np.abs(transform)
        return np.argmax(amplitudes, axis=-1)
    def power(transform): 
        return np.sum(np.abs(transform)*np.abs(transform), axis=-1)
    def standard_deviation(transform): return np.std(np.abs(transform), axis=-1)
    def value_range(transform): return np.ptp(np.abs(transform), axis=-1)
    def variance(transform): return np.var(np.abs(transform), axis=-1)
    
    funcs = [
        maximum, mean, median, minimum, peak_frequency, power, 
        standard_deviation, value_range, variance]
    return funcs


def extract_fourier_features(windows_ds, frequency_bands, fu):
    """Extract fourier transform features. Therefore, iterate all the datasets. 
    Use windows, apply Fourier transform and compute features. 
    
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
    dft_df: DataFrame
        The Fourier domain feature DataFrame including target information and feature 
        name annotations.
    """
    dft_df = []
    for ds_i, ds in enumerate(windows_ds.datasets):
        sfreq = ds.windows.info['sfreq']
        # for dft features only consider the signals that were not yet filtered
        sensors = _get_unfiltered_chs(ds, frequency_bands)
        data = ds.windows.get_data(picks=sensors)
        # transform the signals using fft
        transform = np.fft.rfft(data, axis=-1)
        bins = np.fft.rfftfreq(data.shape[-1], 1/sfreq)
        f, feature_names = [], []
        # TODO: give all bands at the same time?
        # TODO: enables power ratio and spectral entropy
        all_data = []
        for l_freq, h_freq in frequency_bands:
            # select the frequency bins that best fit the chosen frequency bands
            l_id = np.argmin(np.abs(bins-l_freq))
            h_id = np.argmin(np.abs(bins-h_freq))
            # get the data and the bins
            #data = (transform[:,:,l_id:h_id+1], bins[l_id:h_id+1])
            all_data.append(transform[:,:,l_id:h_id+1])
        
        for data, (l_freq, h_freq) in zip(all_data, frequency_bands):
            log.debug(f'dft before union {data.shape}')
            # call all features in the union 
            f.append(fu.fit_transform(data).astype(np.float32))
            # first, manually add the frequency band to the used channel names
            # then generate feature names
            feature_names.append(generate_feature_names(
                fu, ['__'.join([ch, '-'.join([str(l_freq), str(h_freq)])]) for ch in sensors]
            ))
        # concatenate frequency band feature and names in the identical way
        f = np.concatenate(f, axis=-1)
        feature_names = np.concatenate(feature_names, axis=-1)
        feature_names = ['__'.join(['Fourier', name]) for name in feature_names]
        dft_df.append(
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
    dft_df = pd.concat(dft_df, axis=0)
    # generate an additional (redundant) trial column
    series = ((dft_df['i_window_in_trial'] == 0).cumsum() - 1)
    series.name = 'i_trial'
    dft_df = pd.concat([series, dft_df], axis=1)
    assert not pd.isna(dft_df.values).any()
    assert not pd.isnull(dft_df.values).any()
    assert np.abs(dft_df.values).max() < np.inf
    return dft_df
