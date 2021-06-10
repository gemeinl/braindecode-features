import logging

import numpy as np
import pandas as pd

import pywt

from braindecode_features.utils import generate_feature_names, _get_unfiltered_chs


log = logging.getLogger(__name__)


def get_cwt_feature_functions():
    """Get feature functions of the wavelet domain."""
    # CWT
    def bounded_variation(X):
        # https://sci-hub.se/10.1109/issnip.2008.4762005, p310, 5)
        # is this correct? what is mean with 'wavelet band signals'?
        diffs = np.diff(X, axis=-1)
        abs_sums = np.sum(np.abs(diffs), axis=-1)
        max_c = np.max(X, axis=-1)
        min_c = np.min(X, axis=-1)
        return np.mean(np.divide(abs_sums, max_c - min_c), axis=0)
    def maximum(X): return np.mean(np.max(X, axis=-1), axis=0)
    def mean(X): return np.mean(np.mean(X, axis=-1), axis=0)
    def median(X): return np.mean(np.median(X, axis=-1), axis=0)
    def minimum(X): return np.mean(np.min(X, axis=-1), axis=0)
    def power(X): return np.mean(np.sum(np.abs(X)*np.abs(X), axis=-1), axis=0)
    def standard_deviation(X): return np.mean(np.std(X, axis=-1), axis=0)
    def value_range(X): return np.mean(np.ptp(X, axis=-1), axis=0)
    def variance(X): return np.mean(np.var(X, axis=-1), axis=0)
    

    funcs = [bounded_variation, maximum, mean, median, minimum, power, 
             standard_deviation, value_range, variance]
    return funcs


def extract_wavelet_features(windows_ds, frequency_bands, fu):
    """Extract wavelet transform features. Therefore, iterate all the datasets. 
    Use windows of unfiltered signals, apply continuous wavelet transform and 
    compute features. 
    
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
    cwt_df: DataFrame
        The wavelet domain feature DataFrame including target information and feature 
        name annotations.
    """
    w = 'morl'
    central_band = False
    step_width = 1
    cwt_df = []
    for ds_i, ds in enumerate(windows_ds.datasets):
        sfreq = ds.windows.info['sfreq']
        # for cwt features only consider the signals that were not yet filtered
        sensors = _get_unfiltered_chs(ds, frequency_bands)
        data = ds.windows.get_data(picks=sensors)
        f, feature_names = [], []
        for l_freq, h_freq in frequency_bands:
            # either use the central frequency of the given band
            if central_band:
                pseudo_freqs = [(h_freq + l_freq)/2]
            # or use multiple scales between highpass and lowpass
            else:
                pseudo_freqs = np.linspace(l_freq, h_freq, num=int((h_freq-l_freq)/step_width))
            # generate scales from chosen frequencies above
            scales = [_freq_to_scale(freq, w, sfreq) for freq in pseudo_freqs]
            # transformt the signals using cwt
            transform, _ = pywt.cwt(data, scales=scales, wavelet=w, sampling_period=1/sfreq)
            # call all featuresin the union
            f.append(fu.fit_transform(transform).astype(np.float32))
            # first, manually add the frequency band to the used channel names
            # then generate feature names
            feature_names.append(generate_feature_names(
                fu, ['__'.join([ch, '-'.join([str(l_freq), str(h_freq)])]) for ch in sensors]
            ))
        # concatenate frequency band feature and names in the identical way
        f = np.concatenate(f, axis=-1)
        feature_names = np.concatenate(feature_names, axis=-1)
        feature_names = ['__'.join(['CWT', name]) for name in feature_names]
        cwt_df.append(
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
    cwt_df = pd.concat(cwt_df, axis=0)
    # generate an additional (redundant) trial column
    series = ((cwt_df['i_window_in_trial'] == 0).cumsum() - 1)
    series.name = 'i_trial'
    cwt_df = pd.concat([series, cwt_df], axis=1)
    assert not pd.isna(cwt_df.values).any()
    assert not pd.isnull(cwt_df.values).any()
    assert np.abs(cwt_df.values).max() < np.inf
    return cwt_df


def _freq_to_scale(freq, wavelet, sfreq):
    """Compute cwt scale to given frequency
    see: https://de.mathworks.com/help/wavelet/ref/scal2frq.html """
    central_freq = pywt.central_frequency(wavelet)
    if not freq > 0:
        log.warning("'freq' smaller or equal to zero! Using .1 instead.")
        freq = .1
    scale = central_freq / freq
    return scale * sfreq
