import logging

import numpy as np
import pandas as pd

from braindecode_features.utils import _generate_feature_names, _get_unfiltered_chs, _window, _check_df_consistency


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


def extract_fourier_features(concat_ds, frequency_bands, fu, windowing_fn):
    """Extract fourier transform features. Therefore, iterate all the datasets. 
    Use windows, apply Fourier transform and compute features. 
    
    Parameters
    ----------
    concat_ds: BaseConcatDataset of BaseDatasets
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
    windows_ds = _window(
        ds=concat_ds,
        windowing_fn=windowing_fn,
    )
    log.debug('Extracting ...')
    dft_df = []
    for ds_i, ds in enumerate(windows_ds.datasets):
        sfreq = ds.windows.info['sfreq']
        # for dft features only consider the signals that were not yet filtered
        sensors = _get_unfiltered_chs(ds, frequency_bands)
        data = ds.windows.get_data(picks=sensors)
        # transform the signals using fft
        transform = np.fft.rfft(data, axis=-1)
        transform = transform / data.shape[-1]
        bins = np.fft.rfftfreq(data.shape[-1], 1/sfreq)
        f, feature_names = [], []
        # TODO: give all bands at the same time?
        # TODO: enables power ratio and spectral entropy
        all_data = []
        for l_freq, h_freq in frequency_bands:
            # select the frequency bins that best fit the chosen frequency bands
            l_id = np.argmin(np.abs(bins-l_freq))
            h_id = np.argmin(np.abs(bins-h_freq))
            if ds_i == 0 and (bins[l_id] - l_freq != 0 or bins[h_id] - h_freq != 0):
                bin_width = bins[1]-bins[0]
                log.info(f'Am supposed to pick bins between {l_freq} and {h_freq} which is '
                         f'impossible. Will use the bins closest to your selection instead: '
                         f'{l_id*bin_width} – {h_id*bin_width}.')
            # get the data and the bins
            #data = (transform[:,:,l_id:h_id+1], bins[l_id:h_id+1])
            all_data.append(transform[:,:,l_id:h_id+1])
        
        for data, (l_freq, h_freq) in zip(all_data, frequency_bands):
            log.debug(f'dft in {l_freq} – {h_freq} before union {data.shape}')
            # call all features in the union 
            f.append(fu.fit_transform(data).astype(np.float32))
            # first, manually add the frequency band to the used channel names
            # then generate feature names
            feature_names.append(_generate_feature_names(
                fu, ['__'.join([ch, '-'.join([str(l_freq), str(h_freq)])]) for ch in sensors]
            ))
        # concatenate frequency band feature and names in the identical way
        f = np.concatenate(f, axis=-1)
        log.debug(f'feature shape {f.shape}')
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
    _check_df_consistency(df=dft_df)
    return dft_df
