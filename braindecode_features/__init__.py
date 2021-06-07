import os
import re
import glob
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import hilbert
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold, GroupKFold

import pywt


class MyFunctionTransformer(FunctionTransformer):
    """Inspired by mne features. Wrap a feature function. Upon call of transform
    save the shape of the input data and output data. Implement a get_feature_names()
    method that returns a list of length corresponding to input channels.
    """
    def get_feature_names(self):
        assert hasattr(self, 'chs_in_') and hasattr(self, 'chs_out_'), (
            'Call transform on your data first.')
        chs_in = self.chs_in_
        chs_out = self.chs_out_
        if chs_out == chs_in:
            return [f'ch{i}' for i in range(chs_out)]
        elif chs_out == (chs_in * (chs_in - 1)) / 2:
            include_diag = False
        else:
            assert chs_out == chs_in * chs_in / 2, chs_out
            include_diag = True
        return [f'ch{i}-ch{j}' for i, j in zip(*np.triu_indices(chs_in, k=int(not include_diag)))]

    def transform(self, X):
        # TODO: this is not nice. DFT features get a tuple of transform and frequency bins
        # could probably remove the bins. could then use argmax of amplitudes instead of 
        # bins at the position of argmax amplitudes for peak frequency
        n_dim_in = X.ndim if not isinstance(X, tuple) else X[0].ndim
        self.chs_in_ = X.shape[-2] if not isinstance(X, tuple) else X[0].shape[-2]
        X = super().transform(X=X)
        # if there is more than one value per channel, last dimension will not be squeezed
        # then channels should be second last. if there is one value per channel, channels
        # are last dimension
        # TODO: could also try to always keep last dimension as '1', so channels will always be
        # second last dimension
        self.chs_out_ = X.shape[-1] if not n_dim_in == X.ndim else X.shape[-2]
        return X

    
def generate_feature_names(fu, ch_names):
    """From the feature names returned by the feature functions through the feature union,
    replace the unknown channels indicated by ids (ch0 or ch0-ch13) with their actual names.
    """
    feature_names = fu.get_feature_names()
    mapping = {f'ch{i}': ch for i, ch in enumerate(ch_names)}
    # loop below taken from mne-features
    for pattern, translation in mapping.items():
        r = re.compile(rf'{pattern}(?=_)|{pattern}\b')
        feature_names = [
            r.sub(string=feature_name, repl=translation)
            for feature_name in feature_names]
    return feature_names


def extract_time_features(windows_ds, frequency_bands, fu):
    """
    """
    time_df = []
    for ds in windows_ds.datasets:
        # for time domain features only consider the signals filtered in time domain
        #filtered_channels = [ch for ch in ds.windows.ch_names if ch not in sensors]    
        f, feature_names = [], []
        for frequency_band in frequency_bands:
            # pick all channels corresponding to a single frequency band
            band_channels = [ch for ch in ds.windows.ch_names 
                             if ch.endswith('-'.join([str(b) for b in frequency_band]))]
            # get the band data
            data = ds.windows.get_data(picks=band_channels)    
            # call all features in the union
            f.append(fu.fit_transform(data).astype(np.float32))
            # first, remove frequency info from channel names, then generate feature names
            names = generate_feature_names(fu, [ch.split('_')[0] for ch in band_channels])
            # manually re-add the frequency band
            feature_names.append(['__'.join([
                name,
                '-'.join([str(b) for b in frequency_band]),
                ]) for name in names])
        # concatenate frequency band feature and names in the identical way
        f = np.concatenate(f, axis=-1)
        feature_names = np.concatenate(feature_names, axis=-1)
        feature_names = ['__'.join(['Time', name]) for name in feature_names]
        time_df.append(
            pd.concat([
                # add trial and target info to features
                ds.windows.metadata[['i_window_in_trial', 'target']], 
                # create a dataframe of feature values and feature names
                pd.DataFrame(f, columns=feature_names)
            ], axis=1)
        )
    # concatenate all the datasets
    time_df = pd.concat(time_df, axis=0)
    # generate an additional (redundant) trial column
    series = ((time_df['i_window_in_trial'] == 0).cumsum() - 1)
    series.name = 'i_trial'
    time_df = pd.concat([series, time_df], axis=1)
    assert not pd.isna(time_df.values).any()
    assert not pd.isnull(time_df.values).any()
    assert np.abs(time_df.values).max() < np.inf
    return time_df


def extract_connectivity_features(windows_ds, frequency_bands, fu):
    """
    """
    connectivity_df = []
    for ds in windows_ds.datasets:
        # for connectivity domain features only consider the signals filtered in time domain
        #filtered_channels = [ch for ch in ds.windows.ch_names if ch not in sensors]
        f, feature_names = [], []
        for frequency_band in frequency_bands:
            # pick all channels corresponding to a single frequency band
            band_channels = [ch for ch in ds.windows.ch_names 
                             if ch.endswith('-'.join([str(b) for b in frequency_band]))]
            # get the band data
            data = ds.windows.get_data(picks=band_channels)    
            # call all features in the union
            f.append(fu.fit_transform(data).astype(np.float32))
            # first, remove frequency info from channel names, then generate feature names
            names = generate_feature_names(fu, [ch.split('_')[0] for ch in band_channels])
            # manually re-add the frequency band
            feature_names.append(['__'.join([
                name,
                '-'.join([str(b) for b in frequency_band]),
                ]) for name in names])
        # concatenate frequency band feature and names in the identical way
        f = np.concatenate(f, axis=-1)
        feature_names = np.concatenate(feature_names, axis=-1)
        feature_names = ['__'.join(['Connectivity', name]) for name in feature_names]
        connectivity_df.append(
            pd.concat([
                # add trial and target info to features
                ds.windows.metadata[['i_window_in_trial', 'target']], 
                # create a dataframe of feature values and feature names
                pd.DataFrame(f, columns=feature_names)
            ], axis=1)
        )
    # concatenate all the datasets
    connectivity_df = pd.concat(connectivity_df, axis=0)
    # generate an additional (redundant) trial column
    series = ((connectivity_df['i_window_in_trial'] == 0).cumsum() - 1)
    series.name = 'i_trial'
    connectivity_df = pd.concat([series, connectivity_df], axis=1)
    assert not pd.isna(connectivity_df.values).any()
    assert not pd.isnull(connectivity_df.values).any()
    assert np.abs(connectivity_df.values).max() < np.inf
    return connectivity_df


def _get_unfiltered_chs(windows_ds, frequency_bands):
    orig_chs = []
    for ch in windows_ds.windows.ch_names:
        if any([ch.endswith('-'.join([str(low), str(high)])) for (low, high) in frequency_bands]):
            continue
        else:
            orig_chs.append(ch)
    return orig_chs


def extract_ft_features(windows_ds, frequency_bands, fu):
    """
    """
    dft_df = []
    for ds in windows_ds.datasets:
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
        for l_freq, h_freq in frequency_bands:
            # select the frequency bins that best fit the chosen frequency bands
            l_id = np.argmin(np.abs(bins-l_freq))
            h_id = np.argmin(np.abs(bins-h_freq))
            # get the data and the bins
            data = (transform[:,:,l_id:h_id+1], bins[l_id:h_id+1])
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
        feature_names = ['__'.join(['DFT', name]) for name in feature_names]
        dft_df.append(
            pd.concat([
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


def _freq_to_scale(freq, wavelet, sfreq):
    """Compute cwt scale to given frequency
    see: https://de.mathworks.com/help/wavelet/ref/scal2frq.html """
    central_freq = pywt.central_frequency(wavelet)
    if not freq > 0:
        print("freq smaller or equal to zero! using .1 instead")
        freq = .1
    scale = central_freq / freq
    return scale * sfreq


def extract_wavelet_features(windows_ds, frequency_bands, fu):
    """
    """
    w = 'morl'
    central_band = False
    step_width = 1
    cwt_df = []
    for ds in windows_ds.datasets:
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


def extract_cross_frequency_features(windows_ds, frequency_bands, fu):
    """
    """
    # TODO: improve this, sometimes a band is contained in the other which probably
    # does not make too much sense
    # create all possible bands from all freq band limits in the form (low, high)
    all_possible_bands = [(frequency_bands[band_i], frequency_bands[band_j])
                          for band_i in range(len(frequency_bands)) 
                          for band_j in range(band_i+1, len(frequency_bands))]
    cross_frequency_df = []
    for ds in windows_ds.datasets:
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
            # call all features in the union
            f.append(fu.fit_transform((data1, data2)).astype(np.float32))
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


def get_time_feature_functions():
    # TODO: add slow time domain features
    # https://github.com/gilestrolab/pyrem/blob/master/src/pyrem/univariate.py
    # Time domain features
    def covariance(X): 
        covs = [np.cov(x) for x in X]
        # matrix is symmetrical, so vectorize.
        # ignore variance on diagonal, it is a feature itself
        covs_triu = [cov[np.triu_indices(cov.shape[-2], k=1)] for cov in covs]
        return np.array(covs_triu)
    def energy(X): return np.mean(X*X, axis=-1)
    def higuchi_fractal_dimension(X, kmax):
        # multi-dim version tested vs 1d version of pyeeg, 1e-5 max difference
        # http://pyeeg.sourceforge.net/
        N = X.shape[-1]
        L, x = [], []
        for k in range(1, kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += np.abs(X[:,:,m+i*k] - X[:,:,m+i*k-k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk, axis=0)))
            x.append([np.log(float(1) / k), 1])
        L = np.array(L)
        ps = []
        for i in range(L.shape[1]):
            (p, r1, r2, s) = np.linalg.lstsq(x, L[:,i,:], rcond=None)
            ps.append(p[0,:])
        return np.array(ps)
    def interquartile_range(X, q=75): 
        q1, q2 = np.percentile(X, [q, 100-q], axis=-1)
        return q2 - q1
    def kurtosis(X): return stats.kurtosis(X, axis=-1)
    def line_length(X): return np.sum(np.abs(np.diff(X, axis=-1)), axis=-1)
    def maximum(X): return np.max(X, axis=-1)
    def mean(X): return np.mean(X, axis=-1)
    def median(X): return np.median(X, axis=-1)
    def minimum(X): return np.min(X, axis=-1)
    def petrosian_fractal_dimension(X, axis=-1):
        # https://raphaelvallat.com/entropy/build/html/generated/entropy.petrosian_fd.html
        diff = np.diff(X)
        zero_crossings = np.sum((diff[:,:,1:] * diff[:,:,:-1] < 0), axis=axis)
        N = X.shape[axis]
        return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * zero_crossings)))
    def root_mean_square(X): return np.sqrt(np.mean(X*X, axis=-1))
    #def shannon_entropy(X):
        # https://arxiv.org/pdf/2001.08386.pdf
        # unsuitable for time domain, only for frequency domain!
        # see https://www.interscience.in/cgi/viewcontent.cgi?article=1175&context=ijcct
        # for time domain
        #return -np.sum(X * np.log2(X), axis=-1)
    def skewness(X): return stats.skew(X, axis=-1)
    def standard_deviation(X): return np.std(X, axis=-1)
    def value_range(X): return np.ptp(X, axis=-1)
    def variance(X): return np.var(X, axis=-1)
    def zero_crossings(X): return np.sum(((X[:,:,1:] * X[:,:,:-1]) < 0), axis=-1)
    def zero_crossings_derivative(X): 
        diff = np.diff(X, axis=-1)
        return np.sum(((diff[:,:,1:] * diff[:,:,:-1]) < 0), axis=-1)
    
    
    funcs = [
        covariance, energy, 
        partial(higuchi_fractal_dimension, kmax=3), interquartile_range,
        kurtosis, line_length, maximum, mean, median, 
        minimum, petrosian_fractal_dimension, root_mean_square,
        #shannon_entropy,
        skewness, standard_deviation, variance, zero_crossings, 
        zero_crossings_derivative,
    ]
    return [MyFunctionTransformer(func=func) for func in funcs]


def get_connectivity_feature_functions():
    # Connectivity
    # TODO: add autocorrelation
    # https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
    def phase_locking_value(X):
        assert X.ndim == 3
        analytical_signal = hilbert(X, axis=-1)
        instantatneous_phases = np.unwrap(np.angle(analytical_signal), axis=-1)
        plvs = []
        for ch_i, ch_j in zip(*np.triu_indices(X.shape[1], k=1)):
            plv = _phase_locking_value(
                theta1=instantatneous_phases[:,ch_i],
                theta2=instantatneous_phases[:,ch_j],
            )
            plvs.append(plv)
        return np.array(plvs).T
    def _phase_locking_value(theta1, theta2):
        delta = np.subtract(theta1, theta2)
        xs_mean = np.mean(np.cos(delta), axis=-1)
        ys_mean = np.mean(np.sin(delta), axis=-1)
        plv = np.linalg.norm([xs_mean, ys_mean], axis=0)
        return plv

    
    funcs = [phase_locking_value]
    return [MyFunctionTransformer(func) for func in funcs]


def get_ft_feature_functions():
    # DFT
    # TODO: add spectral entropy?
    # TODO: remove freq bins from input tuple?
    def maximum(transform_n_bins): return np.max(np.abs(transform_n_bins[0]), axis=-1)
    def mean(transform_n_bins): return np.mean(np.abs(transform_n_bins[0]), axis=-1)
    def median(transform_n_bins): return np.median(np.abs(transform_n_bins[0]), axis=-1)
    def minimum(transform_n_bins): return np.min(np.abs(transform_n_bins[0]), axis=-1)
    def peak_frequency(transform_n_bins):
        transform, bins = transform_n_bins
        amplitudes = np.abs(transform)
        return bins[np.argmax(amplitudes, axis=-1)]
    def power(transform_n_bins): 
        transform, bins = transform_n_bins
        return np.sum(np.abs(transform)*np.abs(transform), axis=-1)
    def standard_deviation(transform_n_bins): return np.std(np.abs(transform_n_bins[0]), axis=-1)
    def value_range(transform_n_bins): return np.ptp(np.abs(transform_n_bins[0]), axis=-1)
    def variance(transform_n_bins): return np.var(np.abs(transform_n_bins[0]), axis=-1)
    
    funcs = [
        maximum, mean, median, minimum, peak_frequency, power, 
        standard_deviation, value_range, variance]
    return [MyFunctionTransformer(func=func) for func in funcs]


def get_cwt_feature_functions():
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
    return [MyFunctionTransformer(func=func) for func in funcs]


def get_cross_frequency_feature_functions():
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
    return [MyFunctionTransformer(func=func) for func in funcs]


def get_feature_functions():
    return {
        'Connectivity': get_connectivity_feature_functions(),
        'Cross-frequency': get_cross_frequency_feature_functions(),
        'CWT': get_cwt_feature_functions(),
        'DFT': get_ft_feature_functions(),
        'Time': get_time_feature_functions()
    }


def get_extraction_routines():
    return {
        'Connectivity': extract_connectivity_features,
        'Cross-frequency': extract_cross_frequency_features,
        'CWT': extract_wavelet_features,
        'DFT': extract_ft_features,
        'Time': extract_time_features,
    }

        
def get_feature_functions_and_extraction_routines():
        return get_feature_functions(), get_extraction_routines()

        
def _merge_dfs(dfs, on):
    df = dfs[0]
    for df_ in dfs[1:]:
        df = pd.merge(df_, df, on=on)
    return df


def finalize_df(dfs):
    df = _merge_dfs(
        dfs=dfs, 
        on=['i_trial', 'i_window_in_trial', 'target']
    )
    df = df.rename(
        mapper={'i_trial': 'Dataset', 
                'i_window_in_trial': 'Window', 
                'target': 'Target',
               }, axis=1)
    df.columns = pd.MultiIndex.from_tuples(
        [col.split('__') if '__' in col else [col, '', '', ''] for col in df.columns],
        names=['Domain', 'Feature', 'Channel', 'Frequency']
    )
    return df


def _find_col(columns, hint):
    found_col = [c for c in columns if hint in c]
    assert len(found_col) == 1, (
        f'Please be more precise, found: {found_col}')
    return found_col[0]


def save_features_by_trial(df, out_path, subject_id, split_name):
    """Save the feature DataFrame as 'h5' files to out_path one trial at a time.
    Thereby, create subdirectories for subjects and data splits.

    Parameters
    ----------
    df: pd.DataFrame
        The feature DataFrame as returned by `extract_windows_ds_features`
    out_path: str
        The path to the root directory in which subdirectories will be created
        and finally 'h5' files will be stored.
    subject_id: int
        The id of the subject the features belong to.
    split_name: str
        The name of the data split, e.g. 'train', 'valid', 'eval', ...
    """
    # under out_path create a subdirectory wrt subject_id and split_name 
    out_p = os.path.join(out_path, str(subject_id), split_name)
    if not os.path.exists(out_p):
        os.makedirs(out_p)
    trial_col = _find_col(df.columns, 'Dataset')
    for trial, feats in df.groupby(trial_col):
        # store as hdf files, since reading is much faster than csv
        feats.reset_index(inplace=True, drop=True)
        feats.to_hdf(os.path.join(out_p, f'{trial}.h5'), 'data')

        
def filter_df(df, query, exact_match=False, level_to_consider=None):
    """Filter the MultiIndex of a DataFrame wrt 'query'. Thereby, columns required
    for decoding, i.e., 'Target', 'Dataset', 'Window' are always preserved.

    Parameters
    ----------
    df: `pd.DataFrame`
        A DataFrame to be filtered.
    query: str
        The query to look for in the columns of the DataFrame.
    exact_match: bool
        Whether the query has to be matched exactly or just be contained.
    level_to_consider: int
        Limit the filtering to look at a specific level of the MultiIndex.

    Returns
    -------
    df: `pd.DataFrame`
        The DataFrame limited to the columns matching the query.
    """
    masks = []
    assert not isinstance(level_to_consider, str), (
        'Has to be int, or list of int')
    if level_to_consider is None:
        levels = range(df.columns.nlevels)
    elif hasattr(level_to_consider, '__iter__'):
        levels = level_to_consider
    else:
        levels = [level_to_consider]
    # if available add target, trial and window to selection 
    info_cols = []
    for info_col in ['Dataset', 'Window', 'Target']:
        if info_col in df:
            info_col = _find_col(df.columns, info_col)
            info_cols.append(info_col)
    # go through all the levels in the multiindex and check whether the query
    # matches any value exactly or just contains it
    for i in levels:
        if exact_match:
            mask = query == df.columns[len(info_cols):].get_level_values(i)
        else:
            mask = df.columns[len(info_cols):].get_level_values(i).str.contains(
                query)
        masks.append(mask)
    mask = np.sum(masks, axis=0) > 0

    multiindex = pd.MultiIndex.from_tuples(
        info_cols+df.columns[len(info_cols):][mask].to_list())
    return df[multiindex]


def _examples_from_windows(df):
    # for easier handling, convert to multiindex
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = json_to_multiindex(df.columns)
    target_col = _find_col(df.columns, 'Target')
    trial_col = _find_col(df.columns, 'Dataset')
    window_col = _find_col(df.columns, 'Window')
    # Check if we have variable length trials. If so, determine the minimum
    # number of windows of the shortest trial and use this number of windows
    # from every trial.
    n_windows_per_trial = df.groupby('Dataset').tail(1)['Window']
    variable_length_trials = len(n_windows_per_trial.unique()) > 1
    n_windows_min = df.groupby('Dataset').tail(1)['Window'].min()
    if variable_length_trials:
        print(f'Found inconsistent numbers of windows. '
              f'Will use the minimum number of windows ({n_windows_min+1}) '
              f'as maximum.')
    new_df = []
    for group_i, ((_, window_i), g) in enumerate(
        df.groupby([trial_col, window_col])):
        # If trial has excessive windows, skip them
        if window_i > n_windows_min:
            continue
        targets = g.pop(target_col)
        trials = g.pop(trial_col)
        g.pop(window_col)
        # if this is the first window of a trial
        if window_i == 0:
            # if this is not the first trial
            if group_i != 0:
                # it means we just started touching a new trial, so concatenate
                # features already gathered and append to result
                new_df.append(pd.concat(flat, axis=1))
            # this is the very first trial. take its columns required for 
            # decoding and append
            targets.name = tuple(list(targets.name) + [''])
            trials.name = tuple(list(trials.name) + [''])
            flat = [targets.reset_index(drop=True),
                    trials.reset_index(drop=True)]
        # add a level to the multiindex which tells us the id of the window 
        # within each trial
        g.columns = [
            (c[0], c[1], str(window_i), c[2], c[3]) for c in g.columns]
        flat.append(g.reset_index(drop=True))
    # add feature vector of last trial
    new_df.append(pd.concat(flat, axis=1))
    # concatenate all trials and create a proper multiindex
    new_df = pd.concat(new_df, axis=0, ignore_index=True)
    new_df.columns = pd.MultiIndex.from_tuples(
        tuples=new_df.columns,
        names=df.columns.names[:2] + ['Window'] + df.columns.names[2:]
    )
    return new_df


def prepare_features(df, agg_func=None, windows_as_examples=False):
    """Prepare a feature DataFrame for decoding, i.e., generate X and y, groups,
    and feature names. Thereby, optionally aggregates features by trials.
    Compute windows can either be used as independent examples, or to be
    appended to the feature vector.

    Parameters
    ----------
    df: `pd.DataFrame`
        The feature DataFrame as returned by `extract_windows_ds_features` and
        `read_features`.
    agg_func: None | str
        Aggregation function supported by `pd.DataFrame.agg()` used to
        optionally aggregate features by trials.        
    windows_as_examples: bool
        Whether to move the window dimension to examples or features. Without 
        effect if 'agg_func' is not None.
    Returns
    -------
    X: `np.ndarray`
        The feature matrix (n_examples x n_feautures).
    y: `np.ndarray`
        The targets (n_examples).
    groups: `np.ndarray`
        For every example, holds an id corresponding to the trial it originated 
        from. (Relevant if windows_as_examples to combine compute window 
        predictions to trial predictions.)
    feature_names: `pd.DataFrame`
    """
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = json_to_multiindex(df.columns)
    trial_col = _find_col(df.columns, 'Dataset')
    target_col = _find_col(df.columns, 'Target')
    window_col = _find_col(df.columns, 'Window')
    if agg_func is not None:
        if windows_as_examples:
            print("'windows_as_examples' without effect if 'agg_func' is not None.")
        if window_col not in df.columns or len(set(df[window_col])) == 1:
            print("Data was already aggregated.")
        else:
            df = _aggregate_windows(df=df, agg_func=agg_func)
    else:
        if not windows_as_examples:
            df = _examples_from_windows(df)
    # for data and feature names, ignore first 3 columns (Target, Dataset, Window)
    X = df[df.columns[3:]].to_numpy()
    y = df[target_col].to_numpy()
    groups = df[trial_col].to_numpy()
    feature_names = df.columns[3:].to_frame(index=False)
    assert len(feature_names) == X.shape[1]
    assert X.shape[0] == y.shape[0] == groups.shape[0]
    return X, y, groups, feature_names


def _aggregate_windows(df, agg_func):
    trial_col = _find_col(df.columns, 'Dataset')
    grouped = df.groupby(trial_col)
    df = grouped.agg(agg_func)
    df.reset_index(inplace=True)
    # set window id to zero when aggregating, will be dropped later
    # keep it at this point for compatibility
    df[_find_col(df.columns, 'Window')] = len(df) * [0]
    # agg changes dtype of target, force it to keep original dtype
    cols = [_find_col(df, col) for col in ['Dataset', 'Target', 'Window']]
    df[cols] = df[cols].astype(np.int64)
    return df


def _read_and_aggregate(path, agg_func):
    df = pd.read_hdf(path, key='data')
    if agg_func is not None:
        df = _aggregate_windows(
            df=df,
            agg_func=agg_func,
        )
    return df


def read_features(path, agg_func=None, columns_as_json=False, n_jobs=1):
    """Read the features of all 'h5' files in 'path'. Optionally aggregates
    features by trials and formats the output columns as JSON.

    Parameters
    ----------
    path: str
        The path from where to load the features.
    agg_func: None | str
        Aggregation function supported by `pd.DataFrame.agg()` used to
        optionally aggregate features by trials.
    columns_as_json: bool
        Whether to format the columns of the output DataFrame as JSON.
    n_jobs: int
        Number of workers to load files in parallel.
    Returns
    -------
    dfs: `pd.DataFrame`
        The feature DataFrame.
    """
    # read all 'h5' files in path and sort them ascendingly 
    file_paths = glob.glob(os.path.join(path, '*.h5'))
    file_paths = sorted(
        file_paths,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    dfs = Parallel(n_jobs=n_jobs)(delayed(_read_and_aggregate)(p, agg_func) for p in file_paths)
    dfs = pd.concat(dfs).reset_index(drop=True)
    if isinstance(dfs.columns, pd.MultiIndex) and columns_as_json:
        dfs.columns = multiindex_to_json(dfs.columns)
    return dfs


def _build_transformer_list(funcs):
    transformer_list = []
    for func in funcs:
        if hasattr(func.func, '__name__'):
            # for functions wrapped with MyFunctionTransformer
            transformer_list.append((func.func.__name__, func))
        else:
            # For functions additionally initialized with partial 
            transformer_list.append((func.func.func.__name__, func))
    return transformer_list


def extract_windows_ds_features(
    windows_ds, frequency_bands, feature_functions=None, 
    extraction_routines=None, n_jobs=1):
    """Extrac features from a braindecode WindowsDataset."""
    assert (extraction_routines is None and feature_functions is None) or (
        extraction_routines is not None and feature_functions is not None)
    if extraction_routines is None and feature_functions is None:
        feature_functions, extraction_routines = get_feature_functions_and_extraction_routines()
    domain_dfs = {}
    # extract features by domain, since each domain has it's very own routine
    for domain in extraction_routines.keys():
        if len(frequency_bands) == 1 and domain == 'Cross-frequency':
            continue
        print(domain)
        transformer_list = _build_transformer_list(feature_functions[domain])

        # TODO: find a solution that makes 'set_params()' work
        # such that partial does not have to be used
        fu = FeatureUnion(
            transformer_list=transformer_list,
            n_jobs=n_jobs,
        )
        # extract features of one domain at a time
        domain_dfs[domain] = extraction_routines[domain](
            windows_ds=windows_ds,
            frequency_bands=frequency_bands,
            fu=fu
        )
    # concatenate domain dfs and make final df pretty
    df = finalize_df(
        dfs=list(domain_dfs.values()),
    )
    return df


def drop_window(df, window_i):
    """Drop all rows of window_i in the feature DataFrame.
    
    Parameters
    ----------
    df: `pd.DataFrame`
        The feature dataframe.
    window_i: int
        The id of the window to be dropped.
    """
    # select all windows that are not window_i
    df = df[df.Window != window_i]
    windows = df.pop('Window')
    # TODO: is it OK to do this?
    # reindex the windows
    windows -= windows.values > window_i
    # insert the updated windows again
    df.insert(1, 'Window', windows)
    return df


def window_accuracy(y, y_pred):
    return (y == y_pred).mean()


def trial_accuracy(y, y_pred, y_groups):
    pred_df = {'y': y, 'pred': y_pred, 'group': y_groups}
    trial_pred, trial_y = [], []
    for n, g in pd.DataFrame(pred_df).groupby('group'):
        trial_pred.append(g.pred.value_counts().idxmax())  # TODO: verify
        assert len(g.y.unique()) == 1
        trial_y.append(g.y.value_counts().idxmax())    
    trial_pred = np.array(trial_pred)
    trial_y = np.array(trial_y)
    return (trial_pred == trial_y).mean()


def cross_validate(
    df, clf, subject_id, only_last_fold, agg_func, windows_as_examples, 
    out_path=None):
    invalid_cols = [
        (col, ty) for col, ty in df.dtypes.items() if ty not in ['float32', 'int64']]
    if invalid_cols:
        print(f'Only integer and float values are allowed to exist in the DataFrame. '
              f'Found {invalid_cols}. Please convert.')
        return
    assert not pd.isna(df).values.any(), 'Found NaN in DataFrame.'
    assert not pd.isnull(df.values.any()), 'Found null in DataFrame.'
    assert df.isin(df.values).values.all(), 'Found inf in DataFrame.'
    results = pd.DataFrame()
    n_splits = 5
    X, y, groups, feature_names = prepare_features(
        df=df,
        agg_func=agg_func,
        windows_as_examples=windows_as_examples,
    )
    if agg_func is not None or not windows_as_examples:
        # preserves order of examples but might split groups
        # therefore don't use when not aggregating and using windows as examples
        cv = KFold(n_splits=n_splits, shuffle=False)
    else:
        # does not preserve order of examples but guarantees not splitting groups
        cv = GroupKFold(n_splits=n_splits)

    #if isinstance(clf, AutoSklearnClassifier) or isinstance(clf, AutoSklearn2Classifier):
    if hasattr(clf, 'refit'):
        # optimize hyperparameters on entire training data
        dataset_name = '_'.join(['subject', str(subject_id)])
        clf = clf.fit(X=X, y=y, dataset_name=dataset_name)
    # perform validation
    for fold_i, (train_is, valid_is) in enumerate(cv.split(X, y, groups)):
        if only_last_fold and fold_i != cv.n_splits - 1:
            continue
        X_train, y_train, groups_train = X[train_is], y[train_is],groups[train_is]
        X_valid, y_valid, groups_valid = X[valid_is], y[valid_is], groups[valid_is]
        if hasattr(clf, 'refit'):
            # for autosklearn, refit the ensemble found on the entire training set
            # on the training data of the cv split
            clf = clf.refit(X_train, y_train)
        else:
            clf = clf.fit(X_train, y_train)
        pred = clf.predict(X_valid)
        # compute window and trial accuracy
        window_acc = window_accuracy(y_valid, pred)
        trial_acc = trial_accuracy(y_valid, pred, y_groups=groups_valid)
        if agg_func is not None or not windows_as_examples:
            assert window_acc == trial_acc

        info = pd.DataFrame([pd.Series({
            'subject': subject_id,
            'fold': fold_i,
            'estimator': clf.__class__.__name__,
            'window_accuracy': window_acc, 
            'trial_accuracy': trial_acc,
            'predictions': pred.tolist(),
            'targets': y_valid.tolist(),
            'model': clf.show_models() if hasattr(clf, 'show_models') else str(clf),
            'feature_names': feature_names.to_dict(),
            'windows_as_examples': windows_as_examples,
            'agg_func': agg_func.__name__ if agg_func is not None else agg_func,
        })])
        if out_path is not None:
            out_file = os.path.join(os.path.dirname(os.path.dirname(out_path)), 'cv_results.csv')
            info.to_csv(out_file, mode='a', header=not os.path.exists(out_file))
        cols = ['estimator', 'agg_func', 'windows_as_examples', 'window_accuracy', 'trial_accuracy']
        print(info[cols].tail(1))
