import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy.signal import hilbert

from braindecode_features.utils import (
    _generate_feature_names, _filter_and_window, _check_df_consistency)


log = logging.getLogger(__name__)


def get_time_feature_functions():
    """Get feature functions of time domain."""
    # TODO: add slow time domain features
    # https://github.com/gilestrolab/pyrem/blob/master/src/pyrem/univariate.py
    # Time domain features
    def covariance(X, include_diag=False): 
        covs = [np.cov(x) for x in X]
        # matrix is symmetrical, so vectorize.
        # ignore variance on diagonal, it is a feature itself
        covs_triu = [
            cov[np.triu_indices(cov.shape[-2], k=int(not include_diag))]
            for cov in covs]
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
                    Lmk += np.abs(X[:, :, m+i*k] - X[:, :, m+i*k-k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk, axis=0)))
            x.append([np.log(float(1) / k), 1])
        L = np.array(L)
        ps = []
        for i in range(L.shape[-2]):
            (p, r1, r2, s) = np.linalg.lstsq(x, L[:, i, :], rcond=None)
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
        # https://raphaelvallat.com/entropy/build/html/generated/entropy.petrosian_fd.html  # noqa
        diff = np.diff(X)
        crossings = np.sum((diff[:, :, 1:] * diff[:, :, :-1] < 0), axis=axis)
        N = X.shape[axis]
        return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * crossings)))

    def phase_locking_value(X):
        # remove empty first dimension
        X = hilbert(X, axis=-1)
        instantatneous_phases = np.unwrap(np.angle(X), axis=-1)
        plvs = []
        for ch_i, ch_j in zip(*np.triu_indices(X.shape[-2], k=1)):
            plv = _phase_locking_value(
                theta1=instantatneous_phases[:, ch_i],
                theta2=instantatneous_phases[:, ch_j],
            )
            plvs.append(plv)
        plvs = np.array(plvs).T
        return plvs

    def _phase_locking_value(theta1, theta2):
        delta = np.subtract(theta1, theta2)
        xs_mean = np.mean(np.cos(delta), axis=-1)
        ys_mean = np.mean(np.sin(delta), axis=-1)
        plv = np.linalg.norm([xs_mean, ys_mean], axis=0)
        return plv

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

    def zero_crossings(X): return np.sum(((X[:, :, 1:] * X[:, :, :-1]) < 0),
                                         axis=-1)

    def zero_crossings_derivative(X):
        diff = np.diff(X, axis=-1)
        return np.sum(((diff[:, :, 1:] * diff[:, :, :-1]) < 0), axis=-1)
    
    funcs = [
        covariance,
        energy, 
        higuchi_fractal_dimension,
        interquartile_range,
        kurtosis,
        line_length,
        maximum,
        mean,
        median,
        minimum, 
        petrosian_fractal_dimension,
        phase_locking_value,
        root_mean_square,
        # shannon_entropy,
        skewness,
        standard_deviation,
        value_range,
        variance,
        zero_crossings,
        zero_crossings_derivative,
    ]
    return funcs


def extract_time_features(concat_ds, frequency_bands, fu, windowing_fn):
    """Extract features in time domain. Therefore, iterate all the datasets. Use
    windows of prefiltered band signals and compute features.
    
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
    time_df: DataFrame
        The time domain feature DataFrame including target information and
        feature name annotations.
    """
    windows_ds = _filter_and_window(
        ds=concat_ds,
        frequency_bands=frequency_bands,
        windowing_fn=windowing_fn,
    )
    log.debug('Extracting ...')
    frequency_bands_str = ['-'.join([str(b[0]), str(b[1])])
                           for b in frequency_bands]
    all_band_channels = []
    for frequency_band in frequency_bands_str:
        # pick all channels corresponding to a single frequency band
        all_band_channels.append([
            ch for ch in windows_ds.datasets[0].windows.ch_names
            if ch.endswith(frequency_band)])
    time_df = []
    for ds_i, ds in enumerate(tqdm(windows_ds.datasets)):
        # for time domain features only consider the signals filtered in time
        # domain
        # #filtered_channels = [
        # ch for ch in ds.windows.ch_names if ch not in sensors]
        f, feature_names = [], []
        # TODO: do all bands at the same time?
        # get the band data
        all_data = np.array([ds.windows.get_data(picks=band_channels) 
                             for band_channels in all_band_channels])
        #f.append(fu.fit_transform(all_data).astype(np.float32))
        #log.info(f"time all_data after union shape {f[-1].shape}")
        log.debug(f'fu transformer list {len(fu.transformer_list)}')

        for data, band_channels, frequency_band in zip(
                all_data, all_band_channels, frequency_bands_str):
            log.debug(f'time in {frequency_band} before union {data.shape}')
            # call all features in the union
            f.append(fu.fit_transform(data).astype(np.float32))
            # first, remove frequency info from channel names, then generate
            # feature names
            names = _generate_feature_names(fu, [ch.split('_')[0]
                                                 for ch in band_channels])
            # manually re-add the frequency band
            feature_names.append(['__'.join([name, frequency_band])
                                  for name in names])
        # concatenate frequency band feature and names in the identical way
        f = np.concatenate(f, axis=-1)
        log.debug(f'feature shape {f.shape}')
        feature_names = np.concatenate(feature_names, axis=-1)
        feature_names = ['__'.join(['Time', name]) for name in feature_names]
        time_df.append(
            pd.concat([
                # add dataset_id to be able to backtrack
                # pd.DataFrame({'Dataset': len(ds) * [ds_i]}),  # equiv Trial?
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
    _check_df_consistency(df=time_df)
    return time_df
