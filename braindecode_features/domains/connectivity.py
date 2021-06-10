import logging

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from braindecode_features.utils import generate_feature_names


log = logging.getLogger(__name__)


def get_connectivity_feature_functions():
    """Get feature functions of the connectivity domain."""
    # Connectivity
    # TODO: add autocorrelation
    # https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
    def phase_locking_value(X):
        #assert X.ndim == 4, X.shape
        #X = np.squeeze(X, axis=0)
        # remove empty first dimension
        analytical_signal = hilbert(X, axis=-1)
        instantatneous_phases = np.unwrap(np.angle(analytical_signal), axis=-1)
        plvs = []
        for ch_i, ch_j in zip(*np.triu_indices(X.shape[-2], k=1)):
            plv = _phase_locking_value(
                theta1=instantatneous_phases[:,ch_i],
                theta2=instantatneous_phases[:,ch_j],
            )
            plvs.append(plv)
        plvs = np.array(plvs).T
        #plvs = np.expand_dims(plvs, axis=0)
        return plvs
    def _phase_locking_value(theta1, theta2):
        delta = np.subtract(theta1, theta2)
        xs_mean = np.mean(np.cos(delta), axis=-1)
        ys_mean = np.mean(np.sin(delta), axis=-1)
        plv = np.linalg.norm([xs_mean, ys_mean], axis=0)
        return plv

    
    funcs = [phase_locking_value]
    #return [MyFunctionTransformer(func) for func in funcs]
    return funcs


def extract_connectivity_features(windows_ds, frequency_bands, fu):
    """Extract connectivity features from pairs of signals in time domain. 
    Therefore, iterate all the datasets. Use windows of prefiltered band signals 
    and compute features. 
    
    Parameters
    ----------
    windows_ds: BaseConcatDataset of WindowsDataset
        Braindecode dataset to be used for feature extraction.
    frequency_bands: list(tuple)
        A list of frequency bands of prefiltered signals.
    fu: FeatureUnion
        Scikit-learn FeatureUnion of MyFunctionTransformers extracting features.
        
    Returns
    -------
    connectivity_df: DataFrame
        The connectivity domain feature DataFrame including target information and feature 
        name annotations.
    """
    connectivity_df = []
    for ds_i, ds in enumerate(windows_ds.datasets):
        # for connectivity domain features only consider the signals filtered in time domain
        #filtered_channels = [ch for ch in ds.windows.ch_names if ch not in sensors]
        f, feature_names = [], []
        for frequency_band in frequency_bands:
            # pick all channels corresponding to a single frequency band
            band_channels = [ch for ch in ds.windows.ch_names 
                             if ch.endswith('-'.join([str(b) for b in frequency_band]))]
            # get the band data
            data = ds.windows.get_data(picks=band_channels)    
            # add empty fourth dimension representing a single frequency band
            #data = data[None, :]
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
                # add dataset_id to be able to backtrack
                # pd.DataFrame({'Dataset': len(ds) * [ds_i]}),  # equivalent to Trial?
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
