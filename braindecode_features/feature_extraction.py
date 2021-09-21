import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from braindecode_features.domains import *
from braindecode_features.utils import _initialize_windowing_fn


log = logging.getLogger(__name__)


def extract_ds_features(
        ds, frequency_bands, windowing_params=None, funcs_params=None,
        include=None, exclude=None, n_jobs=1):
    """Extract features from a braindecode BaseConcatDataset of WindowsDataset.

    Parameters
    ----------
    ds: BaseConcatDataset of BaseDataset
        Braindecode dataset to be used for feature extraction.
    frequency_bands: list(tuple(int, int))
        A list of frequency bands of prefiltered signals.
    windowing_params: None | dict
        Braindecode windowing function arguments
    funcs_params: None | dict
        {'Time__median__axis': -1}
    include: list | None
        Names of feature domains or feature functions to include, e.g.,
        'Time', 'Fourier__power'
    exclude: list | None
        Names of feature domains or feature functions to exclude.
    n_jobs: int
        Number of processes used for parallelization.

    Returns
    -------
    df: `pd.DataFrame`
        The final feature DataFrame holding all features, target information and
        feature name annotations.
    """
    feature_functions, extraction_routines = \
        _get_feature_functions_and_extraction_routines(
            include=include, exclude=exclude
        )
    if funcs_params is not None:
        funcs_params = _params_to_domain_params(params=funcs_params)
    has_events = len(ds.datasets[0].raw.annotations)
    if windowing_params is None:
        windowing_params = {}
    windowing_params.update({'n_jobs': n_jobs})
    windowing_fn = _initialize_windowing_fn(has_events, windowing_params)
    log.debug(f'got {len(ds.datasets)} datasets')
    domain_dfs = {}
    # extract features by domain, since each domain has it's very own routine
    for domain in extraction_routines.keys():
        # Do not extract cross-frequency features if there is only one band
        if len(frequency_bands) == 1 and domain == 'Cross-frequency':
            continue
        log.info(f'Computing features of domain: {domain}.')
        transformer_list = _build_transformer_list(feature_functions[domain])
        fu = FeatureUnion(
            transformer_list=transformer_list,
            n_jobs=n_jobs,
        )
        # set params
        if funcs_params is not None and domain in funcs_params:
            fu.set_params(**funcs_params[domain])
        # extract features of one domain at a time
        domain_dfs[domain] = extraction_routines[domain](
            concat_ds=ds,
            frequency_bands=frequency_bands,
            fu=fu,
            windowing_fn=windowing_fn,
        )
    # concatenate domain dfs and make final df pretty
    df = _finalize_df(
        dfs=list(domain_dfs.values()),
    )
    return df


def _build_transformer_list(funcs):
    return [(func.func.__name__, func) for func in funcs]


def _params_to_domain_params(params):
    """Expect params as {domain__func__param: value, ...} and
    convert to: {domain: {func__kw_args: {param: value}}}."""
    params_by_domain = {p.split('__')[0]: {} for p in params}
    for p, v in params.items():
        domain, func, param = p.split('__')
        if '__'.join([func, 'kw_args']) not in params_by_domain[domain]:
            params_by_domain[domain]['__'.join([func, 'kw_args'])] = {}
        params_by_domain[domain]['__'.join([func, 'kw_args'])].update(
            {param: v})
    return params_by_domain


def _merge_dfs(dfs, on):
    """Merge several a list of DataFrames on the specified columns."""
    df = dfs[0]
    for df_ in dfs[1:]:
        df = pd.merge(df_, df, on=on)
    return df


def _finalize_df(dfs):
    """Merge feature DataFrames returned by extraction routines to the final
    DataFrame. This means renaming columns, and creating readable MultiIndex.

    Returns
    -------
    df: `pd.DataFrame`
        The final feature DataFrame.
    """
    df = _merge_dfs(
        dfs=dfs,
        on=['i_trial', 'i_window_in_trial', 'target']
    )
    df = df.rename(
        mapper={
            'i_trial': 'Trial',
            'i_window_in_trial': 'Window',
            'target': 'Target',
        }, axis=1)
    df.columns = pd.MultiIndex.from_tuples(
        [col.split('__')
         if '__' in col else ['Description', col, '', '']
         for col in df.columns],
        names=['Domain', 'Feature', 'Channel', 'Frequency']
    )
    return df


# TODO: add restrictions to accepted shape in transform?
class _FunctionTransformer(FunctionTransformer):
    """Inspired by mne features. Wrap a feature function. Upon call of transform
    save the shape of the input data and output data. Implement a
    get_feature_names() method that returns a list of length corresponding to
    input channels.
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
        feature_names = [
            f'ch{i}-ch{j}'
            for i, j in zip(*np.triu_indices(chs_in, k=int(not include_diag)))]
        return feature_names

    def transform(self, X):
        # TODO: always expect 4d input?
        # TODO: always preserve input dimensions?
        # could probably remove the bins. could then use argmax of amplitudes instead of
        # bins at the position of argmax amplitudes for peak frequency
        # TODO: also, cross-frequency features get a tuple of twice the amount of data
        n_dim_in = X.ndim #if not isinstance(X, tuple) else X[0].ndim
        #assert n_dim_in == 4, f'Expected 4D input as n_bands x n_windows x n_channels x n_times. Got {X.shape}.'
        # expect input as (n_bands x n_windows x n_channels x n_times)
        #assert n_dim_in in [3, 4], f'Expected input data as (2 x) n_windows x n_channels x n_times. Got {n_dim_in}.'
        # cross-frequency data input
        #if n_dim_in == 4:
        #    assert X.shape[0] == 2, f'For cross-frequency features expect first dimension to be 2. Got {X.shape}'
        self.chs_in_ = X.shape[-2] #if not isinstance(X, tuple) else X[0].shape[-2]
        examples_in = X.shape[-3]
        X = super().transform(X=X)
        #if X.ndim != n_dim_in: #, f'Expected same number of dimension in input and output. Got {n_dim_in} and {X.ndim}.'
        #    X = np.expand_dims(X, axis=-1)
        # if there is more than one value per channel, last dimension will not be squeezed
        # then channels should be second last. if there is one value per channel, channels
        # are last dimension
        # TODO: could also try to always keep last dimension as '1', so channels will always be
        # second last dimension
        self.chs_out_ = X.shape[-1] if not n_dim_in == X.ndim else X.shape[-2]
        # TODO: make sure number of examples did not change?
        assert X.shape[-2] == examples_in, (
            f'Number of examples changed from {examples_in} to {X.shape[0]}.')
        return X


def _get_feature_functions(include=None, exclude=None):
    """Get feature extraction functions.

    Parameters
    ----------
    include: None | str
        The name of domains and / or features to include.
    exclude: None | str
        The name of domains and / or features to exclude.

    Returns
    -------
    dict
        Mapping of feature domain to feature extraction functions.
    """
    domain_func_getters = {
        'Time': get_time_feature_functions,
        'Fourier': get_fourier_feature_functions,
        'Wavelet': get_wavelet_feature_functions,
        'Cross-frequency': get_cross_frequency_feature_functions,
    }
    domains = list(domain_func_getters.keys())

    # remove domains or features according to exclude argument
    assert not (include is not None and exclude is not None)
    if include is not None:
        assert isinstance(include, list)
    if exclude is not None:
        assert isinstance(exclude, list)
    excludes = {domain: None for domain in domains}
    if exclude is not None:
        for ex in exclude:
            if '__' not in ex:
                if ex in domains:
                    _ = domain_func_getters.pop(ex)
            else:
                domain, feat = ex.split('__')
                if excludes[domain] is None:
                    excludes[domain] = []
                excludes[domain].append(feat)

    # add domains or features according to include argument
    includes = {}
    if include is not None:
        for incl in include:
            if '__' not in incl:
                domain = incl
                feat = None
                includes[domain] = feat
            else:
                domain, feat = incl.split('__')
                if domain not in includes:
                    includes[domain] = []
                includes[domain].append(feat)
        for domain in domains:
            if domain not in includes:
                _ = domain_func_getters.pop(domain)

    # return all the functions wrt include and exclude arguments
    feature_functions = {}
    for domain, func_getter in domain_func_getters.items():
        feature_functions.update({
            domain: [
                _FunctionTransformer(f)
                for f in func_getter(
                    include=includes[domain] if domain in includes else None,
                    exclude=excludes[domain] if domain in excludes else None,
                )
            ],
        })
    return feature_functions


def _get_extraction_routines(domain=None):
    """Get feature extraction routines.

    Parameters
    ----------
    domain: str | None
        The name of the domain. None corresponds to selecting all domains.

    Returns
    -------
    dict
        Mapping of feature domain to extraction routines.
    """
    extraction_routines = {
        'Time': extract_time_features,
        'Fourier': extract_fourier_features,
        'Wavelet': extract_wavelet_features,
        'Cross-frequency': extract_cross_frequency_features,
    }
    if domain is not None:
        extraction_routines = {domain: extraction_routines[domain]}
    return extraction_routines


def _get_feature_functions_and_extraction_routines(include=None, exclude=None):
    """Get feature functions and extraction routines of all or a single domain.

    Parameters
    ----------
    include: None | str
        The name of domains and / or features to include.
    exclude: None | str
        The name of domains and / or features to exclude.

    Returns
    -------
    tuple(dict(str: func), dict(str, func))
        Feature functions and extraction routines ordered by domain.
    """
    assert not (include is not None and exclude is not None)
    feature_funcs = _get_feature_functions(include=include, exclude=exclude)
    extraction_routines = {k: v
                           for domain in feature_funcs.keys()
                           for k, v in _get_extraction_routines(domain).items()}
    return feature_funcs, extraction_routines


'''In case we decide to create a FeatureDataset
def extract_ds_features(
        concat_ds, frequency_bands, windowing_params=None, params=None,
        out_dir=None, n_jobs=-1):
    """Extract features from a braindecode BaseConcatDataset of WindowsDataset.

    Parameters
    ----------
    concat_ds: BaseConcatDataset of BaseDataset
        Braindecode dataset to be used for feature extraction.
    frequency_bands: list(tuple(int, int))
        A list of frequency bands of prefiltered signals.
    windowing_params
        ...
    params
        ...
    n_jobs: int
        Number of processes used for parallelization.

    Returns
    -------
    df: `pd.DataFrame`
        The final feature DataFrame holding all features, target information and
        feature name annotations.
    """
    assert hasattr(concat_ds.datasets[0], 'raw'), 'Expecting unwindowed data.'
    feature_functions, extraction_routines = _get_feature_functions_and_extraction_routines()
    # if domains is not None:
    #    feature_functions = {domain: feature_functions[domain] for domain in domains}
    #    extraction_routines = {domain: extraction_routines[domain] for domain in domains}
    if params is not None:
        params = _params_to_domain_params(params=params)
    has_events = len(concat_ds.datasets[0].raw.annotations)
    windowing_fn = _initialize_windowing_fn(has_events, windowing_params)
    log.debug(f'got {len(concat_ds.datasets)} datasets')
    all_dfs = []
    for i in tqdm(range(len(concat_ds.datasets))):
        one_concat_ds = concat_ds.split([i])['0']
        domain_dfs = {}
        # extract features by domain, since each domain has it's very own routine
        for domain in extraction_routines.keys():
            # Do not extract cross-frequency features if there is only one band
            if len(frequency_bands) == 1 and domain == 'Cross-frequency':
                continue
            log.debug(f'Computing features of domain: {domain}.')
            transformer_list = _build_transformer_list(
                feature_functions[domain])
            fu = FeatureUnion(
                transformer_list=transformer_list,
                n_jobs=n_jobs,
            )
            # set params
            if params is not None and domain in params:
                fu.set_params(**params[domain])
            # extract features of one domain at a time
            domain_dfs[domain] = extraction_routines[domain](
                concat_ds=one_concat_ds,
                frequency_bands=frequency_bands,
                fu=fu,
                windowing_fn=windowing_fn,
            )
        # concatenate domain dfs and make final df pretty
        df = _finalize_df(
            dfs=list(domain_dfs.values()),
        )
        # account for the position of the dataset in the concat
        df[_find_col(df, 'Trial')] += i
        # overwrite datasets in the concat to mimic inplace operation
        assert len(one_concat_ds.datasets) == 1
        concat_ds.datasets[i] = FeatureDataset(
            feature_df=df,
            description=one_concat_ds.datasets[0].description,
            target_name=one_concat_ds.datasets[0].target_name,
        )
        if out_dir is not None:
            concat_ds.save(
                path=out_dir,
            )
    # re-compute cumulative sizes for iterating to work
    concat_ds.cumulative_sizes = concat_ds.cumsum(concat_ds.datasets)
'''
