"""Shared data, feature engineering and pipeline definitions for the
Discover AI major revision experiments.

Mirrors the original manuscript's configuration exactly so that new results
are comparable with the published ones, while exposing every component as a
switch so ablations can be run in any order/combination.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

SEED = 42
TEST_SIZE = 0.3
SVR_SUB = 3000

RAW = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
       'Population', 'AveOccup', 'Latitude', 'Longitude']

DERIVED = ['Income_per_Room', 'Room_Value_Score', 'Location_Score',
           'Coastal_Proximity', 'Bedroom_Ratio', 'Population_Density',
           'Age_Income_Interaction', 'Modernization_Score',
           'Rooms_per_Person', 'Income_Density']

ROBUST_COLS = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup',
               'Room_Value_Score', 'Population_Density', 'Income_Density']
MINMAX_COLS = ['Latitude', 'Longitude', 'HouseAge',
               'Location_Score', 'Coastal_Proximity']
STANDARD_COLS = ['MedInc', 'Income_per_Room', 'Age_Income_Interaction',
                 'Modernization_Score', 'Rooms_per_Person', 'Bedroom_Ratio']


def add_derived(df):
    df = df.copy()
    df['Income_per_Room'] = df['MedInc'] / (df['AveRooms'] + 1)
    df['Room_Value_Score'] = df['MedInc'] * df['AveRooms']
    df['Location_Score'] = (df['Latitude'] * df['Longitude']) / 1000
    df['Coastal_Proximity'] = (df['Latitude'] - 34.05).abs()
    df['Bedroom_Ratio'] = df['AveBedrms'] / (df['AveRooms'] + 1)
    df['Population_Density'] = df['Population'] / (df['AveOccup'] + 1)
    df['Age_Income_Interaction'] = df['HouseAge'] * df['MedInc']
    df['Modernization_Score'] = df['MedInc'] / (df['HouseAge'] + 1)
    df['Rooms_per_Person'] = df['AveRooms'] / (df['AveOccup'] + 1)
    df['Income_Density'] = (df['MedInc'] * df['Population']) / 1000
    return df


def load(with_derived=True):
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame.rename(columns={'MedHouseVal': 'y'})
    if with_derived:
        df = add_derived(df)
    return df


def split(df, seed=SEED, stratify_deciles=True):
    y = df['y']
    strat = pd.qcut(y, q=10, labels=False, duplicates='drop') if stratify_deciles else None
    cols = [c for c in df.columns if c != 'y']
    return train_test_split(df[cols], y, test_size=TEST_SIZE,
                            random_state=seed, stratify=strat)


def subsample(X, y, n=SVR_SUB, seed=SEED):
    if n is None or n >= len(X):
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[idx], y.iloc[idx]


def feature_specific_scaler(features):
    """The manuscript's scaling strategy: RobustScaler on heavy-tailed count
    features, MinMaxScaler on bounded geographic ones, StandardScaler on the
    rest."""
    pick = lambda lst: [c for c in lst if c in features]
    tf = []
    for name, scaler, cols in [('robust', RobustScaler(), pick(ROBUST_COLS)),
                               ('minmax', MinMaxScaler(), pick(MINMAX_COLS)),
                               ('standard', StandardScaler(), pick(STANDARD_COLS))]:
        if cols:
            tf.append((name, scaler, cols))
    rest = [c for c in features
            if c not in ROBUST_COLS + MINMAX_COLS + STANDARD_COLS]
    if rest:
        tf.append(('standard_rest', StandardScaler(), rest))
    return ColumnTransformer(tf)


def uniform_scaler(features):
    """Preethi et al.'s configuration as described: one scaler for every
    feature."""
    return ColumnTransformer([('standard', StandardScaler(), list(features))])


def build_pipeline(features, scaling='feature_specific', svr_params=None,
                   max_iter=2000):
    """scaling: 'none' | 'uniform' | 'feature_specific'"""
    steps = []
    if scaling == 'feature_specific':
        steps.append(('scaler', feature_specific_scaler(features)))
    elif scaling == 'uniform':
        steps.append(('scaler', uniform_scaler(features)))
    elif scaling != 'none':
        raise ValueError(scaling)
    params = dict(kernel='rbf', cache_size=500, max_iter=max_iter)
    params.update(svr_params or {})
    steps.append(('svr', SVR(**params)))
    return Pipeline(steps)


def ensemble_importance(X, y, seed=SEED, weights=(0.4, 0.3, 0.3)):
    """MI / |Pearson| / RF importance, min-max normalised then pooled."""
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor
    mi = mutual_info_regression(X, y, random_state=seed)
    cor = np.array([abs(np.corrcoef(X[c], y)[0, 1]) for c in X.columns])
    rf = RandomForestRegressor(n_estimators=100, random_state=seed,
                               n_jobs=-1).fit(X, y).feature_importances_
    norm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-12)
    w_mi, w_cor, w_rf = weights
    score = w_mi * norm(mi) + w_cor * norm(cor) + w_rf * norm(rf)
    return pd.Series(score, index=X.columns).sort_values(ascending=False)


def select_top_k(X, y, k=12, seed=SEED, weights=(0.4, 0.3, 0.3)):
    return list(ensemble_importance(X, y, seed, weights).head(k).index)


# =========================================================================
# Major-revision additions
# =========================================================================
# Editor #4/#7/#9, Reviewer 1 #7, Reviewer 2: the v1 scaler assignment was a
# hand-made table specific to California Housing, so it could be neither
# stated as a workflow nor tested on another dataset.  It is restated here as
# an explicit, data-driven rule that is fitted on training folds only.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# Rule thresholds (fixed a priori, sensitivity-tested in E7)
OUTLIER_FRAC_T = 0.05      # share of points beyond 1.5 IQR
SKEW_T = 2.0               # |skewness|
BOUNDED_SKEW_T = 0.5       # |skewness| for a "well-behaved bounded" feature


def assign_scalers(X, outlier_t=OUTLIER_FRAC_T, skew_t=SKEW_T,
                   bounded_t=BOUNDED_SKEW_T):
    """Distribution-aware scaler assignment.

    Heavy-tailed / outlier-dominated -> RobustScaler,
    bounded and near-symmetric      -> MinMaxScaler,
    otherwise                       -> StandardScaler.

    Returns {column: scaler_name}.  Computed from training data only.
    """
    out = {}
    for c in X.columns:
        v = pd.to_numeric(X[c], errors='coerce').dropna().values.astype(float)
        if v.size == 0 or np.allclose(v, v[0]):
            out[c] = 'standard'
            continue
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            frac = 0.0
        else:
            frac = float(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean())
        skew = float(pd.Series(v).skew())
        if frac > outlier_t or abs(skew) > skew_t:
            out[c] = 'robust'
        elif frac == 0.0 and abs(skew) < bounded_t:
            out[c] = 'minmax'
        else:
            out[c] = 'standard'
    return out


class DistributionAwareScaler(BaseEstimator, TransformerMixin):
    """ColumnTransformer-equivalent whose scaler-to-feature assignment is
    itself learned from the training fold, so the assignment rule cannot leak
    information from held-out data."""

    def __init__(self, outlier_t=OUTLIER_FRAC_T, skew_t=SKEW_T,
                 bounded_t=BOUNDED_SKEW_T):
        self.outlier_t = outlier_t
        self.skew_t = skew_t
        self.bounded_t = bounded_t

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.assignment_ = assign_scalers(X, self.outlier_t, self.skew_t,
                                          self.bounded_t)
        self.columns_ = list(X.columns)
        groups = {}
        for c, s in self.assignment_.items():
            groups.setdefault(s, []).append(c)
        makers = {'robust': RobustScaler, 'minmax': MinMaxScaler,
                  'standard': StandardScaler}
        self.scalers_ = {s: makers[s]().fit(X[cols]) for s, cols in groups.items()}
        self.groups_ = groups
        return self

    def transform(self, X):
        X = pd.DataFrame(X)[self.columns_]
        parts, names = [], []
        for s, cols in self.groups_.items():
            parts.append(self.scalers_[s].transform(X[cols]))
            names.extend(cols)
        Z = np.hstack(parts)
        order = [names.index(c) for c in self.columns_]
        return Z[:, order]


def make_scaler(scaling, features):
    """scaling: 'none' | 'uniform' | 'manual' | 'auto'."""
    if scaling == 'none':
        return None
    if scaling == 'uniform':
        return uniform_scaler(features)
    if scaling in ('manual', 'feature_specific'):
        return feature_specific_scaler(features)
    if scaling == 'auto':
        return DistributionAwareScaler()
    raise ValueError(scaling)


def make_pipeline(features, scaling='auto', svr_params=None, max_iter=-1,
                  estimator=None, impute=False):
    steps = []
    if impute:
        steps.append(('impute', SimpleImputer(strategy='median')))
    sc = make_scaler(scaling, features)
    if sc is not None:
        steps.append(('scaler', sc))
    if estimator is None:
        params = dict(kernel='rbf', cache_size=500, max_iter=max_iter)
        params.update(svr_params or {})
        estimator = SVR(**params)
        steps.append(('svr', estimator))
    else:
        steps.append(('m', estimator))
    return Pipeline(steps)


# ---- additional datasets (Editor #11, Reviewer 2) -------------------------

def _ames():
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=42165, as_frame=True, parser='auto').frame
    y = pd.to_numeric(d['SalePrice'], errors='coerce') / 1e5   # $100k, as California
    X = d.drop(columns=[c for c in ['SalePrice', 'Id'] if c in d.columns])
    X = X.select_dtypes(include=[np.number]).astype(float)
    X = X.loc[:, X.notna().mean() > 0.8]
    X = X.fillna(X.median())
    keep = y.notna()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def _kc_house():
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name='house_sales', version=3, as_frame=True, parser='auto').frame
    y = pd.to_numeric(d['price'], errors='coerce') / 1e5
    X = d.drop(columns=[c for c in ['price', 'id', 'date'] if c in d.columns])
    X = X.select_dtypes(include=[np.number]).astype(float)
    keep = y.notna()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def _california():
    df = load(with_derived=False)
    return df.drop(columns=['y']), df['y']


DATASETS = {'california': _california, 'kc_house': _kc_house, 'ames': _ames}


def load_dataset(name, n_max=None, seed=SEED):
    X, y = DATASETS[name]()
    if n_max is not None and len(X) > n_max:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), n_max, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X, y
