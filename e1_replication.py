"""E1 -- Direct reproduction of the Preethi et al. (2025) configuration.

Reviewer 1 #2, Reviewer 3, Editor #1: the manuscript's Stage A baseline is an
UNSCALED SVR (R2 = -0.054), but Preethi et al. describe uniform scaling and
report ~0.60.  The two are not the same configuration, so the -0.054 -> 0.690
jump cannot explain the published 0.60.  Here every configuration is run on
one identical split so the gap can be stated exactly.
"""
import json, time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import common

df = common.load()
Xtr, Xte, ytr, yte = common.split(df)
top12 = common.select_top_k(Xtr, ytr, 12)

CONFIGS = [
    # label, feature set, scaling, n_train
    ('A0  unscaled, 8 raw, default SVR (manuscript Stage A)', common.RAW, 'none', 3000),
    ('A1  unscaled, 8 raw, default SVR, full train',         common.RAW, 'none', None),
    ('P0  UNIFORM scaling, 8 raw, default SVR (Preethi config)', common.RAW, 'uniform', 3000),
    ('P1  UNIFORM scaling, 8 raw, default SVR, full train',      common.RAW, 'uniform', None),
    ('F0  feature-specific scaling, 8 raw, default SVR',     common.RAW, 'feature_specific', 3000),
    ('F1  feature-specific scaling, 8 raw, default, full',   common.RAW, 'feature_specific', None),
    ('U0  UNIFORM scaling, top-12, default SVR',             top12, 'uniform', 3000),
    ('U1  UNIFORM scaling, top-12, default, full train',     top12, 'uniform', None),
    ('S0  feature-specific, top-12, default SVR (Stage C)',  top12, 'feature_specific', 3000),
    ('S1  feature-specific, top-12, default, full train',    top12, 'feature_specific', None),
]

rows = []
for label, feats, scaling, n in CONFIGS:
    Xs, ys = common.subsample(Xtr[list(feats)], ytr, n)
    t = time.time()
    pipe = common.build_pipeline(list(feats), scaling=scaling, max_iter=-1).fit(Xs, ys)
    pred = pipe.predict(Xte[list(feats)])
    rows.append(dict(config=label, n_train=len(Xs), n_features=len(feats),
                     scaling=scaling,
                     r2=float(r2_score(yte, pred)),
                     rmse=float(np.sqrt(mean_squared_error(yte, pred))),
                     mae=float(mean_absolute_error(yte, pred)),
                     fit_seconds=round(time.time() - t, 1)))
    print('%-58s n=%5d  R2=%7.4f  RMSE=%.4f  [%.1fs]'
          % (label, len(Xs), rows[-1]['r2'], rows[-1]['rmse'], rows[-1]['fit_seconds']))

json.dump(rows, open('results/e1_replication.json', 'w'), indent=2)
print('\nwrote results/e1_replication.json')
