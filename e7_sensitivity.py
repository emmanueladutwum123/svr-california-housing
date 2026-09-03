"""E7 -- Sensitivity of the two hand-chosen constants and of the scaler rule.

Reviewer 1 #7: the 40/30/30 ensemble-importance weights, the choice of k = 12,
and (new in this revision) the thresholds of the scaler-assignment rule were
all fixed by hand.  Each is swept here and the effect on the selected feature
set and on test R2 is reported.
"""
import json, time, warnings, itertools
import numpy as np
from sklearn.metrics import r2_score
import common

warnings.filterwarnings('ignore')

df = common.load()
Xtr, Xte, ytr, yte = common.split(df)
REF = common.select_top_k(Xtr, ytr, 12)


def score(feats, scaling='uniform'):
    pipe = common.make_pipeline(list(feats), scaling=scaling).fit(Xtr[list(feats)], ytr)
    return float(r2_score(yte, pipe.predict(Xte[list(feats)])))


# ---- (a) ensemble weights over the simplex in steps of 0.1 -----------------
weights = []
t0 = time.time()
for i in range(11):
    for j in range(11 - i):
        w = (i / 10, j / 10, (10 - i - j) / 10)
        feats = common.select_top_k(Xtr, ytr, 12, weights=w)
        weights.append(dict(w_mi=w[0], w_pearson=w[1], w_rf=w[2],
                            overlap_with_default=len(set(feats) & set(REF)),
                            r2=round(score(feats), 4), features=feats))
    print('weights row %d done [%.0fs]' % (i, time.time() - t0), flush=True)
r2s = np.array([w['r2'] for w in weights])
w_summary = dict(n=len(weights), min=float(r2s.min()), max=float(r2s.max()),
                 mean=round(float(r2s.mean()), 4), sd=round(float(r2s.std(ddof=1)), 4),
                 range=round(float(r2s.max() - r2s.min()), 4),
                 default_r2=round(score(REF), 4),
                 min_overlap=int(min(w['overlap_with_default'] for w in weights)))
print('weights sensitivity:', w_summary, flush=True)

# ---- (b) number of selected features --------------------------------------
ks = {}
for k in [4, 6, 8, 10, 12, 14, 16, 18]:
    feats = common.select_top_k(Xtr, ytr, k)
    ks[k] = dict(r2_uniform=round(score(feats), 4), r2_auto=round(score(feats, 'auto'), 4),
                 features=feats)
    print('k=%2d  uniform R2=%.4f  auto R2=%.4f' % (k, ks[k]['r2_uniform'], ks[k]['r2_auto']), flush=True)

# ---- (c) thresholds of the scaler-assignment rule --------------------------
thr = {}
for out_t, skew_t in itertools.product([0.01, 0.05, 0.10], [1.0, 2.0, 3.0]):
    sc = common.DistributionAwareScaler(outlier_t=out_t, skew_t=skew_t)
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR
    pipe = Pipeline([('scaler', sc), ('svr', SVR(kernel='rbf', cache_size=500))])
    pipe.fit(Xtr[REF], ytr)
    assign = common.assign_scalers(Xtr[REF], outlier_t=out_t, skew_t=skew_t)
    thr['out=%.2f|skew=%.1f' % (out_t, skew_t)] = dict(
        r2=round(float(r2_score(yte, pipe.predict(Xte[REF]))), 4),
        n_robust=sum(v == 'robust' for v in assign.values()),
        n_minmax=sum(v == 'minmax' for v in assign.values()),
        n_standard=sum(v == 'standard' for v in assign.values()))
    print('out=%.2f skew=%.1f  R2=%.4f  %s'
          % (out_t, skew_t, thr['out=%.2f|skew=%.1f' % (out_t, skew_t)]['r2'],
             {k2: v for k2, v in thr['out=%.2f|skew=%.1f' % (out_t, skew_t)].items() if k2 != 'r2'}), flush=True)

json.dump(dict(weights=weights, weights_summary=w_summary, k_sweep=ks,
               scaler_thresholds=thr, reference_top12=REF),
          open('results/e7_sensitivity.json', 'w'), indent=2)
print('wrote results/e7_sensitivity.json')
