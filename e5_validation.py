"""E5 -- Spatial validation and feature selection inside the resampling loop.

Editor #5, Reviewer 1 #6 and #8:
  (a) California Housing rows are census block groups, so a random split puts
      spatially adjacent -- and therefore strongly dependent -- block groups on
      both sides of the partition.  Spatially blocked cross-validation
      (Roberts et al., 2017) is run alongside random cross-validation and the
      difference is reported as the spatial optimism of the random estimate.
  (b) v1 selected the top-12 features once, on the whole training set, and then
      cross-validated.  Selection is repeated inside every fold here, and the
      difference is reported as the selection optimism.
  (c) a fully nested cross-validation (outer folds for evaluation, inner folds
      for hyperparameter search) gives the unbiased estimate for both
      partitioning schemes.
"""
import json, time, warnings
import numpy as np
from scipy.stats import loguniform
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import common

warnings.filterwarnings('ignore')

N_FOLDS = 10
N_OUTER = 5
N_JOBS = 6            # explicit: n_jobs=-1 collapsed to one loky worker under load
SVR_GRID = {'svr__C': loguniform(0.1, 300), 'svr__epsilon': loguniform(0.01, 1.0),
            'svr__gamma': loguniform(1e-3, 1.0)}

df = common.load()
X = df.drop(columns=['y']).reset_index(drop=True)
y = df['y'].reset_index(drop=True)

# ---- spatial blocks: k-means on the block-group centroids ------------------
coords = StandardScaler().fit_transform(df[['Latitude', 'Longitude']])
blocks = KMeans(n_clusters=N_FOLDS, random_state=common.SEED, n_init=10).fit_predict(coords)
blocks_outer = KMeans(n_clusters=N_OUTER, random_state=common.SEED, n_init=10).fit_predict(coords)
block_sizes = np.bincount(blocks).tolist()
print('spatial block sizes:', block_sizes, flush=True)

# features selected once on the whole data set -- the v1 (leaky) protocol
GLOBAL_TOP12 = common.select_top_k(X, y, 12)
print('global top-12:', GLOBAL_TOP12, flush=True)


def folds(scheme):
    if scheme == 'random':
        return list(KFold(N_FOLDS, shuffle=True, random_state=common.SEED).split(X))
    return list(GroupKFold(N_FOLDS).split(X, y, groups=blocks))


def run_cv(scheme, select_inside, scaling='auto', tuned=False):
    scores, sel_log = [], []
    for tr, te in folds(scheme):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]
        feats = common.select_top_k(Xtr, ytr, 12) if select_inside else GLOBAL_TOP12
        sel_log.append(feats)
        pipe = common.make_pipeline(feats, scaling=scaling)
        if tuned:
            pipe = RandomizedSearchCV(pipe, SVR_GRID, n_iter=20, cv=3, scoring='r2',
                                      random_state=common.SEED, n_jobs=N_JOBS)
        pipe.fit(Xtr[feats], ytr)
        scores.append(float(r2_score(yte, pipe.predict(Xte[feats]))))
        print('     %s fold %d/%d R2=%7.4f' % (scheme, len(scores), N_FOLDS, scores[-1]), flush=True)
    stability = float(np.mean([len(set(a) & set(b)) / 12
                               for i, a in enumerate(sel_log) for b in sel_log[i + 1:]]))
    return dict(scheme=scheme, select_inside=select_inside, tuned=tuned,
                per_fold=[round(s, 4) for s in scores],
                mean=round(float(np.mean(scores)), 4),
                sd=round(float(np.std(scores, ddof=1)), 4),
                min=round(float(np.min(scores)), 4), max=round(float(np.max(scores)), 4),
                mean_pairwise_selection_overlap=round(stability, 3))


results = {}
for scheme in ['random', 'spatial']:
    for inside in [False, True]:
        t = time.time()
        key = f'{scheme}|selection_{"inside" if inside else "outside"}|default'
        results[key] = run_cv(scheme, inside)
        print('%-46s mean R2=%7.4f  sd=%.4f  [%.0fs]'
              % (key, results[key]['mean'], results[key]['sd'], time.time() - t), flush=True)

# ---- fully nested CV: selection AND tuning inside every outer fold ---------
nested = {}
for scheme in ['random', 'spatial']:
    outer = (KFold(N_OUTER, shuffle=True, random_state=common.SEED).split(X)
             if scheme == 'random' else GroupKFold(N_OUTER).split(X, y, groups=blocks_outer))
    scores, params = [], []
    t = time.time()
    for i_fold, (tr, te) in enumerate(outer):
        tf = time.time()
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        feats = common.select_top_k(Xtr, ytr, 12)
        s = RandomizedSearchCV(common.make_pipeline(feats, scaling='auto'), SVR_GRID,
                               n_iter=20, cv=3, scoring='r2',
                               random_state=common.SEED, n_jobs=N_JOBS).fit(Xtr[feats], ytr)
        scores.append(float(r2_score(yte, s.best_estimator_.predict(Xte[feats]))))
        params.append({k: float(v) for k, v in s.best_params_.items()})
        print('   nested %-8s outer fold %d/%d  n_train=%d  R2=%7.4f  [%.0fs]'
              % (scheme, i_fold + 1, N_OUTER, len(tr), scores[-1], time.time() - tf), flush=True)
        json.dump({'scheme': scheme, 'folds_done': i_fold + 1, 'scores': scores},
                  open('results/e5_nested_checkpoint.json', 'w'), indent=2)
    nested[scheme] = dict(per_fold=[round(v, 4) for v in scores],
                          mean=round(float(np.mean(scores)), 4),
                          sd=round(float(np.std(scores, ddof=1)), 4),
                          best_params_per_fold=params,
                          seconds=round(time.time() - t, 1))
    print('nested %-8s mean R2=%7.4f  sd=%.4f  [%.0fs]'
          % (scheme, nested[scheme]['mean'], nested[scheme]['sd'], nested[scheme]['seconds']), flush=True)

optimism = dict(
    spatial_optimism=round(results['random|selection_inside|default']['mean']
                           - results['spatial|selection_inside|default']['mean'], 4),
    selection_optimism_random=round(results['random|selection_outside|default']['mean']
                                    - results['random|selection_inside|default']['mean'], 4),
    selection_optimism_spatial=round(results['spatial|selection_outside|default']['mean']
                                     - results['spatial|selection_inside|default']['mean'], 4),
    nested_spatial_gap=round(nested['random']['mean'] - nested['spatial']['mean'], 4))

json.dump(dict(block_sizes=block_sizes, outer_block_sizes=np.bincount(blocks_outer).tolist(), global_top12=GLOBAL_TOP12,
               cv=results, nested=nested, optimism=optimism),
          open('results/e5_validation.json', 'w'), indent=2)
print('\noptimism summary:', optimism)
print('wrote results/e5_validation.json')
