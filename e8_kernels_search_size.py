"""E8 -- Kernels, search strategy, training-set size and kernel approximation.

Reviewer 2 asked why only the RBF kernel's hyperparameters were optimised, why
C and epsilon were discretised inside a *randomised* search, why the search was
not exhaustive if it takes seconds, and why SVR was fitted to 3,000 rows at
all.  Reviewer 1 #3 asked for an approximate-kernel fit on the full training
set.  Reviewer 3 asked for the untested claim that more data lowers RMSE to be
either tested or dropped.  All four are answered here.
"""
import json, time, warnings
import numpy as np
from scipy.stats import loguniform
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
import common

warnings.filterwarnings('ignore')

df = common.load()
Xtr, Xte, ytr, yte = common.split(df)
top12 = common.select_top_k(Xtr, ytr, 12)
Xtr, Xte = Xtr[top12], Xte[top12]

out = {}

# ---- (a) every kernel tuned over its own hyperparameters -------------------
KERNELS = {
    'rbf':    (dict(kernel='rbf'),            {'svr__C': loguniform(0.1, 300), 'svr__epsilon': loguniform(0.01, 1.0), 'svr__gamma': loguniform(1e-3, 1.0)}),
    'linear': (dict(kernel='linear', max_iter=500000), {'svr__C': loguniform(0.01, 100), 'svr__epsilon': loguniform(0.01, 1.0)}),
    'poly2':  (dict(kernel='poly', degree=2, max_iter=500000), {'svr__C': loguniform(0.01, 100), 'svr__epsilon': loguniform(0.01, 1.0), 'svr__gamma': loguniform(1e-3, 1.0), 'svr__coef0': [0.0, 1.0]}),
    'poly3':  (dict(kernel='poly', degree=3, max_iter=500000), {'svr__C': loguniform(0.01, 100), 'svr__epsilon': loguniform(0.01, 1.0), 'svr__gamma': loguniform(1e-3, 1.0), 'svr__coef0': [0.0, 1.0]}),
    'sigmoid':(dict(kernel='sigmoid', max_iter=500000), {'svr__C': loguniform(0.01, 100), 'svr__epsilon': loguniform(0.01, 1.0), 'svr__gamma': loguniform(1e-3, 1.0)}),
}
kernels = {}
for name, (params, grid) in KERNELS.items():
    t = time.time()
    pipe = Pipeline([('scaler', common.uniform_scaler(top12)),
                     ('svr', SVR(cache_size=500, **params))])
    s = RandomizedSearchCV(pipe, grid, n_iter=20, cv=3, scoring='r2',
                           random_state=common.SEED, n_jobs=-1).fit(Xtr, ytr)
    pred = s.best_estimator_.predict(Xte)
    kernels[name] = dict(r2=round(float(r2_score(yte, pred)), 4),
                         rmse=round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
                         best_params={k: str(v) for k, v in s.best_params_.items()},
                         seconds=round(time.time() - t, 1))
    print('kernel %-8s R2=%8.4f  [%.0fs]' % (name, kernels[name]['r2'], kernels[name]['seconds']), flush=True)
out['kernels'] = kernels

# ---- (b) search strategy: exhaustive grid vs randomised search -------------
GRID = {'svr__C': [0.1, 1, 10, 100], 'svr__epsilon': [0.01, 0.1, 0.5, 1.0],
        'svr__gamma': ['scale', 'auto', 0.1, 1]}
CONT = {'svr__C': loguniform(0.1, 300), 'svr__epsilon': loguniform(0.01, 1.0),
        'svr__gamma': loguniform(1e-3, 1.0)}
search = {}
def _record(name, s, t):
    pred = s.best_estimator_.predict(Xte)
    search[name] = dict(r2=round(float(r2_score(yte, pred)), 4),
                        cv_best=round(float(s.best_score_), 4),
                        best_params={k: str(v) for k, v in s.best_params_.items()},
                        n_fits=len(s.cv_results_['params']) * 3,
                        seconds=round(time.time() - t, 1))
    print('search %-22s R2=%8.4f  fits=%d  [%.0fs]'
          % (name, search[name]['r2'], search[name]['n_fits'], search[name]['seconds']), flush=True)

t = time.time()
_record('grid_64', GridSearchCV(common.make_pipeline(top12, 'uniform'), GRID, cv=3,
                                scoring='r2', n_jobs=-1).fit(Xtr, ytr), t)
for n_iter in [20, 60, 200]:
    t = time.time()
    _record('random_%d' % n_iter,
            RandomizedSearchCV(common.make_pipeline(top12, 'uniform'), CONT, n_iter=n_iter,
                               cv=3, scoring='r2', random_state=common.SEED,
                               n_jobs=-1).fit(Xtr, ytr), t)
out['search'] = search

# ---- (c) learning curve: does more training data help? --------------------
best = {k.replace('svr__', ''): v for k, v in
        RandomizedSearchCV(common.make_pipeline(top12, 'uniform'), CONT, n_iter=40, cv=3,
                           scoring='r2', random_state=common.SEED,
                           n_jobs=-1).fit(Xtr, ytr).best_params_.items()}
sizes = {}
for n in [1000, 2000, 3000, 6000, 10000, len(Xtr)]:
    Xs, ys = common.subsample(Xtr, ytr, n)
    t = time.time()
    pipe = common.make_pipeline(top12, 'uniform', svr_params=best).fit(Xs, ys)
    fit_s = time.time() - t
    pred = pipe.predict(Xte)
    sizes[n] = dict(r2=round(float(r2_score(yte, pred)), 4),
                    rmse=round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
                    n_support=int(pipe.named_steps['svr'].n_support_.sum()),
                    fit_seconds=round(fit_s, 1))
    print('n=%6d  R2=%.4f  RMSE=%.4f  SV=%d  [%.1fs]'
          % (n, sizes[n]['r2'], sizes[n]['rmse'], sizes[n]['n_support'], fit_s), flush=True)
out['training_size'] = dict(best_params={k: float(v) for k, v in best.items()}, curve=sizes)

# ---- (d) Nystroem kernel approximation on the full training set -----------
approx = {}
for m in [200, 500, 1000, 2000]:
    t = time.time()
    pipe = Pipeline([('scaler', common.uniform_scaler(top12)),
                     ('nys', Nystroem(gamma=best.get('gamma', 0.1), n_components=m,
                                      random_state=common.SEED)),
                     ('lin', LinearSVR(C=best.get('C', 10), epsilon=best.get('epsilon', 0.1),
                                       max_iter=20000, random_state=common.SEED))]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    approx['nystroem_%d' % m] = dict(r2=round(float(r2_score(yte, pred)), 4),
                                     rmse=round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
                                     seconds=round(time.time() - t, 1))
    print('nystroem m=%4d  R2=%.4f  [%.0fs]' % (m, approx['nystroem_%d' % m]['r2'],
                                                approx['nystroem_%d' % m]['seconds']), flush=True)
out['kernel_approximation'] = approx

json.dump(out, open('results/e8_kernels_search_size.json', 'w'), indent=2)
print('wrote results/e8_kernels_search_size.json')
