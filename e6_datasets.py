"""E6 -- The same workflow on three tabular regression datasets.

Editor #11, Reviewer 2: v1 drew workflow-level conclusions from a single
benchmark.  The scaling / selection / tuning factorial is therefore repeated on
two further housing datasets, so it can be seen whether the ordering of the
components is a property of the workflow or of California Housing.

Datasets: California Housing (20,640 x 8, + 10 derived), King County house
sales (21,613 x 18) and Ames Housing (1,460 x ~36 numeric).  Nothing is
hand-tuned per dataset: the scaler assignment, the selection rule and the
search space are identical everywhere.
"""
import json, math, time, itertools, warnings
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import common

warnings.filterwarnings('ignore')

N_MAX_TRAIN = 10000     # identical budget for every cell of every dataset
GRID = {'svr__C': [0.1, 1, 10, 100],
        'svr__epsilon': [0.01, 0.1, 0.5, 1.0],
        'svr__gamma': ['scale', 'auto', 0.1, 1]}

SPECS = [('california', True), ('kc_house', False), ('ames', False)]

out = {}
for name, add_derived in SPECS:
    X, y = common.load_dataset(name)
    if add_derived:
        X = common.add_derived(X)
    strat = None
    try:
        import pandas as pd
        strat = pd.qcut(y, q=10, labels=False, duplicates='drop')
    except Exception:
        pass
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=common.TEST_SIZE,
                                          random_state=common.SEED, stratify=strat)
    Xtr, ytr = common.subsample(Xtr, ytr, N_MAX_TRAIN, seed=common.SEED)
    k = math.ceil(2 * X.shape[1] / 3)
    topk = common.select_top_k(Xtr, ytr, k)
    print('\n=== %s: %d train x %d features (top-%d selected) ==='
          % (name, len(Xtr), X.shape[1], k), flush=True)

    cells = {}
    for scaling, selected, tuned in itertools.product(
            ['none', 'uniform', 'auto'], [False, True], [False, True]):
        feats = topk if selected else list(X.columns)
        key = '%s|%s|%s' % (scaling, 'topk' if selected else 'all',
                            'tuned' if tuned else 'default')
        t = time.time()
        pipe = common.make_pipeline(feats, scaling=scaling)
        if tuned:
            s = GridSearchCV(pipe, GRID, cv=3, scoring='r2', n_jobs=-1).fit(Xtr[feats], ytr)
            model, best = s.best_estimator_, {a: str(b) for a, b in s.best_params_.items()}
        else:
            model, best = pipe.fit(Xtr[feats], ytr), {}
        pred = model.predict(Xte[feats])
        cells[key] = dict(r2=float(r2_score(yte, pred)),
                          rmse=float(np.sqrt(mean_squared_error(yte, pred))),
                          n_features=len(feats), best_params=best,
                          seconds=round(time.time() - t, 1))
        print('%-24s R2=%8.4f  [%.0fs]' % (key, cells[key]['r2'], cells[key]['seconds']), flush=True)

    def cell(st, scaler_on='auto'):
        return cells['%s|%s|%s' % (scaler_on if st['S'] else 'none',
                                   'topk' if st['K'] else 'all',
                                   'tuned' if st['T'] else 'default')]['r2']

    analysis = {}
    for scaler_on in ['uniform', 'auto']:
        comps = 'SKT'
        paths = {}
        for order in itertools.permutations(comps):
            st = {c: False for c in comps}
            prev, inc = cell(st, scaler_on), {}
            for c in order:
                st[c] = True
                cur = cell(st, scaler_on)
                inc[c] = round(cur - prev, 4)
                prev = cur
            paths['->'.join(order)] = inc
        shap = {}
        for c in comps:
            others = [o for o in comps if o != c]
            tot = 0.0
            for r in range(len(others) + 1):
                for sub in itertools.combinations(others, r):
                    st = {x: False for x in comps}
                    for x in sub:
                        st[x] = True
                    w0 = cell(st, scaler_on)
                    st[c] = True
                    w = math.factorial(r) * math.factorial(2 - r) / math.factorial(3)
                    tot += w * (cell(st, scaler_on) - w0)
            shap[c] = round(tot, 4)
        analysis[scaler_on] = dict(paths=paths, shapley=shap)

    out[name] = dict(n_train=len(Xtr), n_test=len(Xte), n_features=int(X.shape[1]),
                     k=k, selected=topk, cells=cells, analysis=analysis,
                     scaler_assignment=common.assign_scalers(Xtr[topk]))
    print('shapley (auto):', analysis['auto']['shapley'], flush=True)

json.dump(out, open('results/e6_datasets.json', 'w'), indent=2)
print('\nwrote results/e6_datasets.json')
