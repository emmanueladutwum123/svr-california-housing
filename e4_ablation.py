"""E4 -- Full-factorial ablation with order-independent attribution.

Reviewer 1 #5 / Editor #4: the published ablation walks a single path
(scaling -> features -> tuning), so its increments are path-dependent and must
not be read as independent contributions.  Here every cell of the
3 x 2 x 2 design is evaluated, all 6 component orderings are traced, and each
component is given an order-independent Shapley value.
"""
import json, math, time, itertools
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import common

df = common.load()
Xtr, Xte, ytr, yte = common.split(df)
top12 = common.select_top_k(Xtr, ytr, 12)

SCALINGS = ['none', 'uniform', 'feature_specific']
FEATURES = {'raw8': common.RAW, 'top12': top12}
TUNING = [False, True]

GRID = {'svr__C': [1, 10, 100],
        'svr__epsilon': [0.01, 0.1, 0.5],
        'svr__gamma': ['scale', 0.1]}

cells = {}
for scaling, (fname, feats), tuned in itertools.product(SCALINGS, FEATURES.items(), TUNING):
    key = f'{scaling}|{fname}|{"tuned" if tuned else "default"}'
    feats = list(feats)
    t = time.time()
    if tuned:
        search = GridSearchCV(common.build_pipeline(feats, scaling, max_iter=-1),
                              GRID, cv=3, scoring='r2', n_jobs=-1)
        search.fit(Xtr[feats], ytr)
        model, best = search.best_estimator_, search.best_params_
    else:
        model, best = common.build_pipeline(feats, scaling, max_iter=-1).fit(Xtr[feats], ytr), None
    r2 = float(r2_score(yte, model.predict(Xte[feats])))
    cells[key] = dict(r2=r2, best_params={k: str(v) for k, v in (best or {}).items()},
                      seconds=round(time.time() - t, 1))
    print('%-42s R2=%7.4f  [%.0fs]' % (key, r2, cells[key]['seconds']), flush=True)

# ---- order-dependent paths -------------------------------------------------
# Components: S (scaling, none->feature_specific), F (features, raw8->top12),
#             T (tuning, default->tuned).  Baseline cell = none|raw8|default.
def cell(state):
    return cells['%s|%s|%s' % ('feature_specific' if state['S'] else 'none',
                               'top12' if state['F'] else 'raw8',
                               'tuned' if state['T'] else 'default')]['r2']

paths = {}
for order in itertools.permutations('SFT'):
    state = dict(S=False, F=False, T=False)
    prev, incs = cell(state), {}
    for comp in order:
        state[comp] = True
        cur = cell(state)
        incs[comp] = round(cur - prev, 4)
        prev = cur
    paths['->'.join(order)] = incs

# ---- Shapley value per component ------------------------------------------
shapley = {}
for comp in 'SFT':
    others = [c for c in 'SFT' if c != comp]
    total, n = 0.0, 0
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            state = dict(S=False, F=False, T=False)
            for c in subset:
                state[c] = True
            without = cell(state)
            state[comp] = True
            weight = (math.factorial(r) * math.factorial(2 - r)) / math.factorial(3)
            total += weight * (cell(state) - without)
            n += 1
    shapley[comp] = round(total, 4)

out = dict(cells=cells, paths=paths, shapley=shapley,
           components=dict(S='feature-specific scaling vs none',
                           F='top-12 engineered set vs 8 raw',
                           T='grid-searched vs default hyperparameters'))
json.dump(out, open('results/e4_ablation.json', 'w'), indent=2)
print('\nPaths:'); [print(' ', k, v) for k, v in paths.items()]
print('Shapley:', shapley)
print('wrote results/e4_ablation.json')
