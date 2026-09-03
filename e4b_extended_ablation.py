"""E4b -- Four-factor ablation with order-independent attribution.

Editor #4, Reviewer 1 #5, Reviewer 2:  v1 walked a single path
(scaling -> features -> tuning) and read the increments as if they were
independent contributions.  Two things are wrong with that: the increments are
path-dependent, and "feature engineering" silently bundled two separate
operations (constructing derived features, and selecting a subset).

Design: S x D x K x T
  S  scaling            none | uniform | manual (v1 hand table) | auto (rule)
  D  derived features   absent | present
  K  feature selection  keep all | keep top ceil(2n/3) by ensemble importance
  T  hyperparameters    library defaults | grid search
= 4 x 2 x 2 x 2 = 32 cells, every cell trained on the full training set and
scored on the same held-out test set.  Shapley values are computed for three
different choices of which scaler counts as the treatment, because that choice
is itself an assumption.
"""
import json, math, time, itertools, warnings
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import common

warnings.filterwarnings('ignore')

GRID = {'svr__C': [0.1, 1, 10, 100],
        'svr__epsilon': [0.01, 0.1, 0.5, 1.0],
        'svr__gamma': ['scale', 'auto', 0.1, 1]}

df_raw = common.load(with_derived=False)
df_der = common.load(with_derived=True)
Xtr_r, Xte_r, ytr, yte = common.split(df_raw)
Xtr_d, Xte_d, _, _ = common.split(df_der)

sel_cache = {}
def features_for(derived, selected):
    """Feature set for a (D, K) combination.  Selection is always computed on
    the training split only."""
    Xtr = Xtr_d if derived else Xtr_r
    cols = list(Xtr.columns)
    if not selected:
        return cols
    key = derived
    if key not in sel_cache:
        k = math.ceil(2 * len(cols) / 3)
        sel_cache[key] = common.select_top_k(Xtr, ytr, k)
    return sel_cache[key]


cells = {}
for scaling, derived, selected, tuned in itertools.product(
        ['none', 'uniform', 'manual', 'auto'], [False, True], [False, True], [False, True]):
    key = '%s|%s|%s|%s' % (scaling, 'derived' if derived else 'raw',
                           'topk' if selected else 'all',
                           'tuned' if tuned else 'default')
    feats = features_for(derived, selected)
    Xtr = (Xtr_d if derived else Xtr_r)[feats]
    Xte = (Xte_d if derived else Xte_r)[feats]
    t = time.time()
    pipe = common.make_pipeline(feats, scaling=scaling)
    if tuned:
        s = GridSearchCV(pipe, GRID, cv=3, scoring='r2', n_jobs=-1).fit(Xtr, ytr)
        model, best = s.best_estimator_, {k2: str(v) for k2, v in s.best_params_.items()}
    else:
        model, best = pipe.fit(Xtr, ytr), {}
    pred = model.predict(Xte)
    cells[key] = dict(r2=float(r2_score(yte, pred)),
                      rmse=float(np.sqrt(mean_squared_error(yte, pred))),
                      mae=float(mean_absolute_error(yte, pred)),
                      n_features=len(feats), best_params=best,
                      seconds=round(time.time() - t, 1))
    print('%-34s R2=%7.4f  p=%2d  [%.0fs]'
          % (key, cells[key]['r2'], len(feats), cells[key]['seconds']), flush=True)


def analyse(scaler_on):
    """Order-dependent paths and Shapley values when S 'on' means scaler_on."""
    def cell(st):
        return cells['%s|%s|%s|%s' % (scaler_on if st['S'] else 'none',
                                      'derived' if st['D'] else 'raw',
                                      'topk' if st['K'] else 'all',
                                      'tuned' if st['T'] else 'default')]['r2']
    comps = 'SDKT'
    paths = {}
    for order in itertools.permutations(comps):
        st = {c: False for c in comps}
        prev, inc = cell(st), {}
        for c in order:
            st[c] = True
            cur = cell(st)
            inc[c] = round(cur - prev, 4)
            prev = cur
        paths['->'.join(order)] = inc
    shapley, spread = {}, {}
    for c in comps:
        others = [o for o in comps if o != c]
        total = 0.0
        n = len(comps)
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                st = {x: False for x in comps}
                for x in subset:
                    st[x] = True
                without = cell(st)
                st[c] = True
                w = math.factorial(r) * math.factorial(n - 1 - r) / math.factorial(n)
                total += w * (cell(st) - without)
        shapley[c] = round(total, 4)
        incs = [p[c] for p in paths.values()]
        spread[c] = dict(min=round(min(incs), 4), max=round(max(incs), 4),
                         range=round(max(incs) - min(incs), 4))
    return dict(paths=paths, shapley=shapley, increment_spread=spread,
                baseline_r2=round(cell({c: False for c in comps}), 4),
                full_r2=round(cell({c: True for c in comps}), 4))


analysis = {s: analyse(s) for s in ['uniform', 'manual', 'auto']}

out = dict(cells=cells, analysis=analysis,
           selected_features={('derived' if k else 'raw'): v for k, v in sel_cache.items()},
           components=dict(S='scaling applied vs none', D='derived features present vs absent',
                           K='top-ceil(2n/3) ensemble-importance selection vs all features',
                           T='grid-searched vs default hyperparameters'))
json.dump(out, open('results/e4b_extended_ablation.json', 'w'), indent=2)

for s, a in analysis.items():
    print('\n=== S = %s ===  baseline %.4f -> full %.4f' % (s, a['baseline_r2'], a['full_r2']))
    print('Shapley:', a['shapley'])
    print('increment range by order:', a['increment_spread'])
print('wrote results/e4b_extended_ablation.json')
