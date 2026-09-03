"""E1b -- Direct reproduction of the configuration described by Preethi et al.

Editor #1, Reviewer 1 #2, Reviewer 3.  Reading their Figs. 7 and 8, the SVR-RBF
result is R2 ~ 0.15 with MSE ~ 1.12; the ~0.60 quoted in v1 of this manuscript
is the score of their Linear / Ridge / Polynomial-Ridge / Elastic-Net models,
not of their SVR.  Their methodology section describes: load the dataset, split
into train and test, fit six regressors, tune SVR's C and epsilon with
GridSearchCV, and evaluate with KFold.  No feature scaling is described
anywhere in the paper.

This script runs that configuration exactly -- 80/20 split as implied by their
learning curves (which run to ~16,500 training rows), 8 raw features, RBF
kernel, gamma left at the library default, a grid over C and epsilon only --
with and without standardisation, over several seeds, and reports R2 and MSE
on the same footing as their figures.
"""
import json, time, warnings
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import common

warnings.filterwarnings('ignore')

SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2                      # implied by their learning curves
PREETHI_GRID = {'svr__C': [0.1, 1, 10, 100],
                'svr__epsilon': [0.01, 0.1, 0.2, 0.5]}   # gamma NOT tuned
REPORTED = dict(r2=0.15, mse=1.12, source='read from Figs. 7 and 8 of '
                'Preethi et al. (2025); the paper gives no numerical table')

df = common.load(with_derived=False)
X, y = df.drop(columns=['y']), df['y']

rows = []
for seed in SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    for scaling in ['none', 'uniform']:
        for tuned in [False, True]:
            t = time.time()
            pipe = common.make_pipeline(list(X.columns), scaling=scaling)
            if tuned:
                s = GridSearchCV(pipe, PREETHI_GRID, cv=5, scoring='r2', n_jobs=-1).fit(Xtr, ytr)
                model, best = s.best_estimator_, {k: float(v) for k, v in s.best_params_.items()}
            else:
                model, best = pipe.fit(Xtr, ytr), {}
            pred = model.predict(Xte)
            rows.append(dict(seed=seed, scaling=scaling, tuned=tuned,
                             r2=float(r2_score(yte, pred)),
                             mse=float(mean_squared_error(yte, pred)),
                             best_params=best, seconds=round(time.time() - t, 1)))
            print('seed %d  %-8s %-8s R2=%8.4f  MSE=%.4f  [%.0fs]'
                  % (seed, scaling, 'tuned' if tuned else 'default',
                     rows[-1]['r2'], rows[-1]['mse'], rows[-1]['seconds']), flush=True)

summary = {}
for scaling in ['none', 'uniform']:
    for tuned in [False, True]:
        sel = [r for r in rows if r['scaling'] == scaling and r['tuned'] == tuned]
        r2 = np.array([r['r2'] for r in sel]); mse = np.array([r['mse'] for r in sel])
        summary['%s|%s' % (scaling, 'tuned' if tuned else 'default')] = dict(
            mean_r2=round(float(r2.mean()), 4), sd_r2=round(float(r2.std(ddof=1)), 4),
            mean_mse=round(float(mse.mean()), 4), sd_mse=round(float(mse.std(ddof=1)), 4),
            gap_to_reported_r2=round(float(r2.mean() - REPORTED['r2']), 4))

json.dump(dict(reported_by_preethi=REPORTED, n_seeds=len(SEEDS),
               test_size=TEST_SIZE, grid=str(PREETHI_GRID),
               summary=summary, runs=rows),
          open('results/e1b_preethi_reproduction.json', 'w'), indent=2)
print('\n--- mean over %d seeds (their reported SVR: R2 ~ %.2f, MSE ~ %.2f) ---'
      % (len(SEEDS), REPORTED['r2'], REPORTED['mse']))
for k, v in summary.items():
    print('%-16s R2=%7.4f +- %.4f   MSE=%.4f   gap to reported %+0.4f'
          % (k, v['mean_r2'], v['sd_r2'], v['mean_mse'], v['gap_to_reported_r2']))
print('wrote results/e1b_preethi_reproduction.json')
