"""E3 -- Repeated-split evaluation, corrected confidence intervals and paired
significance tests.

Editor #3, Reviewer 1 #4, Reviewer 3:
  (a) the v1 interval  mean +- 1.96 SD  is not a confidence interval for the
      mean; it is a (normal-approximation) prediction interval for a single
      fold.  The corrected interval  mean +- t_{0.975,J-1} SD/sqrt(J)  is
      reported alongside it so the difference is explicit;
  (b) resampled folds/splits share training data, so the ordinary paired t
      statistic is anti-conservative.  The Nadeau & Bengio (2003) corrected
      resampled t-test is therefore used as the primary test, with the
      uncorrected t and the Wilcoxon signed-rank test reported for reference
      (Dietterich, 1998);
  (c) all models are compared on IDENTICAL splits so the comparison is paired.
"""
import json, os, time, warnings
import numpy as np
from scipy import stats
from scipy.stats import loguniform
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import common

warnings.filterwarnings('ignore')

N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

SVR_GRID = {'svr__C': loguniform(0.1, 300),
            'svr__epsilon': loguniform(0.01, 1.0),
            'svr__gamma': loguniform(1e-3, 1.0)}
XGB_GRID = {'m__n_estimators': [200, 400, 600],
            'm__learning_rate': loguniform(0.01, 0.3),
            'm__max_depth': [3, 4, 6, 8, 10],
            'm__subsample': [0.7, 0.85, 1.0],
            'm__colsample_bytree': [0.7, 0.85, 1.0]}
RF_GRID = {'m__n_estimators': [200, 400],
           'm__max_depth': [10, 20, 30, None],
           'm__min_samples_leaf': [1, 2, 4],
           'm__max_features': ['sqrt', 0.5, 1.0]}

# label -> (scaling, feature set, n_train, tuned)
SVR_CONFIGS = [
    ('SVR uniform / raw8 / default / n=3000',    'uniform', 'raw8',  3000, False),
    ('SVR manual / raw8 / default / n=3000',     'manual',  'raw8',  3000, False),
    ('SVR uniform / top12 / tuned / full',       'uniform', 'top12', None, True),
    ('SVR manual / top12 / tuned / full',        'manual',  'top12', None, True),
    ('SVR auto / top12 / tuned / full',          'auto',    'top12', None, True),
]


def metrics(yte, pred):
    return dict(r2=float(r2_score(yte, pred)),
                rmse=float(np.sqrt(mean_squared_error(yte, pred))),
                mae=float(mean_absolute_error(yte, pred)))


df = common.load()
per_seed, selected = {}, {}

# Per-seed checkpointing: a seed is expensive (SVR tuned on the full training
# set), so each completed seed is flushed to disk and a restart resumes from
# there rather than repeating work.
CKPT = 'results/e3_checkpoint.json'
done_seeds = set()
if os.path.exists(CKPT):
    _c = json.load(open(CKPT))
    per_seed = _c['per_seed']
    selected = {int(k): v for k, v in _c['selected'].items()}
    done_seeds = set(_c['done_seeds'])
    print('resuming from checkpoint; seeds already complete: %s'
          % sorted(done_seeds), flush=True)

for seed in SEEDS:
    if seed in done_seeds:
        continue
    Xtr, Xte, ytr, yte = common.split(df, seed=seed)
    top12 = common.select_top_k(Xtr, ytr, 12, seed=seed)
    selected[seed] = top12
    fsets = {'raw8': common.RAW, 'top12': top12}

    for label, scaling, fkey, n, tuned in SVR_CONFIGS:
        feats = list(fsets[fkey])
        Xs, ys = common.subsample(Xtr[feats], ytr, n, seed=seed)
        t = time.time()
        pipe = common.make_pipeline(feats, scaling=scaling)
        if tuned:
            s = RandomizedSearchCV(pipe, SVR_GRID, n_iter=12, cv=3, scoring='r2',
                                   random_state=seed, n_jobs=-1).fit(Xs, ys)
            model, best = s.best_estimator_, {k: float(v) for k, v in s.best_params_.items()}
        else:
            model, best = pipe.fit(Xs, ys), {}
        row = metrics(yte, model.predict(Xte[feats]))
        row.update(seed=seed, n_train=len(Xs), best_params=best,
                   seconds=round(time.time() - t, 1))
        per_seed.setdefault(label, []).append(row)
        print('seed %2d  %-40s R2=%7.4f  [%.0fs]' % (seed, label, row['r2'], row['seconds']), flush=True)

    for label, est, grid in [('XGBoost tuned / top12 / full', XGBRegressor(random_state=seed, n_jobs=1, verbosity=0), XGB_GRID),
                             ('RandomForest tuned / top12 / full', RandomForestRegressor(random_state=seed, n_jobs=1), RF_GRID)]:
        t = time.time()
        pipe = Pipeline([('scaler', common.uniform_scaler(top12)), ('m', est)])
        s = RandomizedSearchCV(pipe, grid, n_iter=12, cv=3, scoring='r2',
                               random_state=seed, n_jobs=-1).fit(Xtr[top12], ytr)
        row = metrics(yte, s.best_estimator_.predict(Xte[top12]))
        row.update(seed=seed, n_train=len(Xtr),
                   best_params={k: str(v) for k, v in s.best_params_.items()},
                   seconds=round(time.time() - t, 1))
        per_seed.setdefault(label, []).append(row)
        print('seed %2d  %-40s R2=%7.4f  [%.0fs]' % (seed, label, row['r2'], row['seconds']), flush=True)

    done_seeds.add(seed)
    json.dump(dict(per_seed=per_seed,
                   selected={str(k): v for k, v in selected.items()},
                   done_seeds=sorted(done_seeds)), open(CKPT, 'w'))
    print('   [checkpoint written: %d/%d seeds done]'
          % (len(done_seeds), len(SEEDS)), flush=True)

# ---------------------------------------------------------------- summaries
N_TEST = int(round(len(df) * common.TEST_SIZE))
N_TRAIN = len(df) - N_TEST
RHO = N_TEST / N_TRAIN            # Nadeau-Bengio correction factor


def summarise(rows):
    r2 = np.array([r['r2'] for r in rows]); J = len(r2)
    sd = float(r2.std(ddof=1)); mean = float(r2.mean())
    t_crit = stats.t.ppf(0.975, J - 1)
    se = sd / np.sqrt(J)
    se_nb = sd * np.sqrt(1.0 / J + RHO)          # Nadeau-Bengio corrected SE
    return dict(
        mean_r2=round(mean, 4), sd_r2=round(sd, 4), n_splits=J,
        ci95_correct=[round(mean - t_crit * se, 4), round(mean + t_crit * se, 4)],
        ci95_nadeau_bengio=[round(mean - t_crit * se_nb, 4), round(mean + t_crit * se_nb, 4)],
        ci95_v1_incorrect=[round(mean - 1.96 * sd, 4), round(mean + 1.96 * sd, 4)],
        min_r2=round(float(r2.min()), 4), max_r2=round(float(r2.max()), 4),
        mean_rmse=round(float(np.mean([r['rmse'] for r in rows])), 4),
        mean_mae=round(float(np.mean([r['mae'] for r in rows])), 4))


summary = {k: summarise(v) for k, v in per_seed.items()}

labels = list(per_seed)
tests = {}
for i, a in enumerate(labels):
    for b in labels[i + 1:]:
        da = np.array([r['r2'] for r in per_seed[a]])
        db = np.array([r['r2'] for r in per_seed[b]])
        d = da - db; J = len(d)
        sd = d.std(ddof=1)
        t_plain, p_plain = stats.ttest_rel(da, db)
        t_nb = d.mean() / (sd * np.sqrt(1.0 / J + RHO) + 1e-12)
        p_nb = 2 * (1 - stats.t.cdf(abs(t_nb), J - 1))
        w_stat, p_w = stats.wilcoxon(da, db)
        t_crit = stats.t.ppf(0.975, J - 1)
        half = t_crit * sd * np.sqrt(1.0 / J + RHO)
        tests[f'{a}  vs  {b}'] = dict(
            mean_diff=round(float(d.mean()), 4), sd_diff=round(float(sd), 4),
            ci95_diff_nadeau_bengio=[round(float(d.mean() - half), 4),
                                     round(float(d.mean() + half), 4)],
            t_corrected=round(float(t_nb), 3), p_corrected=float(p_nb),
            t_uncorrected=round(float(t_plain), 3), p_uncorrected=float(p_plain),
            p_wilcoxon=float(p_w),
            significant_corrected=bool(p_nb < 0.05),
            wins=int((d > 0).sum()), losses=int((d < 0).sum()))

# ---- 10-fold CV on the v1 configuration, with the interval done both ways --
Xtr, Xte, ytr, yte = common.split(df, seed=common.SEED)
top12 = common.select_top_k(Xtr, ytr, 12)
Xs, ys = common.subsample(Xtr[top12], ytr, 3000)
cv_scores = cross_val_score(common.make_pipeline(top12, 'manual'), Xs, ys,
                            cv=KFold(10, shuffle=True, random_state=common.SEED),
                            scoring='r2', n_jobs=-1)
m, sd, J = float(cv_scores.mean()), float(cv_scores.std(ddof=1)), len(cv_scores)
t_crit = stats.t.ppf(0.975, J - 1)
cv_block = dict(
    per_fold=[round(float(s), 4) for s in cv_scores],
    mean=round(m, 4), sd=round(sd, 4),
    ci95_v1_incorrect=[round(m - 1.96 * sd, 4), round(m + 1.96 * sd, 4)],
    ci95_correct=[round(m - t_crit * sd / np.sqrt(J), 4), round(m + t_crit * sd / np.sqrt(J), 4)],
    note='v1 reported mean +- 1.96 SD, which is a prediction interval for one '
         'fold, not a confidence interval for the mean.')

out = dict(n_splits=N_SEEDS, rho_nadeau_bengio=round(RHO, 4),
           n_train=N_TRAIN, n_test=N_TEST,
           summary=summary, paired_tests=tests, cv_interval=cv_block,
           selected_per_seed={str(k): v for k, v in selected.items()},
           feature_selection_stability=dict(sorted(
               {f: sum(f in v for v in selected.values()) for v in selected.values() for f in v}.items(),
               key=lambda kv: -kv[1])),
           per_seed=per_seed)
json.dump(out, open('results/e3_statistics.json', 'w'), indent=2)

print('\n--- mean R2 over %d splits ---' % N_SEEDS)
for k, s in summary.items():
    print('%-40s %.4f +- %.4f   correct CI %s   v1-style CI %s'
          % (k, s['mean_r2'], s['sd_r2'], s['ci95_correct'], s['ci95_v1_incorrect']))
print('\n--- paired tests (Nadeau-Bengio corrected) ---')
for k, v in tests.items():
    print('%-86s diff=%+.4f  p_corr=%.4g  p_uncorr=%.4g  %s'
          % (k, v['mean_diff'], v['p_corrected'], v['p_uncorrected'],
             'SIG' if v['significant_corrected'] else 'ns'))
print('\n--- 10-fold CV interval ---'); print(cv_block)
print('wrote results/e3_statistics.json')
