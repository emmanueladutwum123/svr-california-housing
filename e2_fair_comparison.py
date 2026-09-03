"""E2 -- Ten-model comparison with equal training data and equal tuning budget.

Reviewer 1 #3, Reviewer 2, Editor #2: in the published table SVR saw 3,000
rows while the other nine models saw 14,448, and only SVR was tuned.  Here
every model gets the same features, the same rows and the same search budget,
at both training sizes.
"""
import json, time
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import common

df = common.load()
Xtr, Xte, ytr, yte = common.split(df)
top12 = common.select_top_k(Xtr, ytr, 12)

MODELS = {
    'SVR-RBF':           (SVR(kernel='rbf', cache_size=500),
                          {'m__C': loguniform(0.1, 300), 'm__epsilon': loguniform(0.01, 1.0),
                           'm__gamma': loguniform(1e-3, 1.0)}),
    'SVR-Linear':        (SVR(kernel='linear', cache_size=500, max_iter=200000),
                          {'m__C': loguniform(0.01, 100), 'm__epsilon': loguniform(0.01, 1.0)}),
    'K-Nearest Neighbours': (KNeighborsRegressor(),
                          {'m__n_neighbors': randint(3, 50), 'm__weights': ['uniform', 'distance'],
                           'm__p': [1, 2]}),
    'Ridge Regression':  (Ridge(), {'m__alpha': loguniform(1e-3, 100)}),
    'Lasso Regression':  (Lasso(max_iter=10000), {'m__alpha': loguniform(1e-5, 1)}),
    'Linear Regression': (LinearRegression(), {}),
    'Decision Tree':     (DecisionTreeRegressor(random_state=common.SEED),
                          {'m__max_depth': randint(3, 30), 'm__min_samples_leaf': randint(1, 40),
                           'm__max_features': [None, 'sqrt', 0.6]}),
    'Random Forest':     (RandomForestRegressor(random_state=common.SEED, n_jobs=1),
                          {'m__n_estimators': randint(200, 600), 'm__max_depth': randint(5, 30),
                           'm__min_samples_leaf': randint(1, 10), 'm__max_features': ['sqrt', 0.5, 1.0]}),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=common.SEED),
                          {'m__n_estimators': randint(100, 500), 'm__learning_rate': loguniform(0.01, 0.3),
                           'm__max_depth': randint(2, 6), 'm__subsample': [0.7, 0.85, 1.0]}),
    'XGBoost':           (XGBRegressor(random_state=common.SEED, n_jobs=1, verbosity=0),
                          {'m__n_estimators': randint(200, 800), 'm__learning_rate': loguniform(0.01, 0.3),
                           'm__max_depth': randint(3, 10), 'm__subsample': [0.7, 0.85, 1.0],
                           'm__colsample_bytree': [0.7, 0.85, 1.0]}),
}

results = {}
for n_train in (3000, None):
    tag = f'n={n_train or len(Xtr)}'
    results[tag] = {}
    Xs, ys = common.subsample(Xtr[top12], ytr, n_train)
    for name, (est, grid) in MODELS.items():
        t = time.time()
        pipe = Pipeline([('scaler', common.uniform_scaler(top12)), ('m', est)])
        if grid:
            search = RandomizedSearchCV(pipe, grid, n_iter=20, cv=3, scoring='r2',
                                        random_state=common.SEED, n_jobs=-1)
            search.fit(Xs, ys)
            model, best = search.best_estimator_, search.best_params_
        else:
            model, best = pipe.fit(Xs, ys), {}
        pred = model.predict(Xte[top12])
        results[tag][name] = dict(
            r2=float(r2_score(yte, pred)),
            rmse=float(np.sqrt(mean_squared_error(yte, pred))),
            mae=float(mean_absolute_error(yte, pred)),
            best_params={k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                         for k, v in best.items()},
            seconds=round(time.time() - t, 1))
        print('%-8s %-22s R2=%7.4f  RMSE=%.4f  [%.0fs]'
              % (tag, name, results[tag][name]['r2'], results[tag][name]['rmse'],
                 results[tag][name]['seconds']), flush=True)

json.dump(results, open('results/e2_fair_comparison.json', 'w'), indent=2)
print('wrote results/e2_fair_comparison.json')
