"""Generate every figure in the revised manuscript from results/*.json.

Nothing here is hand-drawn or hand-numbered: each figure reads the JSON written
by the corresponding experiment, so the figures cannot drift from the tables.
"""
import json, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common

warnings.filterwarnings('ignore')
OUT = 'paper/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, GREY = '#2F4F7F', '#D97706', '#8C8C8C'


def load(name):
    p = f'results/{name}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig)
    print('wrote', name)


# ---------------------------------------------------------------- fig: EDA
def fig_correlation():
    df = common.load(with_derived=False)
    c = df.corr()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=90)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f'{c.iloc[i, j]:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if abs(c.iloc[i, j]) > 0.5 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, 'fig_correlation')


def fig_importance():
    df = common.load()
    Xtr, _, ytr, _ = common.split(df)
    s = common.ensemble_importance(Xtr, ytr).sort_values()
    top12 = set(common.select_top_k(Xtr, ytr, 12))
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.barh(range(len(s)), s.values,
            color=[BLUE if i in top12 else GREY for i in s.index])
    ax.set_yticks(range(len(s))); ax.set_yticklabels([i.replace('_', ' ') for i in s.index])
    ax.set_xlabel('ensemble importance  (0.4 MI + 0.3 |Pearson| + 0.3 RF)')
    save(fig, 'fig_importance')


# ------------------------------------------------------- fig: reproduction
def fig_reproduction():
    d = load('e1b_preethi_reproduction')
    if not d:
        return
    order = ['none|default', 'none|tuned', 'uniform|default', 'uniform|tuned']
    labels = ['unscaled\ndefault', 'unscaled\nC,$\\varepsilon$ tuned',
              'standardised\ndefault', 'standardised\nC,$\\varepsilon$ tuned']
    vals = [d['summary'][k]['mean_r2'] for k in order]
    errs = [d['summary'][k]['sd_r2'] for k in order]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(labels, vals, yerr=errs, capsize=3,
           color=[GREY, GREY, BLUE, BLUE])
    ax.axhline(d['reported_by_preethi']['r2'], color=ORANGE, ls='--',
               label='SVR result reported by Preethi et al. ($R^2 \\approx %.2f$)'
                     % d['reported_by_preethi']['r2'])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylabel('test $R^2$'); ax.legend(frameon=False, fontsize=8, loc='upper left')
    save(fig, 'fig_reproduction')


# ----------------------------------------------------------- fig: ablation
def fig_ablation():
    d = load('e4b_extended_ablation')
    if not d:
        return
    a = d['analysis']['auto']
    comps = list(a['shapley'])
    names = {'S': 'scaling', 'D': 'derived\nfeatures', 'K': 'selection', 'T': 'tuning'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    lo = [a['increment_spread'][c]['min'] for c in comps]
    hi = [a['increment_spread'][c]['max'] for c in comps]
    y = np.arange(len(comps))
    ax1.hlines(y, lo, hi, color=GREY, lw=6, alpha=0.6)
    ax1.plot(lo, y, 'o', color=GREY, ms=4)
    ax1.plot(hi, y, 'o', color=GREY, ms=4)
    ax1.plot([a['shapley'][c] for c in comps], y, 'D', color=BLUE, ms=7,
             label='Shapley value')
    ax1.set_yticks(y); ax1.set_yticklabels([names[c] for c in comps])
    ax1.set_xlabel('$\\Delta R^2$ attributed to the component')
    ax1.set_title('range over the %d orderings' % len(a['paths']), fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    ax1.axvline(0, color='k', lw=0.5)

    cells = d['cells']
    scal = ['none', 'uniform', 'manual', 'auto']
    conf = [('raw', 'all', 'default'), ('derived', 'topk', 'default'),
            ('raw', 'all', 'tuned'), ('derived', 'topk', 'tuned')]
    w = 0.2
    for i, (dv, kv, tv) in enumerate(conf):
        vals = [cells[f'{s}|{dv}|{kv}|{tv}']['r2'] for s in scal]
        ax2.bar(np.arange(len(scal)) + (i - 1.5) * w, vals, w,
                label=f'{dv}/{kv}/{tv}')
    ax2.set_xticks(range(len(scal))); ax2.set_xticklabels(scal)
    ax2.set_ylabel('test $R^2$'); ax2.set_xlabel('scaling strategy')
    ax2.legend(frameon=False, fontsize=7); ax2.axhline(0, color='k', lw=0.5)
    save(fig, 'fig_ablation')


# ------------------------------------------------------ fig: paired tests
def fig_paired():
    d = load('e3_statistics')
    if not d:
        return
    keys = [k for k in d['paired_tests'] if 'SVR uniform / top12 / tuned / full' in k]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(keys) + 1.4))
    for i, k in enumerate(keys):
        v = d['paired_tests'][k]
        lo, hi = v['ci95_diff_nadeau_bengio']
        ax.plot([lo, hi], [i, i], color=GREY, lw=2)
        ax.plot(v['mean_diff'], i, 'D',
                color=BLUE if v['significant_corrected'] else ORANGE, ms=6)
    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.split('  vs  ')[1].replace(' / ', '/') for k in keys], fontsize=7)
    ax.set_xlabel('$\\Delta R^2$ vs SVR uniform/top12/tuned  '
                  '(95\\% Nadeau--Bengio interval)')
    save(fig, 'fig_paired')


# -------------------------------------------------------- fig: validation
def fig_validation():
    d = load('e5_validation')
    if not d:
        return
    df = common.load()
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    coords = StandardScaler().fit_transform(df[['Latitude', 'Longitude']])
    lab = KMeans(n_clusters=10, random_state=common.SEED, n_init=10).fit_predict(coords)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ax1.scatter(df['Longitude'], df['Latitude'], c=lab, cmap='tab10', s=1, alpha=0.5)
    ax1.set_xlabel('longitude'); ax1.set_ylabel('latitude')
    ax1.set_title('spatial blocks used as CV folds', fontsize=9)
    keys = [k for k in d['cv']]
    x = np.arange(len(keys))
    ax2.bar(x, [d['cv'][k]['mean'] for k in keys],
            yerr=[d['cv'][k]['sd'] for k in keys], capsize=3,
            color=[BLUE if k.startswith('random') else ORANGE for k in keys])
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.replace('|default', '').replace('selection_', 'sel ')
                         .replace('|', '\n') for k in keys], fontsize=7)
    ax2.set_ylabel('mean $R^2$ across folds')
    save(fig, 'fig_validation')


# ----------------------------------------------------- fig: learning curve
def fig_learning():
    d = load('e8_kernels_search_size')
    if not d:
        return
    c = d['training_size']['curve']
    ns = sorted(int(k) for k in c)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(ns, [c[str(n)]['r2'] for n in ns], 'o-', color=BLUE, label='$R^2$')
    ax.set_xlabel('SVR training-set size'); ax.set_ylabel('test $R^2$')
    ax2 = ax.twinx()
    ax2.plot(ns, [c[str(n)]['fit_seconds'] for n in ns], 's--', color=ORANGE,
             label='fit time (s)')
    ax2.set_ylabel('fit time (s)'); ax2.spines['top'].set_visible(False)
    ax.axvline(3000, color=GREY, ls=':')
    ax.text(3100, min(c[str(n)]['r2'] for n in ns), 'v1 subset', fontsize=7, color=GREY)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc='lower right')
    save(fig, 'fig_learning')


# ------------------------------------------------------- fig: datasets
def fig_datasets():
    d = load('e6_datasets')
    if not d:
        return
    names = list(d)
    comps = ['S', 'K', 'T']
    labels = {'S': 'scaling', 'K': 'selection', 'T': 'tuning'}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    w = 0.25
    for i, c in enumerate(comps):
        ax.bar(np.arange(len(names)) + (i - 1) * w,
               [d[n]['analysis']['auto']['shapley'][c] for n in names], w,
               label=labels[c])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', ' ') for n in names])
    ax.set_ylabel('Shapley value  ($\\Delta R^2$)')
    ax.axhline(0, color='k', lw=0.5); ax.legend(frameon=False, fontsize=8)
    save(fig, 'fig_datasets')


# ------------------------------------------------------ fig: sensitivity
def fig_sensitivity():
    d = load('e7_sensitivity')
    if not d:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    r2 = [w['r2'] for w in d['weights']]
    ax1.hist(r2, bins=18, color=BLUE, alpha=0.85)
    ax1.axvline(d['weights_summary']['default_r2'], color=ORANGE, ls='--',
                label='40/30/30 (default)')
    ax1.set_xlabel('test $R^2$ over the weight simplex'); ax1.set_ylabel('count')
    ax1.legend(frameon=False, fontsize=8)
    ks = sorted(int(k) for k in d['k_sweep'])
    ax2.plot(ks, [d['k_sweep'][str(k)]['r2_uniform'] for k in ks], 'o-', color=BLUE,
             label='uniform scaling')
    ax2.plot(ks, [d['k_sweep'][str(k)]['r2_auto'] for k in ks], 's--', color=ORANGE,
             label='rule-based scaling')
    ax2.axvline(12, color=GREY, ls=':')
    ax2.set_xlabel('number of selected features $k$'); ax2.set_ylabel('test $R^2$')
    ax2.legend(frameon=False, fontsize=8)
    save(fig, 'fig_sensitivity')


# ------------------------------------------------- fig: final diagnostics
def fig_diagnostics():
    d = load('e3_statistics')
    df = common.load()
    Xtr, Xte, ytr, yte = common.split(df)
    top12 = common.select_top_k(Xtr, ytr, 12)
    params = {}
    if d:
        best = d['per_seed'].get('SVR uniform / top12 / tuned / full', [{}])[0].get('best_params', {})
        params = {k.replace('svr__', ''): v for k, v in best.items()}
    pipe = common.make_pipeline(top12, 'uniform', svr_params=params).fit(Xtr[top12], ytr)
    pred = pipe.predict(Xte[top12])
    resid = yte - pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax1.scatter(pred, resid, s=2, alpha=0.25, color=BLUE)
    ax1.axhline(0, color=ORANGE, ls='--')
    ax1.set_xlabel('predicted value (\\$100{,}000s)'); ax1.set_ylabel('residual')
    ax2.scatter(yte, pred, s=2, alpha=0.25, color=BLUE)
    lim = [float(min(yte.min(), pred.min())), float(max(yte.max(), pred.max()))]
    ax2.plot(lim, lim, color=ORANGE, ls='--')
    ax2.set_xlabel('actual value (\\$100{,}000s)'); ax2.set_ylabel('predicted value')
    save(fig, 'fig_diagnostics')


if __name__ == '__main__':
    for f in [fig_correlation, fig_importance, fig_reproduction, fig_ablation,
              fig_paired, fig_validation, fig_learning, fig_datasets,
              fig_sensitivity, fig_diagnostics]:
        try:
            f()
        except Exception as e:
            print('SKIP', f.__name__, type(e).__name__, e)
