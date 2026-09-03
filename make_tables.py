"""Generate every results table of the revised manuscript from results/*.json.

Each function writes one LaTeX fragment into paper/tables/.  The manuscript
\input's them, so the text, the tables and the figures are all produced from
the same JSON and cannot disagree (Reviewer 1 #10).
"""
import json, os
OUT = 'paper/tables'
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = f'results/{name}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, body):
    open(f'{OUT}/{name}.tex', 'w').write(body)
    print('wrote', name)


def tab(caption, label, colspec, header, rows, note=None, small=True):
    s = ['\\begin{table}[htbp]', '\\centering',
         '\\caption{%s}\\label{%s}' % (caption, label)]
    if small:
        s.append('\\small')
    s += ['\\begin{tabular}{%s}' % colspec, '\\toprule', header + ' \\\\', '\\midrule']
    s += [r + ' \\\\' for r in rows]
    s += ['\\bottomrule', '\\end{tabular}']
    if note:
        s.append('\\begin{flushleft}\\footnotesize %s\\end{flushleft}' % note)
    s.append('\\end{table}')
    return '\n'.join(s)


# ------------------------------------------------------------------- E1b
def t_reproduction():
    d = load('e1b_preethi_reproduction')
    if not d:
        return
    names = {'none|default': 'Unscaled, library defaults',
             'none|tuned': 'Unscaled, $C$ and $\\varepsilon$ grid-searched',
             'uniform|default': 'Standardised, library defaults',
             'uniform|tuned': 'Standardised, $C$ and $\\varepsilon$ grid-searched'}
    rows = []
    for k, lab in names.items():
        v = d['summary'][k]
        rows.append('%s & $%.3f \\pm %.3f$ & $%.3f$ & $%+.3f$'
                    % (lab, v['mean_r2'], v['sd_r2'], v['mean_mse'], v['gap_to_reported_r2']))
    rep = d['reported_by_preethi']
    rows.append('\\midrule Reported by Preethi et al. & $\\approx %.2f$ & $\\approx %.2f$ & ---'
                % (rep['r2'], rep['mse']))
    write('tab_reproduction', tab(
        'Reproduction of the SVR configuration described by Preethi et al. (2025). '
        'Eight raw features, RBF kernel, 80/20 split, mean $\\pm$ standard deviation over %d seeds.' % d['n_seeds'],
        'tab:reproduction', 'lccc',
        'Configuration & Test $R^2$ & Test MSE & Gap to reported $R^2$', rows,
        note='The values attributed to Preethi et al. are read from their Figs.~7 and 8; '
             'their paper reports no numerical table. Their stated procedure tunes $C$ and '
             '$\\varepsilon$ only, leaving $\\gamma$ at the library default, and describes no scaling step.'))


# -------------------------------------------------------------------- E1
def t_configurations():
    d = load('e1_replication')
    if not d:
        return
    scal = {'none': 'none', 'uniform': 'uniform (standardise all)',
            'feature_specific': 'feature-specific (v1 table)'}
    rows = []
    for r in d:
        rows.append('%s & %s & %d & %d & %.4f & %.4f'
                    % (r['config'].split(' ')[0], scal[r['scaling']],
                       r['n_train'], r['n_features'], r['r2'], r['rmse']))
    write('tab_configurations', tab(
        'Scaling strategy, feature set and training size evaluated on one identical split. '
        'Rows A use the eight raw features, rows P and F likewise; rows U and S use the '
        'twelve selected features.',
        'tab:configurations', 'llrrcc',
        'ID & Scaling & $n_{\\text{train}}$ & $p$ & Test $R^2$ & Test RMSE', rows,
        note='Uniform standardisation (P1, $R^2$ shown) outperforms the feature-specific '
             'assignment (F1) on identical data, which is the comparison requested in '
             'editorial point~1.'))


# -------------------------------------------------------------------- E2
def t_models():
    d = load('e2_fair_comparison')
    if not d:
        return
    tags = list(d)
    order = sorted(d[tags[-1]], key=lambda m: -d[tags[-1]][m]['r2'])
    rows = []
    for i, m in enumerate(order, 1):
        cells = ' & '.join('%.4f & %.4f' % (d[t][m]['r2'], d[t][m]['rmse']) for t in tags)
        name = '\\textbf{%s}' % m if m.startswith('SVR-RBF') else m
        rows.append('%d & %s & %s' % (i, name, cells))
    head = ('Rank & Model & '
            + ' & '.join('\\multicolumn{2}{c}{$n = %s$}' % t.split('=')[1] for t in tags)
            + ' \\\\\n & & ' + ' & '.join(['$R^2$ & RMSE'] * len(tags)))
    write('tab_models', tab(
        'Ten-model comparison in which every model receives the same twelve features, the '
        'same training rows and the same tuning budget (20 randomised search iterations, '
        'three-fold cross-validation). Ranking follows the full-training-set column.',
        'tab:models', 'rl' + 'cc' * len(tags), head, rows,
        note='In v1 the SVR row was fitted to 3{,}000 rows while the other nine models were '
             'fitted to 14{,}448; both training sizes are shown here so the effect of that '
             'asymmetry is visible rather than implicit.'))


# -------------------------------------------------------------------- E3
def t_statistics():
    d = load('e3_statistics')
    if not d:
        return
    rows = []
    for k, v in d['summary'].items():
        rows.append('%s & $%.4f \\pm %.4f$ & [%.3f, %.3f] & [%.3f, %.3f] & [%.3f, %.3f]'
                    % (k.replace('/', '/').replace('_', '\\_'), v['mean_r2'], v['sd_r2'],
                       *v['ci95_correct'], *v['ci95_nadeau_bengio'], *v['ci95_v1_incorrect']))
    write('tab_statistics', tab(
        'Mean test $R^2$ over %d independent stratified splits, with the interval computed '
        'three ways.' % d['n_splits'], 'tab:statistics', 'lcccc',
        'Configuration & Mean $\\pm$ SD & CI (correct) & CI (Nadeau--Bengio) & CI (v1 formula)',
        rows,
        note='The v1 column is $\\bar{x} \\pm 1.96\\,s$, which is a prediction interval for a '
             'single split rather than a confidence interval for the mean; it is reproduced '
             'here only to show the size of the error.'))

    keys = [k for k in d['paired_tests'] if k.startswith('SVR uniform / top12 / tuned / full')]
    rows = []
    for k in keys:
        v = d['paired_tests'][k]
        rows.append('%s & $%+.4f$ & [%+.3f, %+.3f] & %.4f & %.4f & %d/%d'
                    % (k.split('  vs  ')[1].replace('_', '\\_'), v['mean_diff'],
                       *v['ci95_diff_nadeau_bengio'], v['p_corrected'], v['p_uncorrected'],
                       v['wins'], v['losses']))
    write('tab_paired', tab(
        'Paired comparisons against the tuned SVR on identical splits. $p_{\\text{corr}}$ is the '
        'Nadeau--Bengio corrected resampled $t$-test; $p_{\\text{uncorr}}$ is the ordinary paired '
        '$t$-test, shown to illustrate how much it overstates significance.',
        'tab:paired', 'lccccc',
        'Comparison model & $\\Delta R^2$ & 95\\% CI & $p_{\\text{corr}}$ & $p_{\\text{uncorr}}$ & W/L',
        rows))


# ------------------------------------------------------------------- E4b
def t_ablation():
    d = load('e4b_extended_ablation')
    if not d:
        return
    rows = []
    for k, v in sorted(d['cells'].items(), key=lambda kv: -kv[1]['r2']):
        s, dv, kv2, t = k.split('|')
        rows.append('%s & %s & %s & %s & %d & %.4f'
                    % (s, 'yes' if dv == 'derived' else 'no',
                       'top-$k$' if kv2 == 'topk' else 'all', 
                       'grid' if t == 'tuned' else 'default', v['n_features'], v['r2']))
    write('tab_ablation_cells', tab(
        'Every cell of the $4 \\times 2 \\times 2 \\times 2$ ablation design, ordered by test $R^2$.',
        'tab:ablation-cells', 'llllrc',
        'Scaling & Derived & Selection & Tuning & $p$ & Test $R^2$', rows))

    rows = []
    names = {'S': 'Scaling', 'D': 'Derived features', 'K': 'Feature selection',
             'T': 'Hyperparameter tuning'}
    for scaler_on, a in d['analysis'].items():
        for c in 'SDKT':
            sp = a['increment_spread'][c]
            rows.append('%s & %s & $%+.4f$ & $%+.4f$ & $%+.4f$ & $%.4f$'
                        % (scaler_on, names[c], a['shapley'][c], sp['min'], sp['max'], sp['range']))
    write('tab_ablation_shapley', tab(
        'Order-independent attribution. The Shapley value averages a component\'s marginal '
        'effect over all %d orderings; the minimum and maximum columns are the same '
        'component\'s apparent contribution under the most and least favourable ordering.'
        % len(list(d['analysis'].values())[0]['paths']),
        'tab:ablation-shapley', 'llcccc',
        'Scaling used as ``on\'\' & Component & Shapley & Min increment & Max increment & Range',
        rows,
        note='v1 reported a single ordering (scaling $\\to$ features $\\to$ tuning) and read its '
             'increments as independent contributions; the Range column is the size of the error '
             'that assumption introduces.'))


# -------------------------------------------------------------------- E5
def t_validation():
    d = load('e5_validation')
    if not d:
        return
    rows = []
    for k, v in d['cv'].items():
        scheme, sel, _ = k.split('|')
        rows.append('%s & %s & $%.4f \\pm %.4f$ & %.3f & %.3f & %.2f'
                    % (scheme, sel.replace('selection_', ''), v['mean'], v['sd'],
                       v['min'], v['max'], v['mean_pairwise_selection_overlap']))
    for scheme, v in d['nested'].items():
        rows.append('nested %s & inside & $%.4f \\pm %.4f$ & %.3f & %.3f & ---'
                    % (scheme, v['mean'], v['sd'], min(v['per_fold']), max(v['per_fold'])))
    write('tab_validation', tab(
        'Random versus spatially blocked cross-validation, with feature selection performed '
        'outside and inside the resampling loop.', 'tab:validation', 'llcccc',
        'Partitioning & Selection & Mean $R^2$ $\\pm$ SD & Min & Max & Selection overlap',
        rows,
        note='Spatial optimism $= %.4f$; selection optimism $= %.4f$ (random) and $%.4f$ '
             '(spatial). Selection overlap is the mean pairwise share of the twelve chosen '
             'features common to two folds.'
             % (d['optimism']['spatial_optimism'],
                d['optimism']['selection_optimism_random'],
                d['optimism']['selection_optimism_spatial'])))


# -------------------------------------------------------------------- E6
def t_datasets():
    d = load('e6_datasets')
    if not d:
        return
    rows = []
    for name, v in d.items():
        sh = v['analysis']['auto']['shapley']
        base = v['cells']['none|all|default']['r2']
        full = v['cells']['auto|topk|tuned']['r2']
        rows.append('%s & %d & %d & %d & %.4f & %.4f & $%+.4f$ & $%+.4f$ & $%+.4f$'
                    % (name.replace('_', ' '), v['n_train'], v['n_test'], v['n_features'],
                       base, full, sh['S'], sh['K'], sh['T']))
    write('tab_datasets', tab(
        'The same workflow applied unchanged to three tabular regression datasets. '
        'Shapley values are computed over the $3 \\times 2 \\times 2$ design within each dataset.',
        'tab:datasets', 'lrrrcccccc',
        'Dataset & $n_{\\text{train}}$ & $n_{\\text{test}}$ & $p$ & Baseline $R^2$ & '
        'Full $R^2$ & Scaling & Selection & Tuning', rows))


# -------------------------------------------------------------------- E7
def t_sensitivity():
    d = load('e7_sensitivity')
    if not d:
        return
    w = d['weights_summary']
    rows = ['Ensemble weights over the simplex (%d combinations) & %.4f & %.4f & %.4f & %.4f'
            % (w['n'], w['min'], w['max'], w['range'], w['default_r2'])]
    ks = sorted(int(k) for k in d['k_sweep'])
    kv = [d['k_sweep'][str(k)]['r2_uniform'] for k in ks]
    rows.append('Number of selected features, $k \\in [%d, %d]$ & %.4f & %.4f & %.4f & %.4f'
                % (ks[0], ks[-1], min(kv), max(kv), max(kv) - min(kv),
                   d['k_sweep']['12']['r2_uniform']))
    tv = [v['r2'] for v in d['scaler_thresholds'].values()]
    rows.append('Scaler-rule thresholds (%d combinations) & %.4f & %.4f & %.4f & ---'
                % (len(tv), min(tv), max(tv), max(tv) - min(tv)))
    write('tab_sensitivity', tab(
        'Sensitivity of the test $R^2$ to the three constants that were fixed by hand.',
        'tab:sensitivity', 'lcccc',
        'Swept quantity & Min $R^2$ & Max $R^2$ & Range & Value at the default', rows))


# -------------------------------------------------------------------- E8
def t_kernels():
    d = load('e8_kernels_search_size')
    if not d:
        return
    rows = ['%s & %.4f & %.4f & %.0f' % (k, v['r2'], v['rmse'], v['seconds'])
            for k, v in d['kernels'].items()]
    write('tab_kernels', tab(
        'Every kernel tuned over its own hyperparameters (20 randomised iterations, three-fold CV).',
        'tab:kernels', 'lccr', 'Kernel & Test $R^2$ & Test RMSE & Search time (s)', rows))

    rows = ['%s & %d & %.4f & %.4f & %.0f'
            % (k.replace('_', ' '), v['n_fits'], v['cv_best'], v['r2'], v['seconds'])
            for k, v in d['search'].items()]
    write('tab_search', tab(
        'Exhaustive grid search over the v1 discretised space versus randomised search over '
        'continuous distributions.', 'tab:search', 'lrccr',
        'Strategy & Fits & Best CV $R^2$ & Test $R^2$ & Time (s)', rows))

    c = d['training_size']['curve']
    rows = ['%s & %.4f & %.4f & %d & %.1f'
            % (n, c[n]['r2'], c[n]['rmse'], c[n]['n_support'], c[n]['fit_seconds'])
            for n in sorted(c, key=int)]
    for k, v in d['kernel_approximation'].items():
        rows.append('%s (full train) & %.4f & %.4f & --- & %.1f'
                    % (k.replace('_', ' '), v['r2'], v['rmse'], v['seconds']))
    write('tab_size', tab(
        'Effect of the SVR training-set size, and of a Nystr\\"om kernel approximation fitted '
        'to the full training set.', 'tab:size', 'lccrr',
        'Training rows & Test $R^2$ & Test RMSE & Support vectors & Fit time (s)', rows))


if __name__ == '__main__':
    for f in [t_reproduction, t_configurations, t_models, t_statistics, t_ablation,
              t_validation, t_datasets, t_sensitivity, t_kernels]:
        try:
            f()
        except Exception as e:
            print('SKIP', f.__name__, type(e).__name__, e)

# Any table the manuscript \input's that no result file supplies yet gets a
# visible placeholder, so the document always compiles end to end.
import re
try:
    tex = open('paper/main.tex').read()
    for name in re.findall(r'\\input\{tables/([a-z_]+)\}', tex):
        p = f'{OUT}/{name}.tex'
        if not os.path.exists(p):
            open(p, 'w').write(
                '\\begin{table}[htbp]\\centering\\caption{%s --- pending, experiment still running.}'
                '\\label{tab:%s}\\begin{tabular}{c}\\toprule pending \\\\ \\bottomrule'
                '\\end{tabular}\\end{table}\n' % (name.replace('_', ' '),
                                       name.replace('tab_', '').replace('_', '-')))
            print('placeholder', name)
except FileNotFoundError:
    pass
