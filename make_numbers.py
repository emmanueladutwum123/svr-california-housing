"""Emit paper/numbers.tex: one LaTeX macro per number quoted in the prose.

Every figure quoted in the running text of the manuscript is defined here from
results/*.json, so the text cannot disagree with the tables (Reviewer 1 #10).
Missing results resolve to \??\ so the document still compiles while an
experiment is still running.
"""
import json, os

OUT = 'paper/numbers.tex'
os.makedirs('paper', exist_ok=True)
macros = {}


def load(name):
    p = f'results/{name}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def m(name, value, fmt='%.3f'):
    macros[name] = ('??' if value is None else
                    (fmt % value if isinstance(value, (int, float)) else str(value)))


def pval(x):
    """Format a p-value for math mode: 0.166, or 1.9\times 10^{-7}."""
    if x is None:
        return '??'
    if x >= 1e-3:
        return '%.3f' % x
    import math
    e = int(math.floor(math.log10(x)))
    return r'%.1f\times 10^{%d}' % (x / 10 ** e, e)


def mp(name, x):
    macros[name] = pval(x)


# ---- E1b reproduction ------------------------------------------------------
d = load('e1b_preethi_reproduction')
if d:
    s = d['summary']
    m('RepUnscaledDefault', s['none|default']['mean_r2'])
    m('RepUnscaledTuned', s['none|tuned']['mean_r2'])
    m('RepScaledDefault', s['uniform|default']['mean_r2'])
    m('RepScaledTuned', s['uniform|tuned']['mean_r2'])
    m('RepUnscaledDefaultMSE', s['none|default']['mean_mse'])
    m('RepUnscaledTunedMSE', s['none|tuned']['mean_mse'])
    m('RepScaledTunedMSE', s['uniform|tuned']['mean_mse'])
    m('PreethiReportedRtwo', d['reported_by_preethi']['r2'], '%.2f')
    m('PreethiReportedMSE', d['reported_by_preethi']['mse'], '%.2f')
    m('RepSeeds', d['n_seeds'], '%d')

# ---- E1 configuration grid -------------------------------------------------
d = load('e1_replication')
if d:
    by = {r['config'].split(' ')[0]: r for r in d}
    for key, macro in [('A0', 'CfgUnscaledSub'), ('A1', 'CfgUnscaledFull'),
                       ('P0', 'CfgUniformSub'), ('P1', 'CfgUniformFull'),
                       ('F0', 'CfgManualSub'), ('F1', 'CfgManualFull'),
                       ('U1', 'CfgUniformTopFull'), ('S0', 'CfgManualTopSub'),
                       ('S1', 'CfgManualTopFull')]:
        m(macro, by[key]['r2'] if key in by else None, '%.4f')

# ---- E2 equal-budget comparison -------------------------------------------
d = load('e2_fair_comparison')
if d:
    tags = list(d)
    small, full = tags[0], tags[-1]
    for model, macro in [('SVR-RBF', 'SvrRbf'), ('XGBoost', 'Xgb'),
                         ('Random Forest', 'Rf'), ('Gradient Boosting', 'Gbm'),
                         ('K-Nearest Neighbours', 'Knn'), ('Ridge Regression', 'Ridge')]:
        m(macro + 'Small', d[small][model]['r2'] if model in d[small] else None, '%.4f')
        m(macro + 'Full', d[full][model]['r2'] if model in d[full] else None, '%.4f')
    if 'SVR-RBF' in d[full]:
        rank = 1 + sum(1 for k, v in d[full].items() if v['r2'] > d[full]['SVR-RBF']['r2'])
        m('SvrRankFull', rank, '%d')
        rank_s = 1 + sum(1 for k, v in d[small].items() if v['r2'] > d[small]['SVR-RBF']['r2'])
        m('SvrRankSmall', rank_s, '%d')
        m('NModels', len(d[full]), '%d')

# ---- E3 statistics ---------------------------------------------------------
d = load('e3_statistics')
if d:
    m('StatSplits', d['n_splits'], '%d')
    m('StatRho', d['rho_nadeau_bengio'], '%.3f')
    key = 'SVR uniform / top12 / tuned / full'
    if key in d['summary']:
        v = d['summary'][key]
        m('StatSvrMean', v['mean_r2'], '%.4f'); m('StatSvrSd', v['sd_r2'], '%.4f')
        m('StatSvrCIlo', v['ci95_correct'][0], '%.3f'); m('StatSvrCIhi', v['ci95_correct'][1], '%.3f')
        m('StatSvrNBlo', v['ci95_nadeau_bengio'][0], '%.3f')
        m('StatSvrNBhi', v['ci95_nadeau_bengio'][1], '%.3f')
        m('StatSvrVOnelo', v['ci95_v1_incorrect'][0], '%.3f')
        m('StatSvrVOnehi', v['ci95_v1_incorrect'][1], '%.3f')
    for other, macro in [('XGBoost tuned / top12 / full', 'PairXgb'),
                         ('RandomForest tuned / top12 / full', 'PairRf'),
                         ('SVR manual / top12 / tuned / full', 'PairManual')]:
        k = f'{key}  vs  {other}'
        v = d['paired_tests'].get(k)
        m(macro + 'Diff', v['mean_diff'] if v else None, '%+.4f')
        mp(macro + 'P', v['p_corrected'] if v else None)
        mp(macro + 'Puncorr', v['p_uncorrected'] if v else None)

    flips = [(k, t) for k, t in d['paired_tests'].items()
             if t['p_uncorrected'] < 0.05 <= t['p_corrected']]
    m('NFlipped', len(flips), '%d')
    m('NPairs', len(d['paired_tests']), '%d')
    if flips:
        k, t = flips[0]
        a, b = [x.strip() for x in k.split('  vs  ')]
        macros['FlipA'] = a
        macros['FlipB'] = b
        mp('FlipP', t['p_corrected'])
        mp('FlipPuncorr', t['p_uncorrected'])
        m('FlipDiff', t['mean_diff'], '%+.4f')
    cv = d['cv_interval']
    m('CvMean', cv['mean'], '%.4f'); m('CvSd', cv['sd'], '%.4f')
    m('CvVOnelo', cv['ci95_v1_incorrect'][0], '%.3f'); m('CvVOnehi', cv['ci95_v1_incorrect'][1], '%.3f')
    m('CvCorrlo', cv['ci95_correct'][0], '%.3f'); m('CvCorrhi', cv['ci95_correct'][1], '%.3f')

# ---- E4b ablation ----------------------------------------------------------
d = load('e4b_extended_ablation')
if d:
    a = d['analysis']['auto']
    for c, name in [('S', 'Scal'), ('D', 'Deriv'), ('K', 'Select'), ('T', 'Tune')]:
        m('Shap' + name, a['shapley'][c], '%.4f')
        m('Range' + name, a['increment_spread'][c]['range'], '%.4f')
        m('Min' + name, a['increment_spread'][c]['min'], '%+.4f')
        m('Max' + name, a['increment_spread'][c]['max'], '%+.4f')
    m('AblBaseline', a['baseline_r2'], '%.4f'); m('AblFull', a['full_r2'], '%.4f')
    m('AblNorders', len(a['paths']), '%d'); m('AblNcells', len(d['cells']), '%d')
    best = max(d['cells'].items(), key=lambda kv: kv[1]['r2'])
    m('AblBestCell', best[0].replace('|', ' / ')); m('AblBestRtwo', best[1]['r2'], '%.4f')

# ---- E5 validation ---------------------------------------------------------
d = load('e5_validation')
if d:
    o = d['optimism']
    m('SpatialOptimism', o['spatial_optimism'], '%.4f')
    m('SelectionOptimism', o['selection_optimism_random'], '%.4f')
    m('RandomCVmean', d['cv']['random|selection_inside|default']['mean'], '%.4f')
    m('SpatialCVmean', d['cv']['spatial|selection_inside|default']['mean'], '%.4f')
    m('SpatialCVsd', d['cv']['spatial|selection_inside|default']['sd'], '%.4f')
    m('NestedRandom', d['nested']['random']['mean'], '%.4f')
    m('NestedSpatial', d['nested']['spatial']['mean'], '%.4f')
    for _sch, _tag in [('random', 'Random'), ('spatial', 'Spatial')]:
        _pf = d['nested'][_sch]['per_fold']
        m('Nested' + _tag + 'Min', min(_pf), '%.3f')
        m('Nested' + _tag + 'Max', max(_pf), '%.3f')

# ---- E6 datasets -----------------------------------------------------------
d = load('e6_datasets')
if d:
    # Both scaler strategies are reported. Emitting only 'auto' hid the fact that
    # the component ordering reproduces under uniform standardisation but NOT under
    # the distribution-aware rule, which is negative on Ames.
    for name in d:
        tag = name.title().replace('_', '')
        for strat, suffix in [('uniform', 'Uni'), ('auto', '')]:
            sh = d[name]['analysis'][strat]['shapley']
            m('DS' + tag + suffix + 'Scal', sh['S'], '%+.4f')
            m('DS' + tag + suffix + 'Select', sh['K'], '%+.4f')
            m('DS' + tag + suffix + 'Tune', sh['T'], '%+.4f')
        m('DS' + tag + 'Base', d[name]['cells']['none|all|default']['r2'], '%.4f')
        m('DS' + tag + 'Full', d[name]['cells']['auto|topk|tuned']['r2'], '%.4f')
        m('DS' + tag + 'FullUni', d[name]['cells']['uniform|topk|tuned']['r2'], '%.4f')
        m('DS' + tag + 'N', d[name]['n_train'], '%d')
        m('DS' + tag + 'P', d[name]['n_features'], '%d')
    # does the ordering scaling > tuning > selection hold, per strategy?
    def holds(name, strat):
        sh = d[name]['analysis'][strat]['shapley']
        return sh['S'] > sh['T'] > sh['K']
    m('NOrderHoldsUni', sum(holds(n, 'uniform') for n in d), '%d')
    m('NOrderHoldsAuto', sum(holds(n, 'auto') for n in d), '%d')
    m('NDatasets', len(d), '%d')

# ---- E7 sensitivity --------------------------------------------------------
d = load('e7_sensitivity')
if d:
    w = d['weights_summary']
    m('WeightRange', w['range'], '%.4f'); m('WeightMin', w['min'], '%.4f')
    m('WeightMax', w['max'], '%.4f'); m('WeightN', w['n'], '%d')
    m('WeightMinOverlap', w['min_overlap'], '%d')
    kv = {int(k): v['r2_uniform'] for k, v in d['k_sweep'].items()}
    m('KrangeLo', min(kv.values()), '%.4f'); m('KrangeHi', max(kv.values()), '%.4f')
    m('KbestK', max(kv, key=kv.get), '%d')
    tv = [v['r2'] for v in d['scaler_thresholds'].values()]
    m('ThreshRange', max(tv) - min(tv), '%.4f')

# ---- E8 kernels / search / size -------------------------------------------
d = load('e8_kernels_search_size')
if d:
    for k, v in d['kernels'].items():
        m('Kern' + k.title(), v['r2'], '%.4f')
    m('GridRtwo', d['search']['grid_64']['r2'], '%.4f')
    m('GridFits', d['search']['grid_64']['n_fits'], '%d')
    m('GridSecs', d['search']['grid_64']['seconds'], '%.0f')
    m('RandTwentyRtwo', d['search']['random_20']['r2'], '%.4f')
    m('RandTwoHundredRtwo', d['search']['random_200']['r2'], '%.4f')
    c = d['training_size']['curve']
    ns = sorted(int(x) for x in c)
    m('SizeSmall', c[str(ns[0])]['r2'], '%.4f'); m('SizeSmallN', ns[0], '%d')
    m('SizeThreeK', c['3000']['r2'] if '3000' in c else None, '%.4f')
    m('SizeFull', c[str(ns[-1])]['r2'], '%.4f'); m('SizeFullN', ns[-1], '%d')
    m('SizeThreeKrmse', c['3000']['rmse'] if '3000' in c else None, '%.4f')
    m('SizeFullRmse', c[str(ns[-1])]['rmse'], '%.4f')
    m('SizeFullSecs', c[str(ns[-1])]['fit_seconds'], '%.1f')
    best_ny = max(d['kernel_approximation'].items(), key=lambda kv: kv[1]['r2'])
    m('NystroemRtwo', best_ny[1]['r2'], '%.4f')
    m('NystroemSecs', best_ny[1]['seconds'], '%.0f')
    m('NystroemM', best_ny[0].split('_')[1], '%s')

# Any macro the manuscript references but that no result file supplies yet is
# emitted as ?? so the document still compiles while experiments are running.
import re
LATEX_BUILTINS = {
    'Require', 'Ensure', 'State', 'If', 'Else', 'ElsIf', 'EndIf', 'For', 'EndFor',
    'While', 'EndWhile', 'Comment', 'Return', 'Function', 'EndFunction', 'Procedure',
    'EndProcedure', 'Latex', 'Delta', 'Sigma', 'Roman', 'Alph', 'LaTeX', 'TeX',
    'Big', 'Bigl', 'Bigr', 'Bigg', 'Biggl', 'Biggr', 'Left', 'Right', 'Large',
    'Huge', 'Longrightarrow', 'Leftarrow', 'Rightarrow', 'Pr', 'Re', 'Im',
}
try:
    tex = open('paper/main.tex').read()
    used = {m for m in re.findall(r'\\([A-Z][A-Za-z]{2,})\b', tex)
            if m not in LATEX_BUILTINS}
    missing = sorted(used - set(macros))
    for name in missing:
        macros[name] = '??'
    if missing:
        print('placeholder (??) for %d macro(s): %s' % (len(missing), ', '.join(missing)))
except FileNotFoundError:
    pass

with open(OUT, 'w') as f:
    f.write('% Generated by make_numbers.py -- do not edit by hand.\n')
    for k, v in sorted(macros.items()):
        f.write('\\newcommand{\\%s}{%s}\n' % (k, v))
print('wrote %s with %d macros' % (OUT, len(macros)))
