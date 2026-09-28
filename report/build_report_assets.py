"""Builds everything the paper needs from the saved results, so no number is typed by hand.

- copies the figures used in the paper into report/figures/
- writes report/numbers.tex (one LaTeX macro per number quoted in the text)
- writes report/tables/*.tex (the tables of the paper)

Run from the project folder:  python report/build_report_assets.py
Only the Python standard library is used.
"""
import csv
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
REP = os.path.join(ROOT, 'report')
os.makedirs(os.path.join(REP, 'figures'), exist_ok=True)
os.makedirs(os.path.join(REP, 'tables'), exist_ok=True)


def load_json(name):
    with open(os.path.join(RES, 'evaluation_reports', name), encoding='utf-8') as f:
        return json.load(f)


def load_csv(name):
    with open(os.path.join(RES, 'tables', name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def exists(name):
    return os.path.exists(os.path.join(RES, 'evaluation_reports', name))


# ---------------------------------------------------------------- figures
FIGURES = ['fig_02_class_balance.png', 'fig_04_loss_curves.png', 'fig_05_confusion_matrix.png',
           'fig_06_per_category_f1.png', 'fig_09_demo_hate.png', 'fig_10_keyword_bias.png',
           'fig_11_debias_fpr.png']
m_rat = load_json('metrics_rationale.json')
FIGURES += [f'fig_08_human_vs_model_{c}.png' for c in m_rat['figure_comments']]
for name in FIGURES:
    src = os.path.join(RES, 'figures', name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(REP, 'figures', name))
    else:
        print('missing figure (skipped):', name)

# ---------------------------------------------------------------- numbers
macros = {}


def num(name, value, fmt='{:.2f}'):
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def intc(v):
    return f'{int(v):,}'.replace(',', '{,}')


m_data = load_json('metrics_data.json')
m_base = load_json('metrics_baseline.json')
m_train = load_json('metrics_training.json')
m_expl = load_json('explanations_examples.json')
m_bias = load_json('metrics_bias.json')

s = m_data['split_sizes']
num('nTrain', intc(s['train']))
num('nVal', intc(s['validation']))
num('nTest', intc(s['test']))
num('nTotal', intc(sum(s.values())))
hate_total = sum(m_data['hate_counts'].values())
num('nHateTotal', intc(hate_total))
num('hatePct', 100 * hate_total / sum(s.values()), '{:.1f}')
num('nHateTest', intc(m_data['hate_counts']['test']))
cc = m_data['category_counts']
num('bodyShamingTrain', intc(cc['train']['Body Shaming']))
num('bodyShamingTest', intc(cc['test']['Body Shaming']))
num('religiousTest', intc(cc['test']['Religious']))
num('originTest', intc(cc['test']['Origin']))
num('personalOffenseTrain', intc(cc['train']['Personal Offense']))
num('consistencyViolations', intc(sum(v['nonhate_with_category'] + v['hate_without_category']
                                      for v in m_data['label_category_consistency'].values())))
num('urlRows', intc(sum(v['rows_with_url'] for v in m_data['cleaning'].values())))
num('dupTrain', intc(m_data['duplicates']['train_duplicate_rows']))
num('testInTrain', intc(m_data['duplicates']['test_rows_also_in_train']))
num('medianWords', m_data['word_length']['train']['Median'], '{:.0f}')
num('pNinetyFiveWords', m_data['word_length']['train']['95th pct.'], '{:.0f}')
paper_rows = {r['Category']: r for r in load_csv('tab_09_paper_vs_hf_counts.csv')}
num('paperHateTotal', intc(paper_rows['Hate (binary)']['Paper total']))
num('paperBodyShamingTrain', intc(paper_rows['Body Shaming']['Paper train']))

tok = m_train['token_length']
num('tokPNinetyNine', tok['p99'], '{:.0f}')
num('maxLen', tok['max_length_chosen'], '{:d}')
num('truncPct', tok['share_truncated_pct'], '{:.1f}')
num('trainMinutes', m_train['training_minutes'], '{:.1f}')
num('bestEpoch', m_train['best_epoch'], '{:d}')
num('binThr', m_train['thresholds'], '{:.2f}')
num('nParams', m_train['n_parameters'] / 1e6, '{:.0f}')

tb = next(r for r in m_train['test_binary'] if r['Threshold'] == 'Tuned on validation')
tb05 = next(r for r in m_train['test_binary'] if r['Threshold'] == '0.5')
num('binF', tb['Macro-F1'])
num('binMacroAcc', tb['Macro-Acc.'])
num('binAcc', tb['Accuracy'])
num('hateP', tb['Hate Precision'])
num('hateR', tb['Hate Recall'])
num('binFHalf', tb05['Macro-F1'])
ml = {('all' if r['Evaluated on'].startswith('All') else 'hate', r['Threshold']): r for r in m_train['test_multilabel']}
num('mlFall', ml[('all', 'Tuned on validation')]['Macro-F1'])
num('subsetAll', ml[('all', 'Tuned on validation')]['Subset Acc.'])
num('hammAll', ml[('all', 'Tuned on validation')]['Hamming Loss'])
num('mlFhate', ml[('hate', 'Tuned on validation')]['Macro-F1'])
num('microFhate', ml[('hate', 'Tuned on validation')]['Micro-F1'])
num('subsetHate', ml[('hate', 'Tuned on validation')]['Subset Acc.'])
num('hammHate', ml[('hate', 'Tuned on validation')]['Hamming Loss'])
num('mlFhateHalf', ml[('hate', '0.5')]['Macro-F1'])
cat = {r['Category']: r for r in m_train['test_per_category']}
for c, key in [('Personal Offense', 'catPO'), ('Abusive/Violence', 'catAV'), ('Political', 'catPol'),
               ('Gender', 'catGen'), ('Misc', 'catMisc'), ('Religious', 'catRel'), ('Origin', 'catOri'),
               ('Body Shaming', 'catBody')]:
    num(key, cat[c]['F1 (all comments)'], '{:.1f}')
    num(key + 'Hate', cat[c]['F1 (hate comments)'], '{:.1f}')
cm = m_train['test_confusion_matrix']
num('nFN', intc(cm[1][0]))
num('pctFN', 100 * cm[1][0] / sum(cm[1]), '{:.1f}')
num('nFP', intc(cm[0][1]))
num('pctFP', 100 * cm[0][1] / sum(cm[0]), '{:.1f}')

base = {r['Model']: r for r in m_base['results']}
num('blXlmrBin', base['XLM-R']['Binary Macro-F1'])
num('blXlmrAcc', base['XLM-R']['Binary Macro-Acc.'])
num('blXlmrMl', base['XLM-R']['Multi-label Macro-F1'])
num('blBanglaBertBin', base['BanglaBERT']['Binary Macro-F1'])
num('blBestBin', m_base['best']['all_methods']['Binary Macro-F1']['value'])
num('blBestBinModel', m_base['best']['all_methods']['Binary Macro-F1']['model'])
num('blBestMl', m_base['best']['all_methods']['Multi-label Macro-F1']['value'])
num('blBestMlModel', m_base['best']['all_methods']['Multi-label Macro-F1']['model'])
num('blBestMlFt', m_base['best']['fine_tuned_only']['Multi-label Macro-F1']['value'])
num('blBestMlFtModel', m_base['best']['fine_tuned_only']['Multi-label Macro-F1']['model'])
num('gapBin', base['XLM-R']['Binary Macro-F1'] - tb['Macro-F1'])
num('gapMl', base['XLM-R']['Multi-label Macro-F1'] - ml[('hate', 'Tuned on validation')]['Macro-F1'])

num('topThreeOverlap', m_expl['mean_top3_overlap'])

num('ratSample', intc(m_rat['sample_size']))
num('ratA', intc(m_rat['annotated']['Annotator A']))
num('ratB', intc(m_rat['annotated']['Annotator B']))
num('ratBoth', intc(m_rat['annotated_by_both']))
num('ratWords', intc(m_rat['words_compared']))
num('kappaVal', m_rat['cohen_kappa_word_level'], '{:.3f}')
num('rawAgree', 100 * m_rat['raw_word_agreement'], '{:.1f}')
num('faithN', intc(m_rat['faithfulness_comments']))
pl = {(r['Reference'], r['Method']): r for r in m_rat['plausibility']}
for ref, rk in [('Annotator A', 'A'), ('Annotator B', 'B')]:
    num('plN' + rk, intc(pl[(ref, 'SHAP')]['Comments']))
    for meth in ['LIME', 'SHAP', 'Random']:
        k = meth.lower().capitalize() + rk
        num('auprc' + k, pl[(ref, meth)]['AUPRC'], '{:.3f}')
        num('tokF' + k, pl[(ref, meth)]['Token F1'], '{:.3f}')
        num('iouF' + k, pl[(ref, meth)]['IOU F1'], '{:.3f}')
fa = {r['Method']: r for r in m_rat['faithfulness']}
for meth in ['Human', 'LIME', 'SHAP', 'Random']:
    k = meth.lower().capitalize()
    num('comp' + k, fa[meth]['Comp. (k = human)'], '{:.2f}')
    num('suff' + k, fa[meth]['Suff. (k = human)'], '{:.2f}')

bs = {r['Group'][0]: r for r in m_bias['summary']}
num('biasNA', intc(bs['A']['Sentences']))
num('biasNB', intc(bs['B']['Sentences']))
num('biasFlagA', intc(bs['A']['Flagged as hate']))
num('biasFlagB', intc(bs['B']['Flagged as hate']))
num('biasFprA', bs['A']['False positive rate (%)'], '{:.1f}')
num('biasFprB', bs['B']['False positive rate (%)'], '{:.1f}')
num('biasMeanPA', bs['A']['Mean P(hate)'], '{:.2f}')
num('biasMeanPB', bs['B']['Mean P(hate)'], '{:.2f}')
num('biasShapTop', bs['A']['Risky word is SHAP top word (%)'], '{:.1f}')
num('biasNWords', intc(len(m_bias['word_stats'])))
ws = {r['Word']: r for r in m_bias['word_stats']}
for w in ['kuttar', 'kukur', 'baccha', 'pagol', 'hindu', 'bnp']:
    num('hr' + w.capitalize(), ws[w]['Hate rate (%)'], '{:.0f}')
num('trainHatePct', m_bias['train_hate_rate_pct'], '{:.1f}')

HAVE_DEBIAS = exists('metrics_debias.json')
if HAVE_DEBIAS:
    m_deb = load_json('metrics_debias.json')
    num('debNWords', intc(m_deb['word_rule']['n_words']))
    num('debMinCount', intc(m_deb['word_rule']['min_count']))
    num('debMinRate', 100 * m_deb['word_rule']['min_hate_rate'], '{:.0f}')
    num('debNRepeated', intc(m_deb['n_repeated_comments']))
    num('debTrainRows', intc(m_deb['train_rows'][1]))
    num('debBinF', m_deb['test_binary']['Macro-F1'])
    num('debBinAcc', m_deb['test_binary']['Macro-Acc.'])
    num('debMlF', m_deb['test_multilabel_hate']['Macro-F1'])
    num('debFprA', m_deb['probe_fpr']['new']['A'], '{:.1f}')
    num('debFprB', m_deb['probe_fpr']['new']['B'], '{:.1f}')
    t = m_deb['test_fpr']
    num('debTestOldWith', t['old_with'], '{:.1f}')
    num('debTestNewWith', t['new_with'], '{:.1f}')
    num('debTestOldWithout', t['old_without'], '{:.1f}')
    num('debTestNewWithout', t['new_without'], '{:.1f}')
    num('debTestNWith', intc(t['n_with']))
    num('debTestNWithout', intc(t['n_without']))
    num('debBestEpoch', m_deb['best_epoch'], '{:d}')
    num('debThr', m_deb['binary_threshold'], '{:.2f}')
    num('debBinDrop', tb['Macro-F1'] - m_deb['test_binary']['Macro-F1'])
    num('debMlDrop', ml[('hate', 'Tuned on validation')]['Macro-F1'] - m_deb['test_multilabel_hate']['Macro-F1'])
    num('debSubsetHate', m_deb['test_multilabel_hate']['Subset Acc.'])

with open(os.path.join(REP, 'numbers.tex'), 'w', encoding='utf-8') as f:
    f.write('% Generated by build_report_assets.py from results/. Do not edit by hand.\n')
    f.write(f'\\newif\\ifdebias\\debias{"true" if HAVE_DEBIAS else "false"}\n')
    for k, v in macros.items():
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(len(macros), 'number macros written; debias results available:', HAVE_DEBIAS)

# ---------------------------------------------------------------- tables


def tex_escape(s):
    return (str(s).replace('\\', '\\textbackslash{}').replace('&', '\\&').replace('%', '\\%')
            .replace('_', '\\_').replace('#', '\\#'))


def write_table(name, align, header, rows, midrules=()):
    lines = [f'\\begin{{tabular}}{{{align}}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append('\\midrule')
        lines.append(' & '.join(r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(REP, 'tables', name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


f2 = lambda v: f'{float(v):.2f}'
f1 = lambda v: f'{float(v):.1f}'
f3 = lambda v: f'{float(v):.3f}'

# Label distribution
rows = [[tex_escape(r['Label']), intc(r['Train']), intc(r['Validation']), intc(r['Test']), f1(r['Train (%)'])]
        for r in load_csv('tab_02_class_balance.csv')]
write_table('label_distribution.tex', 'lrrrr', ['Label', 'Train', 'Val.', 'Test', 'Train (\\%)'], rows, midrules=(2,))

# Training setup
rows = [[tex_escape(r['Setting']), tex_escape(r['Value'])] for r in load_csv('tab_10_training_setup.csv')]
rows = [r for r in rows if r[0] not in ('Heads',)]
write_table('training_setup.tex', 'lp{5.2cm}', ['Setting', 'Value'], rows)

# Comparison with baselines
keep = ['BanglaBERT', 'mBERT', 'XLM-R', 'TB-BERT', 'TB-mBERT', 'GPT 4o', 'GPT 4o + Few-shot']
comp = load_csv('tab_14_comparison_with_baseline.csv')
rows = []
for r in comp:
    if r['Source'] == 'BanTH paper' and r['Model'] in keep:
        rows.append([tex_escape(r['Model']), f2(r['Binary Macro-F1']), f2(r['Binary Macro-Acc.']),
                     f2(r['Multi-label Macro-F1']), f2(r['Subset Acc.']), f2(r['Hamming Loss'])])
for r in comp:
    if r['Source'] == 'This work':
        label = 'Ours, ML on all' if 'all comments' in r['Model'] else 'Ours, ML on hate'
        rows.append([label, f2(r['Binary Macro-F1']), f2(r['Binary Macro-Acc.']),
                     f2(r['Multi-label Macro-F1']), f2(r['Subset Acc.']), f2(r['Hamming Loss'])])
write_table('comparison.tex', 'lrrrrr', ['Model', 'Bin. F1', 'Bin. Acc.', 'ML F1', 'Subset', 'Hamm.$\\downarrow$'],
            rows, midrules=(len(rows) - 2,))

# Binary results
rows = [[tex_escape(r['Threshold']).replace('Tuned on validation', 'Tuned (val.)'), f2(r['Macro-F1']), f2(r['Macro-Acc.']),
         f2(r['Accuracy']), f2(r['Hate Precision']), f2(r['Hate Recall']), f2(r['Hate F1'])]
        for r in load_csv('tab_11_binary_results.csv')]
write_table('binary_results.tex', 'lrrrrrr', ['Threshold', 'M-F1', 'M-Acc.', 'Acc.', 'P', 'R', 'F1'], rows)

# Per-category results
rows = [[tex_escape(r['Category']), intc(r['Train count']), intc(r['Test support']), f1(r['Precision']),
         f1(r['Recall']), f1(r['F1 (all comments)']), f1(r['F1 (hate comments)'])]
        for r in load_csv('tab_13_per_category_results.csv')]
write_table('per_category.tex', 'lrrrrrr', ['Category', 'Train', 'Test', 'P', 'R', 'F1', 'F1\\textsubscript{hate}'], rows)

# Plausibility
rows = [[r['Reference'].replace('Annotator ', ''), r['Comments'], r['Method'], f3(r['AUPRC']), f3(r['Token F1']), f3(r['IOU F1'])]
        for r in load_csv('tab_18_plausibility.csv')]
write_table('plausibility.tex', 'cclrrr', ['Ref.', '$n$', 'Method', 'AUPRC', 'Tok. F1', 'IOU F1'], rows, midrules=(3,))

# Faithfulness
rows = []
for r in load_csv('tab_19_faithfulness.csv'):
    aopc_c = f2(r['Comp. (AOPC)']) if r['Comp. (AOPC)'] else '--'
    aopc_s = f2(r['Suff. (AOPC)']) if r['Suff. (AOPC)'] else '--'
    rows.append([r['Method'], f2(r['Comp. (k = human)']), f2(r['Suff. (k = human)']), aopc_c, aopc_s])
write_table('faithfulness.tex', 'lrrrr', ['Rationale', 'Comp.$\\uparrow$', 'Suff.$\\downarrow$',
                                          'Comp.\\textsubscript{AOPC}$\\uparrow$', 'Suff.\\textsubscript{AOPC}$\\downarrow$'], rows)

# Keyword bias summary
rows = []
for r in load_csv('tab_23_bias_probe_summary.csv'):
    top = f1(r['Risky word is SHAP top word (%)']) if r['Risky word is SHAP top word (%)'] else '--'
    rows.append([tex_escape(r['Group']), r['Sentences'], r['Flagged as hate'], f1(r['False positive rate (%)']),
                 f2(r['Mean P(hate)']), top])
write_table('bias_summary.tex', 'lrrrrr', ['Group', '$n$', 'Flagged', 'FPR (\\%)', 'Mean $p$', 'SHAP top (\\%)'], rows)

# Debias comparison
if HAVE_DEBIAS:
    rows = []
    for r in load_csv('tab_24_debias_comparison.csv'):
        rows.append([tex_escape(r['Model']), f2(r['Binary Macro-F1']), f2(r['Multi-label Macro-F1 (hate)']),
                     f1(r['Probe FPR, risky word']), f1(r['Probe FPR, no risky word']),
                     f1(r['Test FPR, with listed word']), f1(r['Test FPR, without'])])
    write_table('debias.tex', 'lrrrrrr', ['Model', 'Bin. F1', 'ML F1', 'Probe A', 'Probe B', 'Test w/', 'Test w/o'], rows)

print('tables written to report/tables/')
