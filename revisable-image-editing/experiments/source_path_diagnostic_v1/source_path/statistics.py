"""Independent evaluation summaries; never used by acquisition/selection."""
from collections import defaultdict
from . import common as c


def schedule(cases):
    tasks = []
    for case in cases:
        for pair, arms in c.PAIR_ARMS.items():
            flip = int(c.canonical([case['id'], pair])[:8], 16) % 2
            for order in range(2):
                display = arms if order == flip else arms[::-1]
                tasks.append({'id': f'{case["id"]}_{pair}_{order}', 'case_id': case['id'], 'pair': pair,
                              'a_arm': display[0], 'b_arm': display[1], 'order': order, 'control': False})
    for family in c.FAMILIES:
        case = next(r for r in cases if r['family'] == family)
        tasks.append({'id': 'identical_'+case['id'], 'case_id': case['id'], 'pair': 'identical',
                      'a_arm': 'base', 'b_arm': 'base', 'order': 0, 'control': True})
    return sorted(tasks, key=lambda r: c.canonical(['judge-order-v1', r['id']]))


def parse_judgment(raw):
    import json
    def unique(items):
        value = {}
        for k, v in items:
            if k in value: raise ValueError('Duplicate judgment field')
            value[k] = v
        return value
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != {*c.FIELDS, 'reason'}:
        raise ValueError('Unexpected judgment schema')
    if any(value[k] not in ('A', 'B', 'tie', 'unassessable') for k in c.FIELDS):
        raise ValueError('Invalid judgment choice')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('Missing visible-evidence reason')
    return value


def estimate(rows, field, families, draws, seed):
    import numpy as np
    grouped = {f: defaultdict(list) for f in families}
    for row in rows:
        if row['family'] in grouped: grouped[row['family']][row['source_group']].append(row[field])
    groups = sorted({g for values in grouped.values() for g in values})
    if len(groups) < 2 or any(not grouped[f] for f in families):
        raise ValueError('Incomplete family/source-group denominator')
    matrix = np.full((len(families), len(groups)), np.nan)
    for i, family in enumerate(families):
        for j, group in enumerate(groups):
            values = grouped[family].get(group)
            if values: matrix[i,j] = sum(values)/len(values)
    valid = np.isfinite(matrix); values = np.nan_to_num(matrix)
    point = float((values.sum(1)/valid.sum(1)).mean())
    rng = np.random.default_rng(seed); outcomes = []; skipped = 0
    while len(outcomes) < draws:
        weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)))
        denominator = valid @ weights
        if not np.all(denominator):
            skipped += 1
            if skipped > draws: raise RuntimeError('Cannot bootstrap the fixed category denominator')
            continue
        outcomes.append(float(((values @ weights)/denominator).mean()))
    return {'mean': point, 'ci95': [float(v) for v in np.quantile(outcomes, [.025,.975])],
            'cases': sum(r['family'] in families for r in rows), 'source_groups': len(groups), 'draws': draws}


def summarize(cases, tasks, receipts, case_receipts, cfg):
    expected = {task['id']: task for task in tasks}
    if len(receipts) != len(tasks) or {r['task']['id'] for r in receipts} != set(expected):
        raise ValueError('Incomplete external evaluation; do not drop rows')
    by_case = {r['id']: r for r in cases}; grouped = defaultdict(list)
    for row in receipts:
        task = row['task']
        if task != expected[task['id']]: raise ValueError('External task identity changed')
        value = parse_judgment(__import__('json').dumps(row['judgment']))
        if any(value[k] == 'unassessable' for k in c.FIELDS):
            raise ValueError('Unassessable case blocks the decision; no denominator reduction')
        if task['control']:
            if any(value[k] != 'tie' for k in c.FIELDS): raise ValueError('Identical-image judge control failed')
            continue
        first = c.PAIR_ARMS[task['pair']][0]
        credit = {k: .5 if value[k] == 'tie' else float(task['a_arm' if value[k]=='A' else 'b_arm'] == first) for k in c.FIELDS}
        grouped[(task['pair'], task['case_id'])].append(credit)
    rows = defaultdict(list)
    for (pair, case_id), orders in grouped.items():
        if len(orders) != 2: raise ValueError('Both display orders are required')
        case = by_case[case_id]
        rows[pair].append({'family': case['family'], 'source_group': case['fingerprint']['decoded_rgb_sha256'],
                          **{k: sum(v[k] for v in orders)/2 for k in c.FIELDS}})
    statistics = {}
    for pair, values in rows.items():
        statistics[pair] = {'macro': {}, 'families': {f:{} for f in c.FAMILIES}}
        for k in c.FIELDS:
            seed = int(c.canonical(['bootstrap',pair,k])[:16],16)
            statistics[pair]['macro'][k] = estimate(values, k, c.FAMILIES, cfg['bootstrap_samples'], seed)
            for f in c.FAMILIES:
                statistics[pair]['families'][f][k] = estimate(values, k, [f], cfg['bootstrap_samples'], seed)
    if set(statistics) != set(c.PAIR_ARMS): raise ValueError('Required comparisons are incomplete')
    coverage = {arm: sum(r['selection'][arm]['accepted'] for r in case_receipts)/len(cases)
                for arm in ('selected','resample','clarify')}
    selection_frequencies = {}
    for arm in ('selected','resample','clarify'):
        counts = defaultdict(int)
        for row in case_receipts: counts[row['selection'][arm].get('candidate','base42')] += 1
        selection_frequencies[arm] = dict(counts)
    gates = {'native_selection_coverage': coverage['selected'] >= cfg['minimum_selection_coverage']}
    for pair in c.PRIMARY_PAIRS:
        macro = statistics[pair]['macro']; semantic = macro['semantic']
        gates[pair+'_semantic'] = semantic['mean'] >= cfg['minimum_semantic_credit'] and semantic['ci95'][0] > .5
        for k in ('preservation','quality'):
            gates[pair+'_'+k] = macro[k]['ci95'][0] >= cfg['minimum_retention_lower_bound']
        gates[pair+'_breadth'] = sum(statistics[pair]['families'][f]['semantic']['mean'] >= .5 for f in c.FAMILIES) >= cfg['minimum_nonnegative_families']
    passed = all(gates.values())
    return {'status': 'complete', 'passed': passed, 'decision': 'eligible_for_separate_mechanism_confirmation' if passed else 'do_not_train',
            'gates': gates, 'coverage': coverage, 'selection_frequencies': selection_frequencies,
            'intervention_selected_fraction': sum(n for name,n in selection_frequencies['selected'].items() if name != 'base42')/len(cases),
            'statistics': statistics, 'cases': len(cases), 'judgments': len(tasks),
            'credit': 'win=1, tie=0.5, loss=0; average both orders, physical-source groups, equal-family macro',
            'secondary_intervals': 'descriptive, unadjusted; not evidence for an externally selected best setting',
            'training_updates': 0, 'automatic_training': False, 'novelty_claim_established': False}
