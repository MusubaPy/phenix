#!/usr/bin/env python3
"""Aggregate compute_mech_heat.py JSON results into per-parameter summaries.

Usage: scripts/aggregate_sweep.py --raw raw.json --out summary.json
"""
import argparse
import json
import os
import re
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True, help='Raw JSON from compute_mech_heat.py')
    p.add_argument('--out', required=True, help='Output aggregated JSON')
    p.add_argument('--top', type=int, default=5, help='Top results to store')
    return p.parse_args()


FNAME_RE = re.compile(r'p(?P<p>[^_]+)_a(?P<a>[^_]+)_fx(?P<fx>[^_]+)_r(?P<r>\d+)\.csv$')


def main():
    args = parse_args()
    with open(args.raw, 'r') as f:
        data = json.load(f)

    results = data.get('results', {})

    groups = defaultdict(list)

    for fn, vals in results.items():
        base = os.path.basename(fn)
        m = FNAME_RE.search(base)
        if not m:
            # try vanilla patterns
            if 'vanila' in base.lower() or 'vanilla' in base.lower() or 'mjpc_vanila' in base:
                key = ('vanilla', 'vanilla', 'vanilla')
                groups[key].append((base, vals))
            else:
                # skip unknown
                continue
        else:
            p = m.group('p')
            a = m.group('a')
            fx = m.group('fx')
            key = (p, a, fx)
            groups[key].append((base, vals))

    summary = {}
    agg = {}
    for key, entries in groups.items():
        totals = {'mechanical_J': [], 'heat_J': [], 'total_J': [], 'distance_m': []}
        for (fname, vals) in entries:
            totals['mechanical_J'].append(vals.get('mechanical_J', float('nan')))
            totals['heat_J'].append(vals.get('heat_J', float('nan')))
            totals['total_J'].append(vals.get('total_J', float('nan')))
            totals['distance_m'].append(vals.get('distance_m', float('nan')))
        import math
        def mean(lst):
            lst2 = [x for x in lst if x is not None and not (isinstance(x, float) and math.isnan(x))]
            return sum(lst2)/len(lst2) if lst2 else float('nan')

        # Use a string key for JSON compatibility and include params separately
        key_str = f"p={key[0]},a={key[1]},fx={key[2]}"
        agg[key_str] = {
            'params': {'p': key[0], 'a': key[1], 'fx': key[2]},
            'n_runs': len(entries),
            'mean_mechanical_J': mean(totals['mechanical_J']),
            'mean_heat_J': mean(totals['heat_J']),
            'mean_total_J': mean(totals['total_J']),
            'mean_distance_m': mean(totals['distance_m']),
        }

    # pick best combinations by mean_total_J (ignore nan)
    # Build a ranked list of dicts for JSON output (ignore NaN mean_total_J)
    ranked = []
    for k, v in agg.items():
        mt = v.get('mean_total_J')
        if isinstance(mt, float) and mt != mt:
            continue
        ranked.append({'key': k, 'params': v.get('params'), 'metrics': v})
    ranked.sort(key=lambda x: x['metrics']['mean_total_J'])

    out = {'aggregate': agg, 'top': ranked[:args.top]}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote aggregated results to', args.out)


if __name__ == '__main__':
    main()
