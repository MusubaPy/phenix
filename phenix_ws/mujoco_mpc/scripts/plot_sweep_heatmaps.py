#!/usr/bin/env python3
"""Plot 2D heatmaps (p vs a) of mean_total_J for each fx value.

Usage: scripts/plot_sweep_heatmaps.py --agg <summary_agg.json> --outdir <logs/..>

Writes files: heatmap_fx1e-9.png and heatmap_fx1e-10.png into outdir.
"""
import argparse
import json
import os
import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--agg', required=True, help='aggregated JSON (summary_agg_fixed.json)')
    p.add_argument('--outdir', required=True, help='output directory for PNGs')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.agg) as f:
        data = json.load(f)

    agg = data.get('aggregate', {})
    # Build DataFrame
    rows = []
    for key, vals in agg.items():
        params = vals.get('params', {})
        p = params.get('p')
        a = params.get('a')
        fx = params.get('fx')
        mt = vals.get('mean_total_J')
        rows.append({'p': p, 'a': a, 'fx': fx, 'mean_total_J': mt})

    df = pd.DataFrame(rows)
    if df.empty:
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(rows)
            if df.empty:
                raise SystemExit('No aggregated data found in file.')
            # Convert p,a to numeric for sorting and display
            df['p_num'] = df['p'].astype(float)
            df['a_num'] = df['a'].astype(float)
        else:
            # fallback: use list of dicts, convert numeric values on the fly
            if not rows:
                raise SystemExit('No aggregated data found in file.')

    for fxval in sorted(df['fx'].unique()):
        sub = df[df['fx'] == fxval].copy()
        fx_values = sorted(set(r['fx'] for r in rows))
        for fxval in fx_values:
            if PANDAS_AVAILABLE:
                sub = df[df['fx'] == fxval].copy()
                if sub.empty:
                    print(f'No data for fx={fxval}, skipping')
                    continue
                # pivot: rows=a, cols=p
                pivot = sub.pivot_table(index='a_num', columns='p_num', values='mean_total_J')
                # sort axes
                pivot = pivot.sort_index(ascending=True)
                pivot = pivot.reindex(sorted(pivot.columns), axis=1)
                xlabels = [str(c) for c in pivot.columns]
                ylabels = [str(c) for c in pivot.index]
                Z = pivot.values
            else:
                # build sorted unique lists
                pvals = sorted({float(r['p']) for r in rows})
                avals = sorted({float(r['a']) for r in rows})
                p_to_i = {v:i for i,v in enumerate(pvals)}
                a_to_i = {v:i for i,v in enumerate(avals)}
                Z = np.full((len(avals), len(pvals)), np.nan)
                for r in rows:
                    if r['fx'] != fxval:
                        continue
                    pi = p_to_i[float(r['p'])]
                    ai = a_to_i[float(r['a'])]
                    Z[ai, pi] = r['mean_total_J']
                xlabels = [str(v) for v in pvals]
                ylabels = [str(v) for v in avals]
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', cbar_kws={'label': 'mean_total_J (J)'})
            plt.figure(figsize=(10, 8))
            if PANDAS_AVAILABLE:
                im = sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', cbar_kws={'label': 'mean_total_J (J)'})
            else:
                im = plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
                plt.colorbar(im, label='mean_total_J (J)')
                plt.xticks(ticks=np.arange(len(xlabels)), labels=xlabels, rotation=45)
                plt.yticks(ticks=np.arange(len(ylabels)), labels=ylabels)
            plt.xlabel('alex_power p (numeric)')
            plt.ylabel('alex_align a (numeric)')
            plt.title(f'Mean total_J (mech + heat) — fx={fxval}')
            # annotate a few cells
            for (i, j), val in np.ndenumerate(Z):
                if not np.isnan(val):
                    plt.text(j, i, f'{val:.0f}', ha='center', va='center', color='w', fontsize=7)
            outpng = os.path.join(args.outdir, f'heatmap_fx{fxval.replace("/","_")}.png')
            plt.tight_layout()
            plt.savefig(outpng, dpi=200)
            plt.close()
            print('Wrote', outpng)
    df_out = df[['p','a','fx','mean_total_J']].sort_values(['fx','p','a'])
    csv_out = os.path.join(args.outdir, 'aggregate_table.csv')
    df_out.to_csv(csv_out, index=False)
    print('Wrote', csv_out)


if __name__ == '__main__':
    main()
