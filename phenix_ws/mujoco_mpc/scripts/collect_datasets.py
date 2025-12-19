#!/usr/bin/env python3
"""Universal dataset collector for mjpc runs (vanila + mod).

This script runs the mjpc binary multiple times with a given task and
collects CSV logs (using MJPC_CSV_LOG), converts them with
`convert_mjpc_csv.py`, computes metrics and writes a summary CSV.

Example:
  python3 collect_datasets.py --bin build/bin/mjpc --task "Quadruped Flat" \
      --outdir logs/myruns --runs 3 --tag baseline
"""
import argparse
import os
import subprocess
from pathlib import Path
import csv
import shlex


def run_once(binpath, task, out_csv, env_overrides, max_sim_time):
  env = os.environ.copy()
  env.update(env_overrides)
  env['MJPC_CSV_LOG'] = str(out_csv)
  if max_sim_time is not None:
    env['MJPC_MAX_SIM_TIME'] = str(max_sim_time)
  cmd = [str(binpath), '--task=' + task]
  print('Running', ' '.join(shlex.quote(c) for c in cmd), '->', out_csv)
  # pipe newline to avoid any interactive prompt
  p = subprocess.run(cmd, input='\n', text=True, env=env)
  return p.returncode == 0 and out_csv.exists()


def convert_and_metrics(conv_script_dir, csv_path):
  conv = csv_path.with_suffix('')
  conv = Path(str(conv) + '_conv.csv')
  # convert
  subprocess.run(['python3', str(conv_script_dir / 'convert_mjpc_csv.py'), str(csv_path), str(conv)], check=True)
  # compute metrics (may exit with codes for NO_DATA / NO_TRAVEL)
  res = subprocess.run(['python3', str(conv_script_dir / 'compute_metrics.py'), str(conv)], capture_output=True, text=True)
  return conv, res


def main():
  parser = argparse.ArgumentParser(description='Collect datasets from mjpc runs')
  parser.add_argument('--bin', required=True, help='Path to mjpc binary')
  parser.add_argument('--task', required=True, help='Task name to pass to mjpc')
  parser.add_argument('--outdir', required=True, help='Output directory for CSVs')
  parser.add_argument('--runs', type=int, default=3, help='Number of runs')
  parser.add_argument('--tag', default='run', help='Tag prefix for CSV files')
  parser.add_argument('--max-sim-time', type=float, default=45.0, help='MJPC_MAX_SIM_TIME (s)')
  parser.add_argument('--tmin', type=float, default=None, help='Trim converted CSV to start at tmin (seconds)')
  parser.add_argument('--tmax', type=float, default=None, help='Trim converted CSV to end at tmax (seconds)')
  parser.add_argument('--env', action='append', default=[], help='Extra env overrides, format KEY=VAL')
  args = parser.parse_args()

  binpath = Path(args.bin)
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  conv_script_dir = Path(__file__).resolve().parent

  env_overrides = {}
  for kv in args.env:
    if '=' in kv:
      k, v = kv.split('=', 1)
      env_overrides[k] = v

  results_csv = outdir / 'results.csv'
  with results_csv.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['tag','run','margin_pitch_mean','margin_roll_mean','energy_abs_mean','csv','distance_traveled_m','status'])

  for i in range(1, args.runs + 1):
    csv_path = outdir / f"{args.tag}_r{i}.csv"
    # remove stale
    if csv_path.exists():
      csv_path.unlink()

    ok = run_once(binpath, args.task, csv_path, env_overrides, args.max_sim_time)
    if not csv_path.exists():
      print('ERROR: missing', csv_path)
      with results_csv.open('a', newline='') as f:
        csv.writer(f).writerow([args.tag, i, 'NA','NA','NA', str(csv_path), 'NA', 'ERROR'])
      continue

    try:
      conv, res = convert_and_metrics(conv_script_dir, csv_path)
    except subprocess.CalledProcessError as e:
      print('Conversion failed for', csv_path)
      with results_csv.open('a', newline='') as f:
        csv.writer(f).writerow([args.tag, i, 'NA','NA','NA', str(csv_path), 'NA', 'CONVERT_FAIL'])
      continue

    stdout = res.stdout.strip()
    stderr = res.stderr.strip()
    # optionally trim converted CSV to [tmin, tmax]
    if args.tmin is not None or args.tmax is not None:
      trimmed = Path(str(conv).replace('.csv', f'_trim_{int(args.tmin or 0)}_{int(args.tmax or 0)}.csv'))
      def trim_csv(inp, outp, tmin, tmax):
        import csv as _csv
        with open(inp, newline='') as inf, open(outp, 'w', newline='') as outf:
          reader = _csv.reader(inf)
          writer = _csv.writer(outf)
          try:
            header = next(reader)
          except StopIteration:
            return False
          writer.writerow(header)
          for row in reader:
            if not row:
              continue
            try:
              t = float(row[0])
            except Exception:
              # skip rows with non-float time
              continue
            if (tmin is None or t >= tmin) and (tmax is None or t <= tmax):
              writer.writerow(row)
        return True

      ok_trim = trim_csv(conv, trimmed, args.tmin, args.tmax)
      if ok_trim:
        conv = trimmed
      else:
        print('Trimming produced no data for', conv)
    if res.returncode == 2 or stdout == 'NO_DATA':
      status = 'NO_DATA'
      with results_csv.open('a', newline='') as f:
        csv.writer(f).writerow([args.tag, i, 'NA','NA','NA', str(conv), 'NA', status])
      continue
    if res.returncode == 3 or stdout.startswith('NO_TRAVEL'):
      # try to get distance (5th column)
      try:
        with conv.open() as cf:
          last = list(csv.reader(cf))[-1]
          dist = last[4] if len(last) > 4 else 'NA'
      except Exception:
        dist = 'NA'
      status = 'NO_TRAVEL'
      with results_csv.open('a', newline='') as f:
        csv.writer(f).writerow([args.tag, i, 'NA','NA','NA', str(conv), dist, status])
      continue

    if res.returncode != 0:
      print('compute_metrics returned non-zero:', res.returncode, stderr)
      status = 'METRIC_FAIL'
      with results_csv.open('a', newline='') as f:
        csv.writer(f).writerow([args.tag, i, 'NA','NA','NA', str(conv), 'NA', status])
      continue

    mp, mr, en = stdout.split(',')
    # extract distance from conv file
    try:
      with conv.open() as cf:
        last = list(csv.reader(cf))[-1]
        dist = last[4] if len(last) > 4 else 'NA'
    except Exception:
      dist = 'NA'

    with results_csv.open('a', newline='') as f:
      csv.writer(f).writerow([args.tag, i, mp, mr, en, str(conv), dist, 'OK'])

  print('Done. Results:', results_csv)


if __name__ == '__main__':
  main()
