#!/usr/bin/env python3
"""Smoke tests for the dataset conversion and metrics pipeline.

This test checks two things:
 - Converted CSVs have a single-line header (no embedded newlines in the header)
 - The metrics computation can read the converted + trimmed file and produce numbers

Run with: python3 -m pytest scripts/tests/test_smoke_pipeline.py
"""
import io
import os
import csv
import subprocess


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(ROOT, "..", "logs", "baseline_vs_mod")


def first_line_has_no_newline(path):
    with open(path, "rb") as f:
        data = f.read()
    # Find index of first LF
    idx = data.find(b"\n")
    assert idx != -1, "file has no newline"
    header = data[:idx]
    # header should not contain any CR/LF characters other than final LF we sliced by
    assert b"\r" not in header, f"CR found in header of {path}"
    assert b"\n" not in header, f"LF found in header of {path}"


def test_converted_trimmed_headers():
    # Look for any *_conv_trim_10_60.csv files
    found = 0
    for name in os.listdir(OUTDIR):
        if name.endswith("_conv_trim_10_60.csv"):
            found += 1
            path = os.path.join(OUTDIR, name)
            first_line_has_no_newline(path)
    assert found >= 1, f"no trimmed converted files found in {OUTDIR}"


def test_metrics_run_on_one_file():
    # pick one converted+trimmed file
    for name in os.listdir(OUTDIR):
        if name.endswith("_conv_trim_10_60.csv"):
            path = os.path.join(OUTDIR, name)
            break
    else:
        raise AssertionError("no trimmed converted files to test metrics on")

    # run compute_metrics.py on the file and ensure it prints a numeric result line
    proc = subprocess.run([
        "python3",
        os.path.join(ROOT, "compute_metrics.py"),
        path,
    ], stdout=subprocess.PIPE, check=True)
    out = proc.stdout.decode().strip()
    # compute_metrics.py may print a CSV row like: pitch,roll,energy
    parts = [p.strip() for p in out.split(',') if p.strip()]
    assert len(parts) >= 3, out
    # ensure the first three parts are floats
    for p in parts[:3]:
        float(p)
