#!/usr/bin/env python3
"""
Add a 'Hook' label column to every GPU kernel row in a comparison TSV file
(produced by compare_cg_traces_graphs_only.py) that corresponds to a step
of apply_householder() from the per-layer Householder reflection hooks.

Usage:
    python label_householder_hooks.py [INPUT] [--cg PATH] [--inplace]

    INPUT       TSV file to label  (default: comparison_traces_latest.txt)
    --cg PATH   CG trace JSON used to identify Householder kernel indices
    --inplace   Overwrite INPUT instead of printing to stdout

How it works
------------
In each CUDA-graph section of the CG trace, the 14 kernels that immediately
follow each act_and_mul_kernel (offset +2 through +15 after the act_and_mul
index) are Householder kernels:

    act_and_mul_idx + 1  →  down-proj GEMM
    act_and_mul_idx + 2..8   →  apply_householder(hidden)   steps 1-7
    act_and_mul_idx + 9..15  →  apply_householder(residual)  steps 1-7

Step labels per tensor (apply_householder(x, v) = x − 2·(x·v)·v):
    1  v→dtype cast       bfloat16_copy_kernel  (v.to(x.dtype))
    2  x·v                elementwise MulFunctor
    3  dot=(x·v).sum()    reduce sum_functor
    4  2·dot              vectorized AUnaryFunctor·Mul
    5  2·dot·v            elementwise MulFunctor
    6  x−2·dot·v          vectorized CUDAFunctor_add
    7  .copy_()           memcpy32_post          (in-place write-back)

The TSV rows within each '## Section N' block are in the same chronological
order as the CG graph kernels, so kernel index K maps directly to row K of
that section (after the two header lines: '## Section …' and 'Kernel Name…').
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT = 'comparison_traces_latest.txt'
DEFAULT_CG = (
    '../torch_profiler_logs_Qwen2.5/'
    '1779706924603540744-rank-0.1779707084062901763.pt.trace.json'
)

STEP_LABELS = [
    'v→dtype cast',
    'x·v',
    'dot=(x·v).sum()',
    '2·dot',
    '2·dot·v',
    'x−2·dot·v',
    '.copy_()',
]


# ---------------------------------------------------------------------------
# Build per-section Householder index maps from the CG trace
# ---------------------------------------------------------------------------
def build_hook_index_maps(cg_path: str) -> list[dict[int, str]]:
    """Return one dict per CUDA-graph section.

    Each dict maps {kernel_index -> label_string} for the 14 Householder
    kernels that follow every act_and_mul_kernel in that graph.
    """
    with open(cg_path) as f:
        data = json.load(f)
    events = data.get('traceEvents', data) if isinstance(data, dict) else data

    launches = sorted(
        [
            (e['args']['correlation'], e['ts'])
            for e in events
            if e.get('name') == 'cudaGraphLaunch'
            and e.get('cat') == 'cuda_runtime'
            and 'correlation' in e.get('args', {})
        ],
        key=lambda x: x[1],
    )
    graph_corr_set = {c for c, _ in launches}

    by_corr: dict = defaultdict(list)
    for e in events:
        if e.get('cat') == 'kernel':
            c = e.get('args', {}).get('correlation')
            if c in graph_corr_set:
                by_corr[c].append(e)

    section_maps: list[dict[int, str]] = []
    for corr, _ in launches:
        ks = sorted(by_corr.get(corr, []), key=lambda k: k['ts'])
        labels: dict[int, str] = {}
        for act_i in (i for i, k in enumerate(ks) if 'act_and_mul_kernel' in k['name']):
            for step, ki in enumerate(range(act_i + 2, act_i + 9)):
                if ki < len(ks):
                    labels[ki] = f'hook:hidden:  {STEP_LABELS[step]}'
            for step, ki in enumerate(range(act_i + 9, act_i + 16)):
                if ki < len(ks):
                    labels[ki] = f'hook:residual:{STEP_LABELS[step]}'
        section_maps.append(labels)

    return section_maps


# ---------------------------------------------------------------------------
# Apply labels to TSV lines
# ---------------------------------------------------------------------------
def label_tsv(lines: list[str], section_maps: list[dict[int, str]]) -> list[str]:
    """Return a new list of lines with the 'Hook' column appended."""
    out: list[str] = []
    cur_sec = -1
    row_idx = -1
    in_extras = False

    for raw in lines:
        line = raw.rstrip('\n')

        if line.startswith('## Section'):
            cur_sec += 1
            row_idx = -1
            in_extras = False
            out.append(line + '\tHook\n')
            continue

        if line.startswith('## '):
            in_extras = True
            out.append(line + '\n')
            continue

        if line.startswith('Kernel Name\t'):
            row_idx = -1
            out.append(line + '\tHook\n')
            continue

        if (line.startswith('CG TOTAL\t')
                or line.startswith('Eager TOTAL')
                or line.startswith('OVERALL DIFF\t')
                or line == ''):
            out.append(line + '\n')
            continue

        # Data row
        row_idx += 1
        label = ''
        if not in_extras and 0 <= cur_sec < len(section_maps):
            label = section_maps[cur_sec].get(row_idx, '')
        out.append(line + '\t' + label + '\n')

    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', nargs='?', default=DEFAULT_INPUT,
                    help='TSV file to label (default: %(default)s)')
    ap.add_argument('--cg', default=DEFAULT_CG, metavar='PATH',
                    help='CG trace JSON (default: %(default)s)')
    ap.add_argument('--inplace', action='store_true',
                    help='Overwrite the input file instead of printing to stdout')
    args = ap.parse_args()

    cg_path = Path(args.cg)
    if not cg_path.exists():
        # Try resolving relative to the input file's directory
        cg_path = Path(args.input).parent / args.cg
    if not cg_path.exists():
        ap.error(f'CG trace not found: {args.cg}')

    print(f'Loading CG trace: {cg_path}', file=sys.stderr)
    section_maps = build_hook_index_maps(str(cg_path))
    print(f'  {len(section_maps)} CUDA-graph sections found', file=sys.stderr)
    for i, m in enumerate(section_maps):
        h = sum(1 for v in m.values() if 'hidden' in v)
        r = sum(1 for v in m.values() if 'residual' in v)
        print(f'  Section {i + 1}: {len(m)} hook rows ({h} hidden, {r} residual)',
              file=sys.stderr)

    print(f'Reading: {args.input}', file=sys.stderr)
    with open(args.input, encoding='utf-8') as f:
        lines = f.readlines()

    labeled = label_tsv(lines, section_maps)

    if args.inplace:
        with open(args.input, 'w', encoding='utf-8') as f:
            f.writelines(labeled)
        print(f'Written {len(labeled)} lines → {args.input}', file=sys.stderr)
    else:
        sys.stdout.writelines(labeled)


if __name__ == '__main__':
    main()
