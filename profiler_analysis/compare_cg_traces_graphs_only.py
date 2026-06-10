#!/usr/bin/env python3
"""
Compare GPU-kernel durations between the CUDA-graph forward passes captured
in one vLLM trace and the corresponding eager forward passes in another.

Left side  (CG)   : kernels that belong to cudaGraphLaunch events in
                    CG_TRACE  (1779706924603540744-…)
Right side (eager): kernels from the matching forward-pass time windows in
                    EAGER_TRACE (1779707341751836738-…)

Kernels are matched by (name-prefix-45-chars, occurrence-number) within each
forward pass.  Unmatched slots are shown as '—'.

Usage:
    python compare_cg_traces.py
"""

import json
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CG_TRACE = (
    'torch_profiler_logs_Qwen2.5/'
    '1779706924603540744-rank-0.1779707084062901763.pt.trace.json'
)
EAGER_TRACE = (
    'torch_profiler_logs_Qwen2.5/'
    '1779707341751836738-rank-0.1779707402814146526.pt.trace.json'
)

MATCH_PREFIX = 45   # chars used as matching key
DISPLAY_W    = 80   # display truncation for kernel names
N_LAYERS     = 28   # transformer layers per forward pass


# ---------------------------------------------------------------------------
# Load CG trace → list of 6 sections, one per cudaGraphLaunch
# ---------------------------------------------------------------------------
def load_cg_sections(path: str):
    """Return chronologically-ordered list of CUDA-graph sections.

    Each section dict:
        corr      – correlation ID of the cudaGraphLaunch CPU event
        cpu_ts    – CPU timestamp of the launch (µs)
        kernels   – list of {name, dur} dicts sorted by GPU ts
        total_dur – sum of all kernel durations
        label     – human-readable tag ('prefill' or 'decode N')
    """
    with open(path) as f:
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

    sections = []
    decode_idx = 0
    for corr, cpu_ts in launches:
        ks = sorted(by_corr.get(corr, []), key=lambda e: e['ts'])
        total_dur = sum(k['dur'] for k in ks)
        has_causal = any('MaskMode)1' in k['name'] for k in ks)
        if has_causal:
            label = 'prefill'
        else:
            decode_idx += 1
            label = f'decode {decode_idx}'
        sections.append({
            'corr':      corr,
            'cpu_ts':    cpu_ts,
            'kernels':   [{'name': k['name'], 'dur': k['dur']} for k in ks],
            'total_dur': total_dur,
            'label':     label,
        })
    return sections


# ---------------------------------------------------------------------------
# Load eager trace → list of 6 forward-pass regions (same logic as before)
# ---------------------------------------------------------------------------
def load_eager_regions(path: str):
    """Return list of 6 kernel lists from the eager trace.

    Each region starts at the rms_norm_kernel that immediately precedes the
    first reshape_and_cache of that forward pass so that layer-0 kernels
    (input-layernorm, QKV GEMM, rotary) are included.
    """
    with open(path) as f:
        data = json.load(f)
    events = data.get('traceEvents', data) if isinstance(data, dict) else data
    all_kernels = sorted(
        [e for e in events if e.get('cat') == 'kernel'],
        key=lambda e: e['ts'],
    )

    reshape_ts = sorted(
        k['ts'] for k in all_kernels
        if 'reshape_and_cache_flash_kernel' in k['name']
    )
    first_reshape_per_fwd = reshape_ts[::N_LAYERS]

    rms_ts = sorted(
        k['ts'] for k in all_kernels if 'rms_norm_kernel' in k['name']
    )

    def last_rms_before(t):
        candidates = [r for r in rms_ts if r < t]
        return candidates[-1] if candidates else None

    boundaries = [
        (last_rms_before(t) or t)
        for t in first_reshape_per_fwd
    ]
    ends = list(boundaries[1:]) + [float('inf')]

    regions = []
    for t_start, t_end in zip(boundaries, ends):
        ks = [k for k in all_kernels if t_start <= k['ts'] < t_end]
        # Trim after the last forward-pass kernel.  model.norm
        # (fused_add_rms_norm_kernel) runs after the final down-proj GEMM and
        # is still part of the forward pass; everything after that is
        # post-processing (Haar rotation, sphere-k, decode setup) covered by
        # compare_cg_traces_kernels_btw_graphs.py — exclude here to avoid overlap.
        last_fwd = max(
            (i for i, k in enumerate(ks)
             if 'allspark' in k['name']
             or 'f16_gemm_splitk_reduce' in k['name']
             or 'fused_add_rms_norm_kernel' in k['name']),
            default=len(ks) - 1,
        )
        regions.append(ks[:last_fwd + 1])
    return regions


# ---------------------------------------------------------------------------
# Occurrence-indexed lookup: name_prefix → [dur0, dur1, …]
# ---------------------------------------------------------------------------
def build_index(kernels, prefix_len=MATCH_PREFIX):
    idx: dict = defaultdict(list)
    for k in kernels:
        idx[k['name'][:prefix_len]].append(k['dur'])
    return idx


# ---------------------------------------------------------------------------
# Rendering helpers (tab-separated output)
# ---------------------------------------------------------------------------
def _dur(d) -> str:
    return f'{d:.3f}' if d is not None else '—'

def _pct(p) -> str:
    return f'{p:+.1f}%' if p is not None else '—'

def _tsv(*cols) -> str:
    return '\t'.join(str(c) for c in cols)

_HDR = _tsv('Kernel Name', 'CG (µs)', 'Eager (µs)', 'Diff (µs)', 'Diff%')


# ---------------------------------------------------------------------------
# Render one comparison table
# ---------------------------------------------------------------------------
def render_table(cg_section: dict, eager_kernels: list, sec_num: int):
    cg_kernels  = cg_section['kernels']
    cg_total    = cg_section['total_dur']
    eager_total = sum(k['dur'] for k in eager_kernels)

    eager_idx  = build_index(eager_kernels)
    eager_used = defaultdict(int)

    print(f"\n## Section {sec_num} | {cg_section['label'].upper()}"
          f"\tcorr={cg_section['corr']}"
          f"\tCG kernels={len(cg_kernels)}"
          f"\tCG total={cg_total:.3f} µs")
    print(_HDR)

    for k in cg_kernels:
        name   = k['name']
        cg_dur = k['dur']
        prefix = name[:MATCH_PREFIX]
        occ    = eager_used[prefix]
        avail  = eager_idx.get(prefix, [])

        if occ < len(avail):
            e_dur = avail[occ]
            diff  = e_dur - cg_dur
            pct   = diff / cg_dur * 100 if cg_dur else 0.0
            print(_tsv(name, _dur(cg_dur), _dur(e_dur), _dur(diff), _pct(pct)))
        else:
            print(_tsv(name, _dur(cg_dur), '—', '—', '—'))

        eager_used[prefix] += 1

    # Eager-only extras
    extras = [
        (p, durs[eager_used[p]:])
        for p, durs in eager_idx.items()
        if eager_used[p] < len(durs)
    ]
    if extras:
        print('\n## Eager-only — no counterpart in CG graph')
        print(_HDR)
        for prefix, durs in sorted(extras):
            print(_tsv(prefix, '—', f'{sum(durs):.3f} ({len(durs)}×)', '—', '—'))

    # Totals
    eager_matched = sum(sum(durs[:eager_used[p]]) for p, durs in eager_idx.items())
    eager_extras  = sum(sum(durs) for _, durs in extras)

    print(_tsv('CG TOTAL', _dur(cg_total), '', '', ''))
    print(_tsv('Eager TOTAL (matched)', '', _dur(eager_matched), '', ''))
    if eager_extras > 0:
        print(_tsv('Eager TOTAL (extras)', '', _dur(eager_extras), '', ''))
    print(_tsv('Eager TOTAL (all)', '', _dur(eager_total), '', ''))
    overall_diff = eager_total - cg_total
    overall_pct  = overall_diff / cg_total * 100 if cg_total else 0.0
    print(_tsv('OVERALL DIFF', '', '', _dur(overall_diff), _pct(overall_pct)))

    return cg_total, eager_total


# ---------------------------------------------------------------------------
# Summary across all sections
# ---------------------------------------------------------------------------
def render_summary(results: list):
    print('\n## SUMMARY — total kernel duration per forward pass')
    print(_tsv('Section', 'CG (µs)', 'Eager (µs)', 'Diff (µs)', 'Diff%'))

    cg_grand = eager_grand = 0.0
    for label, cg, eager in results:
        diff = eager - cg
        pct  = diff / cg * 100 if cg else 0.0
        print(_tsv(label, _dur(cg), _dur(eager), _dur(diff), _pct(pct)))
        cg_grand    += cg
        eager_grand += eager

    diff_g = eager_grand - cg_grand
    pct_g  = diff_g / cg_grand * 100 if cg_grand else 0.0
    print(_tsv('GRAND TOTAL', _dur(cg_grand), _dur(eager_grand),
               _dur(diff_g), _pct(pct_g)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f'Loading CG trace:    {CG_TRACE}', file=sys.stderr)
    cg_sections = load_cg_sections(CG_TRACE)
    print(f'  {len(cg_sections)} CUDA-graph sections found', file=sys.stderr)

    print(f'Loading eager trace: {EAGER_TRACE}', file=sys.stderr)
    eager_regions = load_eager_regions(EAGER_TRACE)
    print(f'  {len(eager_regions)} eager forward-pass regions found', file=sys.stderr)

    if len(cg_sections) != len(eager_regions):
        print(
            f'WARNING: section count mismatch '
            f'(CG={len(cg_sections)}, eager={len(eager_regions)})',
            file=sys.stderr,
        )

    n_pairs  = min(len(cg_sections), len(eager_regions))
    summary  = []
    for i in range(n_pairs):
        sec   = cg_sections[i]
        label = f"Section {i + 1} | {sec['label']} | corr={sec['corr']}"
        cg, eager = render_table(sec, eager_regions[i], sec_num=i + 1)
        summary.append((label, cg, eager))

    render_summary(summary)


if __name__ == '__main__':
    main()
