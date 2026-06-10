#!/usr/bin/env python3
"""
Show GPU kernels that run between specified CUDA-graph-launch blocks in a CG
trace, with matching durations from an eager trace for comparison.

The output table for each slot contains:
  • A summary row for the CUDA graph at the left boundary
  • Individual rows for every eager kernel running in the inter-graph gap,
    matched with their counterparts from the eager trace
  • A summary row for the CUDA graph at the right boundary

For the inter-graph gaps the eager counterpart is the post-forward-pass tail
of the corresponding eager region (all kernels after the last GEMM of the
forward pass — Haar rotation, sphere-k, decode setup, etc.).

For the graph summary rows the eager counterpart is the forward-pass portion
of the corresponding eager region (up to and including the last GEMM).

Virtual block IDs:
  0      = "before graph 1" (shows kernels from trace start to graph 1)
  1…N    = actual CUDA graph index (1-based)
  N+1    = "after graph N"  (shows kernels from graph N end to trace end)

Usage:
  python kernels_between_graphs.py BLOCK_A BLOCK_B [options]

  BLOCK_A, BLOCK_B  1-based graph IDs.  For non-adjacent IDs every
                    intermediate graph and gap is shown in sequence.

Examples:
  python kernels_between_graphs.py 1 2   # gap between prefill and decode-1
  python kernels_between_graphs.py 0 1   # kernels before graph 1 + graph 1
  python kernels_between_graphs.py 6 7   # graph 6 + post-graph-6 kernels
  python kernels_between_graphs.py 1 3   # graphs 1–3 and gaps between them
"""

import argparse
import json
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CG_DEFAULT = (
    'torch_profiler_logs_Qwen2.5/'
    '1779706924603540744-rank-0.1779707084062901763.pt.trace.json'
)
EAGER_DEFAULT = (
    'torch_profiler_logs_Qwen2.5/'
    '1779707341751836738-rank-0.1779707402814146526.pt.trace.json'
)

MATCH_PREFIX = 45   # chars used as matching key
DISPLAY_W    = 80   # display truncation for kernel names
N_LAYERS     = 28   # transformer layers per forward pass


# ---------------------------------------------------------------------------
# Load CG trace
# ---------------------------------------------------------------------------
def load_cg_data(path: str):
    """Return (graphs, gaps).

    graphs – list of N dicts, one per cudaGraphLaunch, 1-indexed:
        {idx, corr, label, kernels, total_dur, gpu_start, gpu_end}

    gaps   – list of N+1 gap dicts, 0-indexed:
        gap[0]   = eager kernels before graph 1
        gap[i]   = eager kernels strictly between graph i and graph i+1
        gap[N]   = eager kernels after graph N
        Each gap dict: {gap_idx, kernels (list of {name,dur}), total_dur}
    """
    with open(path) as f:
        data = json.load(f)
    events = data.get('traceEvents', data) if isinstance(data, dict) else data

    all_kernels = sorted(
        [e for e in events if e.get('cat') == 'kernel'],
        key=lambda e: e['ts'],
    )

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
    for k in all_kernels:
        c = k.get('args', {}).get('correlation')
        if c in graph_corr_set:
            by_corr[c].append(k)

    graphs = []
    decode_idx = 0
    for i, (corr, cpu_ts) in enumerate(launches):
        ks = sorted(by_corr.get(corr, []), key=lambda k: k['ts'])
        total_dur = sum(k['dur'] for k in ks)
        gpu_start = ks[0]['ts'] if ks else 0.0
        gpu_end   = max(k['ts'] + k['dur'] for k in ks) if ks else 0.0
        has_causal = any('MaskMode)1' in k['name'] for k in ks)
        if has_causal:
            label = 'prefill'
        else:
            decode_idx += 1
            label = f'decode {decode_idx}'
        graphs.append({
            'idx':       i + 1,
            'corr':      corr,
            'label':     label,
            'kernels':   [{'name': k['name'], 'dur': k['dur']} for k in ks],
            'total_dur': total_dur,
            'gpu_start': gpu_start,
            'gpu_end':   gpu_end,
        })

    # Non-graph (eager) kernels grouped into gaps
    eager_only = [
        k for k in all_kernels
        if k.get('args', {}).get('correlation') not in graph_corr_set
    ]

    # Gap boundaries in GPU time
    gap_bounds = (
        [(0.0, graphs[0]['gpu_start'])]                                   # before graph 1
        + [(graphs[i]['gpu_end'], graphs[i + 1]['gpu_start'])
           for i in range(len(graphs) - 1)]                               # between graphs
        + [(graphs[-1]['gpu_end'], float('inf'))]                         # after last graph
    ) if graphs else [(0.0, float('inf'))]

    gaps = []
    for gi, (t0, t1) in enumerate(gap_bounds):
        ks = [{'name': k['name'], 'dur': k['dur']}
              for k in eager_only if t0 <= k['ts'] < t1]
        gaps.append({
            'gap_idx':   gi,
            'kernels':   ks,
            'total_dur': sum(k['dur'] for k in ks),
        })

    return graphs, gaps


# ---------------------------------------------------------------------------
# Load eager trace
# ---------------------------------------------------------------------------
def _split_at_last_gemm(kernels: list):
    """Split a region into (forward_pass, post_processing).

    The boundary is after the last forward-pass kernel: either the last
    GEMM (allspark / f16_gemm_splitk_reduce) or the last
    fused_add_rms_norm_kernel — whichever comes later.  model.norm runs
    after the final down-proj GEMM and is still part of the forward pass;
    everything after that is post-processing (Haar rotation, sphere-k, etc.).
    """
    last_fwd = max(
        (i for i, k in enumerate(kernels)
         if 'allspark' in k['name']
         or 'f16_gemm_splitk_reduce' in k['name']
         or 'fused_add_rms_norm_kernel' in k['name']),
        default=-1,
    )
    if last_fwd < 0:
        return kernels, []
    return kernels[:last_fwd + 1], kernels[last_fwd + 1:]


def load_eager_data(path: str):
    """Return (pre_region_kernels, regions).

    pre_region_kernels – kernels before the first rms_norm_kernel in the trace.

    regions – list of region dicts, 1-indexed:
        {idx, fwd_kernels, post_kernels, fwd_total, post_total, all_total}

    fwd_kernels  – forward-pass kernels (up to and including the last GEMM)
    post_kernels – post-processing kernels (Haar rotation, sphere-k, etc.)
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
        cands = [r for r in rms_ts if r < t]
        return cands[-1] if cands else None

    boundaries = [(last_rms_before(t) or t) for t in first_reshape_per_fwd]
    ends = list(boundaries[1:]) + [float('inf')]

    pre_region = [k for k in all_kernels if k['ts'] < boundaries[0]]

    regions = []
    for i, (t0, t1) in enumerate(zip(boundaries, ends)):
        ks = [k for k in all_kernels if t0 <= k['ts'] < t1]
        fwd, post = _split_at_last_gemm(ks)
        regions.append({
            'idx':       i + 1,
            'fwd_kernels':  fwd,
            'post_kernels': post,
            'fwd_total':  sum(k['dur'] for k in fwd),
            'post_total': sum(k['dur'] for k in post),
            'all_total':  sum(k['dur'] for k in ks),
        })

    return pre_region, regions


# ---------------------------------------------------------------------------
# Occurrence-indexed lookup
# ---------------------------------------------------------------------------
def _build_index(kernels: list, prefix_len: int = MATCH_PREFIX) -> dict:
    idx: dict = defaultdict(list)
    for k in kernels:
        idx[k['name'][:prefix_len]].append(k['dur'])
    return idx


# ---------------------------------------------------------------------------
# Rendering helpers  (tab-separated output)
# ---------------------------------------------------------------------------

def _dur(d) -> str:
    return f'{d:.3f}' if d is not None else '—'

def _pct(p) -> str:
    return f'{p:+.1f}%' if p is not None else '—'

def _tsv(*cols) -> str:
    return '\t'.join(str(c) for c in cols)

_HDR_DATA = _tsv('Kernel Name', 'CG (µs)', 'Eager (µs)', 'Diff (µs)', 'Diff%')
_HDR_GRAPH = _tsv('Kernel Name', 'CG (µs)', 'Eager fwd (µs)', 'Diff (µs)', 'Diff%')


def _print_graph_row(graph: dict, eager_fwd_total: float | None):
    """Print a TSV summary block for one CUDA graph launch."""
    cg = graph['total_dur']
    print(f"\n## CUDA GRAPH {graph['idx']} [{graph['label'].upper()}]"
          f"\tcorr={graph['corr']}\t{len(graph['kernels'])} kernels")
    print(_HDR_GRAPH)

    if eager_fwd_total is not None:
        diff = eager_fwd_total - cg
        pct  = diff / cg * 100 if cg else 0.0
        print(_tsv('[TOTAL — forward pass only]',
                   _dur(cg), _dur(eager_fwd_total), _dur(diff), _pct(pct)))
    else:
        print(_tsv('[TOTAL]', _dur(cg), '—', '—', '—'))


def _print_gap_rows(gap: dict, eager_post: list):
    """Print one TSV row per kernel in the inter-graph gap with eager counterpart."""
    gap_ks   = gap['kernels']
    cg_total = gap['total_dur']

    eager_idx  = _build_index(eager_post)
    eager_used: dict = defaultdict(int)

    print(f"\n## GAP {gap['gap_idx']}"
          f"\t{len(gap_ks)} eager kernels\tCG total = {cg_total:.3f} µs")
    print(_HDR_DATA)

    if not gap_ks:
        print('(no eager kernels in this gap)')
        return 0.0, 0.0

    eager_matched = 0.0
    for k in gap_ks:
        name   = k['name']
        cg_dur = k['dur']
        prefix = name[:MATCH_PREFIX]
        occ    = eager_used[prefix]
        avail  = eager_idx.get(prefix, [])

        if occ < len(avail):
            e_dur = avail[occ]
            diff  = e_dur - cg_dur
            pct   = diff / cg_dur * 100 if cg_dur else 0.0
            eager_matched += e_dur
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
    eager_extra = 0.0
    if extras:
        print('\n## Eager-only — present in post-fwd tail but absent from CG gap')
        print(_HDR_DATA)
        for prefix, durs in sorted(extras):
            s = sum(durs)
            eager_extra += s
            print(_tsv(prefix, '—', f'{s:.3f} ({len(durs)}×)', '—', '—'))
        eager_matched += eager_extra

    # Gap totals
    eager_post_total = sum(k['dur'] for k in eager_post)
    diff_t = eager_post_total - cg_total
    pct_t  = diff_t / cg_total * 100 if cg_total else 0.0
    print(_tsv('[GAP TOTAL]', _dur(cg_total), _dur(eager_post_total),
               _dur(diff_t), _pct(pct_t)))

    return cg_total, eager_post_total


# ---------------------------------------------------------------------------
# Main rendering loop
# ---------------------------------------------------------------------------
def render(block_a: int, block_b: int,
           graphs: list, gaps: list,
           pre_region: list, regions: list):
    N = len(graphs)

    print(f'## Slot: block {block_a} → block {block_b}'
          f'\t{block_b - block_a} gap(s)\t{len(graphs)} graphs total')

    cg_graph_grand = 0.0
    cg_gap_grand   = 0.0
    eg_graph_grand = 0.0
    eg_gap_grand   = 0.0

    for current in range(block_a, block_b + 1):
        is_graph = 1 <= current <= N

        if is_graph:
            g = graphs[current - 1]
            eager_fwd_total = regions[current - 1]['fwd_total'] if current <= len(regions) else None
            _print_graph_row(g, eager_fwd_total)
            cg_graph_grand += g['total_dur']
            if eager_fwd_total is not None:
                eg_graph_grand += eager_fwd_total

        if current < block_b:
            gap_idx = current
            gap = gaps[gap_idx]

            if gap_idx == 0:
                eager_post = [{'name': k['name'], 'dur': k['dur']} for k in pre_region]
            elif gap_idx <= len(regions):
                eager_post = [{'name': k['name'], 'dur': k['dur']}
                              for k in regions[gap_idx - 1]['post_kernels']]
            else:
                eager_post = []

            cg_g, eg_g = _print_gap_rows(gap, eager_post)
            cg_gap_grand += cg_g
            eg_gap_grand += eg_g

    # Grand totals
    print('\n## GRAND TOTALS')
    print(_tsv('Category', 'CG (µs)', 'Eager (µs)', 'Diff (µs)', 'Diff%'))

    def total_row(label, cg, eg):
        d = eg - cg
        p = d / cg * 100 if cg else 0.0
        print(_tsv(label, _dur(cg), _dur(eg), _dur(d), _pct(p)))

    total_row('CUDA graphs (forward pass only)', cg_graph_grand, eg_graph_grand)
    total_row('Inter-graph gaps (post-processing)', cg_gap_grand, eg_gap_grand)
    total_row('COMBINED TOTAL', cg_graph_grand + cg_gap_grand,
              eg_graph_grand + eg_gap_grand)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('block_a', type=int,
                        help='Left boundary (0 = before first graph)')
    parser.add_argument('block_b', type=int,
                        help='Right boundary (N+1 = after last graph)')
    parser.add_argument('--cg',    default=CG_DEFAULT,    metavar='PATH',
                        help='CG trace (cudaGraphLaunch trace)')
    parser.add_argument('--eager', default=EAGER_DEFAULT, metavar='PATH',
                        help='Eager execution trace')
    args = parser.parse_args()

    print(f'Loading CG trace:    {args.cg}', file=sys.stderr)
    graphs, gaps = load_cg_data(args.cg)
    N = len(graphs)
    print(f'  {N} CUDA-graph launches found', file=sys.stderr)
    for g in graphs:
        print(f'    graph {g["idx"]}: {g["label"]:12s}  corr={g["corr"]:6d}  '
              f'{len(g["kernels"])} kernels  {g["total_dur"]:.2f} µs', file=sys.stderr)

    print(f'Loading eager trace: {args.eager}', file=sys.stderr)
    pre_region, regions = load_eager_data(args.eager)
    print(f'  {len(regions)} eager regions found', file=sys.stderr)
    for r in regions:
        print(f'    region {r["idx"]}: fwd={r["fwd_total"]:.2f} µs  '
              f'post={r["post_total"]:.2f} µs  all={r["all_total"]:.2f} µs',
              file=sys.stderr)

    # Validate
    if not (0 <= args.block_a < args.block_b <= N + 1):
        print(f'ERROR: require 0 <= block_a < block_b <= {N + 1}', file=sys.stderr)
        sys.exit(1)

    render(args.block_a, args.block_b, graphs, gaps, pre_region, regions)


if __name__ == '__main__':
    main()
