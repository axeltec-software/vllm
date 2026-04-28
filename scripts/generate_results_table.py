#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def parse_run_stats(stats_file: Path) -> Dict[str, Any]:
    with open(stats_file, 'r') as f:
        data = json.load(f)
    
    model_name = data.get('model_name', 'Unknown')
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    batch_size = data.get('batch_size', 0)
    
    poc_requests = 0
    folder_name = stats_file.parent.name
    if folder_name.startswith('chat_') and '_poc_' in folder_name:
        parts = folder_name.split('_')
        if len(parts) >= 4:
            poc_requests = int(parts[3])
    
    elapsed = data.get('elapsed_seconds', 0)
    
    gsm8k = data.get('gsm8k', {})
    strict_match = gsm8k.get('strict_match', 0.0) if gsm8k else 0.0
    flexible_extract = gsm8k.get('flexible_extract', 0.0) if gsm8k else 0.0
    
    median_time = data.get('poc_median_time', None)
    
    return {
        'model': model_name,
        'batch_size': batch_size,
        'poc_requests': poc_requests,
        'strict_match': strict_match,
        'flexible_extract': flexible_extract,
        'elapsed': elapsed,
        'median_time': median_time,
        'folder': folder_name,
    }


def generate_table(eval_results_dir: Path, output_file: Path = None):
    results = []
    
    for folder in eval_results_dir.iterdir():
        if not folder.is_dir():
            continue
        
        if not folder.name.startswith('chat_'):
            continue
        
        stats_file = folder / 'run_stats.json'
        if not stats_file.exists():
            continue
        
        try:
            stats = parse_run_stats(stats_file)
            results.append(stats)
        except Exception as e:
            print(f"Warning: Failed to parse {stats_file}: {e}", file=sys.stderr)
    
    if not results:
        print("No results found!", file=sys.stderr)
        return
    
    results.sort(key=lambda x: (x['model'], x['batch_size'], x['poc_requests']))
    
    lines = []
    lines.append("| Configuration | Batch Size | PoC Batch | Accuracy (Strict/Flexible) | Time gsm8k (s) | Median Time PoC (s) |")
    lines.append("|---------------|------------|-----------|----------------------------|----------------|---------------------|")
    
    for r in results:
        model = r['model']
        batch = r['batch_size']
        poc = r['poc_requests']
        strict = r['strict_match']
        flexible = r['flexible_extract']
        elapsed = int(r['elapsed'])
        median = r['median_time']
        
        median_str = f"{median:.3f}" if median is not None else ""
        
        line = f"| {model:<13} | {batch:<10} | {poc:<9} | {strict:.4f} / {flexible:.4f}          | {elapsed:<14} | {median_str:<19} |"
        lines.append(line)
    
    table = '\n'.join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(table + '\n')
        print(f"Table saved to: {output_file}")
    else:
        print(table)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate results table from eval_results folders")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("./eval_results"),
        help="Directory containing results folders (default: ./eval_results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for the table (default: print to stdout)",
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Directory not found: {args.results_dir}", file=sys.stderr)
        return 1
    
    generate_table(args.results_dir, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

