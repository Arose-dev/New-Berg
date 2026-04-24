"""
Analyze expert activations from the LEGO-Lite benchmark (4 tasks).

This script reads expert_logs.jsonl and computes the top 20 most active
experts for each of the 4 LEGO-Lite categories: Height, Position, Rotation, Ordering.

Usage:
    python analyze_expert_activations.py --results-dir ~/results/lego_qwen_2026-XX-XX_XX-XX-XX
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def load_expert_logs(log_path):
    """Load expert logs from JSONL file."""
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            logs.append(json.loads(line.strip()))
    return logs


def analyze_expert_activations(logs):
    """
    Aggregate expert activations by category and layer.
    
    Returns:
        dict: {category: {layer: {expert_id: count, ...}, ...}, ...}
    """
    cat_layer_expert = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for log in logs:
        category = log['category']
        expert_log = log.get('expert_log', {})
        layers = expert_log.get('layers', [])
        
        for layer_data in layers:
            layer_idx = layer_data['layer']
            topk_experts = layer_data['topk_experts']  # List of lists: [tokens, top_k]
            
            # Aggregate expert activations across all tokens
            for token_experts in topk_experts:
                for expert_id in token_experts:
                    cat_layer_expert[category][layer_idx][expert_id] += 1
    
    return cat_layer_expert


def get_top_n_experts(cat_layer_expert, n=20):
    """
    Extract top N experts per category and layer.
    
    Returns:
        dict: {category: {layer: [(expert_id, count), ...], ...}, ...}
    """
    result = {}
    
    for category, layers in cat_layer_expert.items():
        result[category] = {}
        for layer_idx, experts in layers.items():
            # Sort by activation count (descending)
            top_experts = sorted(
                experts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n]
            result[category][layer_idx] = top_experts
    
    return result


def print_results(top_experts):
    """Print top experts in a readable format."""
    categories = sorted(top_experts.keys())
    
    for category in categories:
        print(f"\n{'='*70}")
        print(f"Category: {category}")
        print(f"{'='*70}")
        
        layers = sorted(top_experts[category].keys())
        for layer_idx in layers:
            experts = top_experts[category][layer_idx]
            print(f"\n  Layer {layer_idx}:")
            print(f"  {'Rank':<6} {'Expert ID':<12} {'Activations':<15}")
            print(f"  {'-'*6} {'-'*12} {'-'*15}")
            for rank, (expert_id, count) in enumerate(experts, 1):
                print(f"  {rank:<6} {expert_id:<12} {count:<15}")


def write_summary_csv(top_experts, output_path):
    """Write a CSV summary of top experts."""
    import csv
    
    # Flatten the data for CSV
    rows = []
    for category in sorted(top_experts.keys()):
        for layer_idx in sorted(top_experts[category].keys()):
            for rank, (expert_id, count) in enumerate(top_experts[category][layer_idx], 1):
                rows.append({
                    'Category': category,
                    'Layer': layer_idx,
                    'Rank': rank,
                    'Expert ID': expert_id,
                    'Activation Count': count,
                })
    
    if rows:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Category', 'Layer', 'Rank', 'Expert ID', 'Activation Count'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary CSV written to: {output_path}")


def write_summary_json(top_experts, output_path):
    """Write a JSON summary of top experts."""
    # Convert to serializable format
    json_data = {}
    for category in top_experts:
        json_data[category] = {}
        for layer_idx in top_experts[category]:
            json_data[category][str(layer_idx)] = [
                {"expert_id": exp_id, "count": count}
                for exp_id, count in top_experts[category][layer_idx]
            ]
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Summary JSON written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert activations from LEGO-Lite benchmark"
    )
    parser.add_argument(
        '--results-dir', required=True,
        help='Path to results directory containing expert_logs.jsonl'
    )
    parser.add_argument(
        '--top-n', type=int, default=20,
        help='Number of top experts to extract per layer (default: 20)'
    )
    parser.add_argument(
        '--csv-output', default=None,
        help='Path to write CSV summary (default: results_dir/top_experts.csv)'
    )
    parser.add_argument(
        '--json-output', default=None,
        help='Path to write JSON summary (default: results_dir/top_experts.json)'
    )
    args = parser.parse_args()
    
    # Resolve paths
    results_dir = Path(args.results_dir).expanduser()
    log_path = results_dir / 'expert_logs.jsonl'
    
    if not log_path.exists():
        print(f"Error: expert_logs.jsonl not found at {log_path}")
        return
    
    print(f"Loading expert logs from: {log_path}")
    logs = load_expert_logs(log_path)
    print(f"Loaded {len(logs)} questions\n")
    
    # Analyze activations
    print("Analyzing expert activations...")
    cat_layer_expert = analyze_expert_activations(logs)
    
    # Get top N
    top_experts = get_top_n_experts(cat_layer_expert, n=args.top_n)
    
    # Print results
    print_results(top_experts)
    
    # Write summaries
    csv_output = args.csv_output or str(results_dir / f'top_{args.top_n}_experts.csv')
    json_output = args.json_output or str(results_dir / f'top_{args.top_n}_experts.json')
    
    write_summary_csv(top_experts, csv_output)
    write_summary_json(top_experts, json_output)
    
    print(f"\nAnalysis complete!")


if __name__ == '__main__':
    main()
