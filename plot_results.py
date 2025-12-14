#!/usr/bin/env python3
"""
Plot benchmark results from benchmark_results.json
Creates performance comparison charts
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filename="benchmark_results.json"):
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_performance_comparison(results, output_file="benchmark_comparison.png"):
    """Create a bar chart comparing all implementations"""
    
    # Extract data
    configs = list(results.keys())
    implementations = ['cpu', 'cuda', 'numba', 'openacc']
    impl_names = ['C++ CPU', 'CUDA GPU', 'Numba Python', 'OpenACC']
    
    # Prepare data for plotting
    data = {}
    for impl in implementations:
        data[impl] = []
        for config in configs:
            if impl in results[config]:
                data[impl].append(results[config][impl]['avg'])
            else:
                data[impl].append(np.nan)  # Use NaN instead of 0 for log scale
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Average execution time (bar chart)
    x = np.arange(len(configs))
    width = 0.2
    
    for i, impl in enumerate(implementations):
        offset = (i - 1.5) * width
        ax1.bar(x + offset, data[impl], width, label=impl_names[i])
    
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Ray Tracer Performance Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(' (', '\n(') for c in configs], fontsize=9)
    ax1.set_yscale('log')  # Use logarithmic scale
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, which='both')
    
    # Plot 2: Speedup vs CPU baseline (bar chart)
    # Only include configs that have CPU baseline
    configs_with_cpu = [c for c in configs if 'cpu' in results[c]]
    
    for config_idx, config in enumerate(configs_with_cpu):
        cpu_time = results[config]['cpu']['avg']
        speedups = []
        labels = []
        
        for impl, name in zip(implementations[1:], impl_names[1:]):  # Skip CPU
            if impl in results[config]:
                speedup = cpu_time / results[config][impl]['avg']
                speedups.append(speedup)
                labels.append(name)
        
        if speedups:  # Only plot if we have data
            x_pos = np.arange(len(speedups))
            offset = (config_idx - len(configs_with_cpu)/2 + 0.5) * 0.25
            ax2.bar(x_pos + offset, speedups, 0.25, label=config)
    
    ax2.set_xlabel('Implementation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup vs CPU (log scale)', fontsize=12, fontweight='bold')
    
    # Add note if some configs don't have CPU baseline
    title = 'Speedup Comparison (Higher is Better)'
    if len(configs_with_cpu) < len(configs):
        title += '\n(Configs without CPU baseline excluded)'
    ax2.set_title(title, fontsize=14, fontweight='bold')
    
    if labels:  # Only set ticks if we have data
        ax2.set_xticks(np.arange(len(labels)))
        ax2.set_xticklabels(labels)
    ax2.set_yscale('log')  # Use logarithmic scale
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='CPU Baseline')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance comparison to {output_file}")
    
    return fig

def plot_detailed_comparison(results, output_file="benchmark_detailed.png"):
    """Create detailed comparison with error bars"""
    
    configs = list(results.keys())
    implementations = ['cpu', 'cuda', 'numba', 'openacc']
    impl_names = ['C++ CPU', 'CUDA GPU', 'Numba Python', 'OpenACC']
    
    fig, axes = plt.subplots(1, len(configs), figsize=(18, 5))
    
    for idx, config in enumerate(configs):
        ax = axes[idx] if len(configs) > 1 else axes
        
        names = []
        means = []
        mins = []
        maxs = []
        
        for impl, name in zip(implementations, impl_names):
            if impl in results[config]:
                data = results[config][impl]
                names.append(name)
                means.append(data['avg'])
                mins.append(data['avg'] - data['min'])
                maxs.append(data['max'] - data['avg'])
        
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=[mins, maxs], capsize=5, alpha=0.7)
        
        # Color bars
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        for bar, color in zip(bars, colors[:len(bars)]):
            bar.set_color(color)
        
        ax.set_xlabel('Implementation', fontsize=11, fontweight='bold')
        ax.set_ylabel('Time (seconds, log scale)', fontsize=11, fontweight='bold')
        ax.set_title(config, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yscale('log')  # Use logarithmic scale
        ax.grid(axis='y', alpha=0.3, which='both')
        
        # Add value labels on bars
        for i, (mean, bar) in enumerate(zip(means, bars)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.4f}s', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Detailed Performance Comparison with Min/Max Range', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved detailed comparison to {output_file}")
    
    return fig

def print_summary_table(results):
    """Print a summary table of results"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80 + "\n")
    
    for config in results.keys():
        print(f"\n{config}")
        print("-" * 80)
        print(f"{'Implementation':<20} {'Avg Time':<15} {'Min Time':<15} {'Speedup vs CPU':<15}")
        print("-" * 80)
        
        # Check if CPU baseline exists
        has_cpu = 'cpu' in results[config]
        cpu_time = results[config]['cpu']['avg'] if has_cpu else None
        
        for impl in ['cpu', 'cuda', 'numba', 'openacc']:
            if impl in results[config]:
                data = results[config][impl]
                if cpu_time:
                    speedup = cpu_time / data['avg']
                    print(f"{data['description']:<20} {data['avg']:>13.6f}s {data['min']:>13.6f}s {speedup:>13.2f}x")
                else:
                    # No CPU baseline - just show times
                    print(f"{data['description']:<20} {data['avg']:>13.6f}s {data['min']:>13.6f}s {'N/A':>13}")
    
    print("\n" + "="*80 + "\n")

def main():
    print("Ray Tracer Benchmark Visualization\n")
    
    # Check if results file exists
    if not Path("benchmark_results.json").exists():
        print("ERROR: benchmark_results.json not found!")
        print("Run 'python3 benchmark_enhanced.py' first to generate results.")
        return 1
    
    # Load results
    print("Loading benchmark results...")
    results = load_results()
    
    # Print summary table
    print_summary_table(results)
    
    # Create plots
    print("Generating plots...")
    try:
        plot_performance_comparison(results)
        plot_detailed_comparison(results)
        print("\n✓ All plots generated successfully!")
        print("\nGenerated files:")
        print("  - benchmark_comparison.png")
        print("  - benchmark_detailed.png")
    except Exception as e:
        print(f"\nWARNING: Could not generate plots: {e}")
        print("Install matplotlib: pip install matplotlib")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
