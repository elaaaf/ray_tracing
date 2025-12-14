#!/usr/bin/env python3
"""
GPU vs CPU Ray Tracer Benchmark
Comprehensive performance comparison with detailed metrics
"""

import subprocess
import time
import sys
import os
import json
from pathlib import Path

class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Color.BOLD}{Color.HEADER}{'='*70}{Color.END}")
    print(f"{Color.BOLD}{Color.HEADER}{text.center(70)}{Color.END}")
    print(f"{Color.BOLD}{Color.HEADER}{'='*70}{Color.END}\n")

def print_section(text):
    print(f"\n{Color.BOLD}{Color.CYAN}{'─'*70}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{text}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'─'*70}{Color.END}")

def run_benchmark(executable, args=[], description="", runs=3):
    """Run a benchmark multiple times and return average time"""
    if not os.path.exists(executable):
        print(f"{Color.RED}ERROR: {executable} not found. Run 'make all' first.{Color.END}")
        return None
    
    print(f"{Color.BLUE}Running: {description}...{Color.END}")
    times = []
    
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        start = time.time()
        
        try:
            result = subprocess.run(
                [executable] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"{Color.GREEN}OK {elapsed:.3f}s{Color.END}")
            
            if result.returncode != 0:
                print(f"{Color.YELLOW}Warning: Non-zero return code{Color.END}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
        
        except subprocess.TimeoutExpired:
            print(f"{Color.RED}TIMEOUT{Color.END}")
            return None
        except Exception as e:
            print(f"{Color.RED}ERROR: {e}{Color.END}")
            return None
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"  {Color.GREEN}Avg: {avg_time:.3f}s | Min: {min_time:.3f}s | Max: {max_time:.3f}s{Color.END}")
    
    return {
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
        'runs': times,
        'description': description
    }

def format_speedup(speedup):
    """Format speedup value with color coding"""
    if speedup >= 10:
        color = Color.GREEN
    elif speedup >= 5:
        color = Color.CYAN
    elif speedup >= 2:
        color = Color.BLUE
    else:
        color = Color.YELLOW
    
    return f"{color}{speedup:.2f}x{Color.END}"

def print_comparison(cpu_result, gpu_result):
    """Print detailed comparison between CPU and GPU results"""
    print_section("Performance Comparison")
    
    if cpu_result is None or gpu_result is None:
        print(f"{Color.RED}Cannot compare - missing results{Color.END}")
        return
    
    speedup = cpu_result['avg'] / gpu_result['avg']
    
    # Summary table
    print(f"\n{Color.BOLD}{'Metric':<30} {'CPU (C++)':<20} {'GPU (CUDA)':<20} {'Speedup':<15}{Color.END}")
    print("─" * 85)
    
    print(f"{'Average Time':<30} {cpu_result['avg']:>18.3f}s {gpu_result['avg']:>18.3f}s {speedup:>13.2f}x")
    print(f"{'Best Time':<30} {cpu_result['min']:>18.3f}s {gpu_result['min']:>18.3f}s {cpu_result['min']/gpu_result['min']:>13.2f}x")
    print(f"{'Worst Time':<30} {cpu_result['max']:>18.3f}s {gpu_result['max']:>18.3f}s {cpu_result['max']/gpu_result['max']:>13.2f}x")
    
    # Performance metrics
    cpu_mpixels = 0.48 / cpu_result['avg']  # 800x600 = 480,000 pixels
    gpu_mpixels = 0.48 / gpu_result['avg']
    
    print(f"{'Throughput (Mpixels/s)':<30} {cpu_mpixels:>18.2f} {gpu_mpixels:>18.2f} {gpu_mpixels/cpu_mpixels:>13.2f}x")
    
    # Energy efficiency estimate (relative)
    cpu_energy = cpu_result['avg'] * 100  # Assume 100W CPU
    gpu_energy = gpu_result['avg'] * 250  # Assume 250W GPU
    efficiency_ratio = cpu_energy / gpu_energy
    
    print(f"{'Est. Energy (J)':<30} {cpu_energy:>18.1f} {gpu_energy:>18.1f} {efficiency_ratio:>13.2f}x")
    
    print("\n" + "─" * 85)
    print(f"\n{Color.BOLD}GPU is {format_speedup(speedup)} faster than CPU!{Color.END}\n")
    
    # Performance insights
    print_section("Performance Insights")
    
    if speedup >= 10:
        print(f"{Color.GREEN}[+] Excellent GPU acceleration! The parallel nature of ray tracing is well-suited for GPU.{Color.END}")
    elif speedup >= 5:
        print(f"{Color.CYAN}[+] Good GPU speedup. Consider optimizing memory access patterns for more gains.{Color.END}")
    elif speedup >= 2:
        print(f"{Color.BLUE}[+] Moderate speedup. GPU overhead might be limiting performance on this workload.{Color.END}")
    else:
        print(f"{Color.YELLOW}[!] Lower than expected speedup. Check for bottlenecks in memory transfers or kernel launches.{Color.END}")
    
    print(f"\n  CPU Consistency: {(cpu_result['max'] - cpu_result['min']) / cpu_result['avg'] * 100:.1f}% variation")
    print(f"  GPU Consistency: {(gpu_result['max'] - gpu_result['min']) / gpu_result['avg'] * 100:.1f}% variation")

def save_results(results, filename="benchmark_results.json"):
    """Save benchmark results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{Color.GREEN}Results saved to {filename}{Color.END}")

def main():
    print_header("GPU vs CPU Ray Tracer Benchmark Suite")
    
    print(f"{Color.BOLD}System Information:{Color.END}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working Directory: {os.getcwd()}")
    
    # Check for executables
    cpu_exe = "./ray_tracer_single"
    gpu_exe = "./ray_tracer_cuda"
    
    if sys.platform == "win32":
        cpu_exe = "ray_tracer_single.exe"
        gpu_exe = "ray_tracer_cuda.exe"
    
    # Benchmark configurations
    configs = [
        {
            'name': 'Standard Resolution (800x600)',
            'cpu_args': [],
            'gpu_args': [],
            'runs': 3
        },
        {
            'name': 'Small Resolution (400x300)',
            'cpu_args': ['--small'],
            'gpu_args': ['--small'],
            'runs': 5
        },
        {
            'name': 'Simple Scene (800x600)',
            'cpu_args': ['--simple'],
            'gpu_args': ['--simple'],
            'runs': 5
        }
    ]
    
    all_results = {}
    
    for config in configs:
        print_section(f"Benchmark: {config['name']}")
        
        # Run CPU benchmark
        cpu_result = run_benchmark(
            cpu_exe,
            config['cpu_args'],
            f"CPU ({config['name']})",
            config['runs']
        )
        
        # Run GPU benchmark
        gpu_result = run_benchmark(
            gpu_exe,
            config['gpu_args'],
            f"GPU ({config['name']})",
            config['runs']
        )
        
        # Compare results
        if cpu_result and gpu_result:
            print_comparison(cpu_result, gpu_result)
            all_results[config['name']] = {
                'cpu': cpu_result,
                'gpu': gpu_result,
                'speedup': cpu_result['avg'] / gpu_result['avg']
            }
        
        print()
    
    # Final summary
    if all_results:
        print_header("Final Summary")
        
        print(f"{Color.BOLD}{'Configuration':<35} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<15}{Color.END}")
        print("─" * 80)
        
        for name, result in all_results.items():
            print(f"{name:<35} {result['cpu']['avg']:>13.3f}s {result['gpu']['avg']:>13.3f}s {format_speedup(result['speedup'])}")
        
        avg_speedup = sum(r['speedup'] for r in all_results.values()) / len(all_results)
        print("\n" + "─" * 80)
        print(f"{Color.BOLD}Average Speedup: {format_speedup(avg_speedup)}{Color.END}\n")
        
        # Save results
        save_results(all_results)
        
        print_section("Benchmark Complete")
        print(f"\n{Color.GREEN}Output files generated:{Color.END}")
        print("  - output_complex.ppm (CPU render)")
        print("  - output_cuda.ppm (GPU render)")
        print("  - benchmark_results.json (performance data)")
        
    else:
        print(f"\n{Color.RED}No valid results obtained. Please check that executables are built correctly.{Color.END}")
        print(f"\nTo build: {Color.BOLD}make all{Color.END}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}Benchmark interrupted by user{Color.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Color.RED}Error: {e}{Color.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

