#!/usr/bin/env python3
"""
GPU vs CPU Ray Tracer Benchmark (Enhanced with Numba support)
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
    print(f"\n{Color.BOLD}{Color.HEADER}{'='*80}{Color.END}")
    print(f"{Color.BOLD}{Color.HEADER}{text.center(80)}{Color.END}")
    print(f"{Color.BOLD}{Color.HEADER}{'='*80}{Color.END}\n")

def print_section(text):
    print(f"\n{Color.BOLD}{Color.CYAN}{'─'*80}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{text}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'─'*80}{Color.END}")

def run_benchmark(executable, args=[], description="", runs=3):
    """Run a benchmark multiple times and return average time"""
    # Handle both single executables and command strings (like "python3 script.py")
    if executable is None:
        print(f"{Color.YELLOW}SKIP: {description} not available{Color.END}")
        return None
    
    # Check if executable exists (for single executables, not commands)
    if not executable.startswith("python") and not os.path.exists(executable):
        print(f"{Color.YELLOW}SKIP: {executable} not found{Color.END}")
        return None
    
    print(f"{Color.BLUE}Running: {description}...{Color.END}")
    times = []
    
    # Check if executable supports --quiet flag
    supports_quiet = "numba" in executable.lower() or "numba" in description.lower()
    
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        start = time.time()
        
        try:
            # Build command - handle both single executable and command strings
            if " " in executable:
                # It's a command string like "python3 script.py"
                cmd_parts = executable.split()
                cmd_args = cmd_parts + args
            else:
                cmd_args = [executable] + args
            
            if supports_quiet:
                cmd_args.append("--quiet")
            
            result = subprocess.run(
                cmd_args,
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

def print_comparison(results, baseline_key='cpu'):
    """Print detailed comparison between multiple implementations"""
    print_section("Performance Comparison")
    
    has_baseline = baseline_key in results and results[baseline_key] is not None
    
    if not has_baseline:
        print(f"{Color.YELLOW}Note: No CPU baseline for this configuration{Color.END}")
        print(f"\n{Color.BOLD}{'Implementation':<20} {'Avg Time':<15} {'Best Time':<15}{Color.END}")
        print("─" * 50)
        
        for key, result in results.items():
            if result is not None:
                print(f"{result['description']:<20} {result['avg']:>13.3f}s {result['min']:>13.3f}s")
        
        print("\n" + "─" * 50)
        return
    
    baseline = results[baseline_key]
    
    # Summary table
    print(f"\n{Color.BOLD}{'Implementation':<20} {'Avg Time':<15} {'Best Time':<15} {'Speedup':<15}{Color.END}")
    print("─" * 65)
    
    # Print baseline first
    print(f"{baseline['description']:<20} {baseline['avg']:>13.3f}s {baseline['min']:>13.3f}s {'1.00x':>13}")
    
    # Print other implementations
    for key, result in results.items():
        if key != baseline_key and result is not None:
            speedup = baseline['avg'] / result['avg']
            best_speedup = baseline['min'] / result['min']
            print(f"{result['description']:<20} {result['avg']:>13.3f}s {result['min']:>13.3f}s", end="")
            print(f" {speedup:>13.2f}x")
    
    print("\n" + "─" * 65)
    
    # Calculate best speedup
    best_speedup = 0
    best_impl = None
    for key, result in results.items():
        if key != baseline_key and result is not None:
            speedup = baseline['avg'] / result['avg']
            if speedup > best_speedup:
                best_speedup = speedup
                best_impl = result['description']
    
    if best_impl:
        print(f"\n{Color.BOLD}Best: {best_impl} is {format_speedup(best_speedup)} faster than baseline!{Color.END}\n")

def save_results(results, filename="benchmark_results.json"):
    """Save benchmark results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{Color.GREEN}Results saved to {filename}{Color.END}")

def main():
    print_header("Ray Tracer Benchmark Suite - Multi-Implementation Comparison")
    
    print(f"{Color.BOLD}System Information:{Color.END}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working Directory: {os.getcwd()}")
    
    # Check for executables
    cpu_exe = "./ray_tracer_single"
    gpu_exe = "./ray_tracer_cuda"
    openacc_exe = "./ray_tracer_openacc"
    
    # Numba can be Python script or wrapper
    if os.path.exists("./ray_tracer_numba"):
        numba_exe = "./ray_tracer_numba"
    elif os.path.exists("ray_tracer_numba.py"):
        numba_exe = "python3 ray_tracer_numba.py"
    else:
        numba_exe = None
    
    if sys.platform == "win32":
        cpu_exe = "ray_tracer_single.exe"
        gpu_exe = "ray_tracer_cuda.exe"
        openacc_exe = "ray_tracer_openacc.exe"
        if not numba_exe:
            numba_exe = "python ray_tracer_numba.py"
    
    # Benchmark configurations
    configs = [
        {
            'name': 'Standard Resolution (800x600)',
            'args': [],
            'runs': 3
        },
        {
            'name': 'Small Resolution (400x300)',
            'args': ['--small'],
            'runs': 5
        },
        {
            'name': 'Simple Scene (800x600)',
            'args': ['--simple'],
            'runs': 5
        },
        {
            'name': 'Extreme Scene - 80 Spheres (800x600)',
            'args': ['--extreme'],  # Numba uses 80 spheres, others use their default complex (~70+ spheres)
            'runs': 3,
            'numba_only': False  # Run on all implementations
        }
    ]
    
    all_results = {}
    
    for config in configs:
        print_section(f"Benchmark: {config['name']}")
        
        config_results = {}
        numba_only = config.get('numba_only', False)
        
        if not numba_only:
            # Run CPU benchmark (baseline)
            cpu_result = run_benchmark(
                cpu_exe,
                config['args'],
                "C++ CPU",
                config['runs']
            )
            if cpu_result:
                config_results['cpu'] = cpu_result
            
            # Run CUDA GPU benchmark
            gpu_result = run_benchmark(
                gpu_exe,
                config['args'],
                "CUDA GPU",
                config['runs']
            )
            if gpu_result:
                config_results['cuda'] = gpu_result
        
        # Run Numba Python benchmark (always)
        numba_result = run_benchmark(
            numba_exe,
            config['args'],
            "Numba Python",
            config['runs']
        )
        if numba_result:
            config_results['numba'] = numba_result
        
        if not numba_only:
            # Run OpenACC benchmark
            openacc_result = run_benchmark(
                openacc_exe,
                config['args'],
                "OpenACC",
                config['runs']
            )
        if openacc_result:
            config_results['openacc'] = openacc_result
        
        # Compare results
        if config_results:
            print_comparison(config_results, baseline_key='cpu')
            all_results[config['name']] = config_results
        
        print()
    
    # Final summary
    if all_results:
        print_header("Final Summary - All Configurations")
        
        for config_name, config_results in all_results.items():
            print(f"\n{Color.BOLD}{config_name}{Color.END}")
            print(f"{'Implementation':<20} {'Avg Time':<15} {'Speedup vs CPU':<20}")
            print("─" * 55)
            
            baseline = config_results.get('cpu')
            if baseline:
                print(f"{'C++ CPU':<20} {baseline['avg']:>13.3f}s {'1.00x':>18}")
                
                for key, result in config_results.items():
                    if key != 'cpu' and result is not None:
                        speedup = baseline['avg'] / result['avg']
                        print(f"{result['description']:<20} {result['avg']:>13.3f}s {speedup:>18.2f}x")
        
        # Save results
        save_results(all_results)
        
        print_section("Benchmark Complete")
        print(f"\n{Color.GREEN}Output files generated:{Color.END}")
        print("  - output_complex.ppm (CPU render)")
        print("  - output_cuda.ppm (CUDA GPU render)")
        print("  - output_numba.ppm (Numba Python render)")
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
