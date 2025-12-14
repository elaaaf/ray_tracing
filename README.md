# 🚀 Ray Tracer Performance Benchmark

Multi-platform ray tracing implementation comparing **C++ CPU, CUDA GPU, Numba Python, and OpenACC** performance.

## Quick Start

```bash
# Run benchmark (tests all 4 implementations)
python3 benchmark_enhanced.py

# Visualize results (generates log-scale plots)
python3 plot_results.py

# Quick validation
make test
```

## 📊 Implementations

| Implementation | Description | Acceleration |
|----------------|-------------|--------------|
| **C++ CPU** | Single-threaded baseline | None |
| **CUDA GPU** | Custom CUDA kernels | NVIDIA GPU |
| **Numba Python** | JIT-compiled Python (auto GPU/CPU fallback) | GPU or Multi-core CPU |
| **OpenACC** | Directive-based | GPU |

## 🛠️ Build

```bash
# Build all implementations
make all

# Build individual versions
make ray_tracer_single    # C++ CPU
make ray_tracer_cuda       # CUDA GPU (requires nvcc)
make ray_tracer_numba      # Numba Python wrapper
make ray_tracer_openacc    # OpenACC (requires nvc++)

# Clean up
make clean
```

## 💻 Usage

### Individual Implementations

```bash
# C++ CPU
./ray_tracer_single [--small] [--simple]

# CUDA GPU
./ray_tracer_cuda [--small] [--simple]

# Numba Python
python3 ray_tracer_numba.py [--small] [--simple] [--force-cpu] [--quiet]

# OpenACC
./ray_tracer_openacc [--small] [--simple]
```

### Options
- `--small`: 400x300 resolution (faster)
- `--simple`: Simple scene (4 spheres, 1 light)
- `--extreme`: **Extreme scene (80 spheres, 3 lights)** - Numba only. Other implementations ignore this flag.
- Default: 800x600 resolution, complex scene (C++/CUDA/OpenACC: ~70+ spheres, Numba: 5 spheres)

**Note**: The C++, CUDA, and OpenACC implementations already have a complex scene with 70+ spheres by default. Numba uses 5 spheres by default, hence the `--extreme` flag to match the complexity.

### Benchmarking

```bash
# Enhanced benchmark (all implementations, multiple configs)
python3 benchmark_enhanced.py

# Original benchmark (CPU vs CUDA only)
python3 benchmark.py

# Quick test
./quick_test.sh
```

## 📈 Visualization

```bash
# Generate performance plots (logarithmic scale for better comparison)
python3 plot_results.py

# Output:
#   - benchmark_comparison.png (log-scale bar charts with speedup)
#   - benchmark_detailed.png (log-scale with error bars)
```

**Note**: Plots use logarithmic scale to clearly show performance differences across orders of magnitude.

Requires: `pip install matplotlib`

## 🎯 Features

- **Blinn-Phong shading** with ambient, diffuse, and specular components
- **Multiple spheres** with varying materials
- **Multiple light sources** with different colors/intensities
- **Sky gradient** background
- **Gamma correction** for proper color output

## 📋 Requirements

### Base
- C++ compiler (g++ with C++17)
- Python 3.7+ with NumPy and Numba
- Make

### GPU (Optional)
- **CUDA**: NVIDIA GPU + CUDA Toolkit 11.0+
- **OpenACC**: NVIDIA HPC SDK (nvc++)

```bash
# Install Python dependencies
pip install numpy numba matplotlib
```

## 🔧 Numba Implementation Highlights

The Numba version (`ray_tracer_numba.py`) is unique:
- ✅ **Automatic CUDA detection** - uses GPU if available
- ✅ **CPU fallback** - optimized multi-core CPU execution
- ✅ **Same codebase** - unified Python code for both modes
- ✅ **No compilation** - JIT compiles at runtime

## 📊 Expected Performance

Typical speedups vs single-threaded CPU (hardware dependent):
- **CUDA GPU**: 10-50x faster
- **Numba (GPU)**: 8-40x faster  
- **Numba (CPU)**: 2-8x faster
- **OpenACC**: 5-30x faster

## 📁 Project Structure

```
├── ray_tracer.cpp              # C++ CPU implementation
├── ray_tracer_cuda.cu          # CUDA GPU implementation
├── ray_tracer_numba.py         # Numba Python (GPU/CPU)
├── ray_tracer_openacc.cpp      # OpenACC implementation
├── benchmark_enhanced.py       # Multi-implementation benchmark
├── plot_results.py             # Results visualization
├── quick_test.sh               # Fast validation
├── Makefile                    # Build automation
└── README.md                   # This file
```

## 🐛 Troubleshooting

**CUDA not available?**
- Numba automatically falls back to CPU mode
- Check: `python3 -c "from numba import cuda; print(cuda.is_available())"`

**Build errors?**
- Missing nvcc: Install CUDA Toolkit
- Missing nvc++: Install NVIDIA HPC SDK
- Use `make ray_tracer_single ray_tracer_numba` for CPU-only build

## 📚 Learn More

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Numba Documentation](https://numba.pydata.org/)
- [Ray Tracing in One Weekend](https://raytracing.github.io/)

---

**Built with ❤️ for parallel computing education**
