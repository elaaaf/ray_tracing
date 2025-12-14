# Makefile for Ray Tracer - CPU and GPU versions
CXX = g++
NVCC = nvcc
CXXFLAGS_BASE = -std=c++17 -Wall
CXXFLAGS_OACC = -gpu=cc90 -Minfo=accel
CXXFLAGS_SINGLE = $(CXXFLAGS_BASE) -Wno-unknown-pragmas
NVCCFLAGS = -O3 --std=c++17 -allow-unsupported-compiler

# Detect available compilers
NVCC_EXISTS := $(shell command -v nvcc 2> /dev/null)
NVC_EXISTS := $(shell command -v nvc++ 2> /dev/null)

# Build targets based on available compilers
TARGETS := ray_tracer_single ray_tracer_numba
ifdef NVCC_EXISTS
    TARGETS += ray_tracer_cuda
endif
ifdef NVC_EXISTS
    TARGETS += ray_tracer_openacc
endif

# Build all available versions
all: $(TARGETS)
	@echo "Built all available implementations"

# Single-threaded CPU version (better for profiling)
ray_tracer_single: ray_tracer.cpp
	$(CXX) $(CXXFLAGS_SINGLE) $< -o $@ -lm
	@echo "Built single-threaded version: ray_tracer_single"

# CUDA GPU version (optional - only if nvcc is available)
ray_tracer_cuda: ray_tracer_cuda.cu
	@if command -v nvcc >/dev/null 2>&1; then \
		$(NVCC) $(NVCCFLAGS) $< -o $@; \
		echo "Built CUDA version: ray_tracer_cuda"; \
	else \
		echo "SKIP: nvcc not found - CUDA version not built"; \
	fi

#OPENACC (optional - only if nvc++ is available)
ray_tracer_openacc: ray_tracer_openacc.cpp
	@if command -v nvc++ >/dev/null 2>&1; then \
		nvc++ $(CXXFLAGS_OACC) $< -o $@ -lm; \
		echo "Built OpenACC version: ray_tracer_openacc"; \
	else \
		echo "SKIP: nvc++ not found - OpenACC version not built"; \
	fi

#Numba Python (creates executable wrapper)
ray_tracer_numba: ray_tracer_numba.py
	@echo "#!/bin/bash" > ray_tracer_numba
	@echo "exec python3 $(PWD)/ray_tracer_numba.py \"\$$@\"" >> ray_tracer_numba
	@chmod +x ray_tracer_numba
	@echo "Built numba version: ray_tracer_numba"

clean:
	rm -f ray_tracer ray_tracer_single ray_tracer_cuda ray_tracer_openacc ray_tracer_numba *.ppm *.png *.nsys-rep *.sqlite benchmark_results.json
	rm -rf __pycache__

benchmark: all
	@echo "Running benchmark..."
	python3 benchmark.py

benchmark-enhanced: all
	@echo "Running enhanced multi-implementation benchmark..."
	python3 benchmark_enhanced.py

plot: benchmark_results.json
	@echo "Generating performance plots..."
	python3 plot_results.py

test: ray_tracer_single ray_tracer_numba
	@echo "Running quick tests..."
	@./quick_test.sh

# Force build CUDA even if nvcc is not detected (will fail if truly unavailable)
force-cuda:
	$(NVCC) $(NVCCFLAGS) ray_tracer_cuda.cu -o ray_tracer_cuda
	@echo "Built CUDA version: ray_tracer_cuda"

# Force build OpenACC even if nvc++ is not detected (will fail if truly unavailable)
force-openacc:
	nvc++ $(CXXFLAGS_OACC) ray_tracer_openacc.cpp -o ray_tracer_openacc -lm
	@echo "Built OpenACC version: ray_tracer_openacc"

.PHONY: all clean benchmark benchmark-enhanced plot test force-cuda force-openacc
