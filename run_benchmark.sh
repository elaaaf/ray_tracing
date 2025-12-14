#!/bin/bash
# Quick benchmark runner for Linux/Mac

echo "========================================"
echo "Ray Tracer GPU vs CPU Benchmark"
echo "========================================"
echo

echo "[1/3] Building projects..."
make all
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo
echo "[2/3] Running benchmark..."
python3 benchmark.py

echo
echo "[3/3] Done! Check benchmark_results.json"
echo

