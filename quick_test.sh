#!/bin/bash
echo "================================"
echo "Quick Ray Tracer Test"
echo "================================"
echo

echo "[1/3] Testing C++ CPU version..."
./ray_tracer_single --small --simple > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ C++ CPU version works!"
else
    echo "✗ C++ CPU version failed!"
fi

echo
echo "[2/3] Testing Numba Python version..."
./ray_tracer_numba --small --simple --quiet > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Numba Python version works!"
else
    echo "✗ Numba Python version failed!"
fi

echo
echo "[3/3] Checking output files..."
if [ -f "output_complex.ppm" ]; then
    echo "✓ C++ output file created"
fi
if [ -f "output_numba.ppm" ]; then
    echo "✓ Numba output file created"
fi

echo
echo "================================"
echo "Test Complete!"
echo "================================"
