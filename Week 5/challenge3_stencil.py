"""
CSS 314: Parallel Computing - Week 5 Lab Practicum
Challenge 3: Stencil Computation & Memory Bandwidth (2D Heat Diffusion)

Student Name: Ruslan Ussen
Student ID: 230103026
Date: October 1, 2026
Hardware: AMD Ryzen 7 7435HS (8 Cores / 16 Threads)
"""

import time
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def heat_step(u, u_next, alpha=0.20):
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4.0 * u[i, j]
            )

GRID_SIZE = 1500
STEPS = 300

# ---------------------------------------------
# Part A: float64 Benchmark
# ---------------------------------------------
print("--- Challenge 3: Heat Stencil (float64) ---")
u64 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
u64_next = np.zeros_like(u64)
u64[0, :] = 100.0
u64[:, 0] = 100.0
u64_next[0, :] = 100.0
u64_next[:, 0] = 100.0

# Warmup
heat_step(u64, u64_next)

# Benchmark Execution
start = time.perf_counter()
for step in range(STEPS):
    heat_step(u64, u64_next)
    # Pointer swap (avoids array allocation copies)
    u64, u64_next = u64_next, u64
elapsed64 = time.perf_counter() - start

cells_per_sec64 = (GRID_SIZE * GRID_SIZE * STEPS) / elapsed64 / 1e6
print(f"Heat Diffusion Complete (float64): {elapsed64:.3f} s")
print(f"Throughput (float64): {cells_per_sec64:.2f} Megacells/sec")

# ---------------------------------------------
# Part B: float32 Benchmark (Question 3B)
# ---------------------------------------------
print("\n--- Challenge 3: Heat Stencil (float32) ---")
u32 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
u32_next = np.zeros_like(u32)
u32[0, :] = 100.0
u32[:, 0] = 100.0
u32_next[0, :] = 100.0
u32_next[:, 0] = 100.0

# Warmup
heat_step(u32, u32_next)

# Benchmark Execution
start = time.perf_counter()
for step in range(STEPS):
    heat_step(u32, u32_next)
    # Pointer swap (avoids array allocation copies)
    u32, u32_next = u32_next, u32
elapsed32 = time.perf_counter() - start

cells_per_sec32 = (GRID_SIZE * GRID_SIZE * STEPS) / elapsed32 / 1e6
print(f"Heat Diffusion Complete (float32): {elapsed32:.3f} s")
print(f"Throughput (float32): {cells_per_sec32:.2f} Megacells/sec")

speedup = elapsed64 / elapsed32
print(f"\nPrecision Speedup Factor (T_float64 / T_float32): {speedup:.2f}x")
