"""
CSS 314: Parallel Computing - Week 5 Lab Practicum
Challenge 1: The Core Speedometer & Amdahl's Law (Monte Carlo Pi)

Student Name: Ruslan Ussen
Student ID: 230103026
Date: October 1, 2026
Hardware: AMD Ryzen 7 7435HS (8 Cores / 16 Threads)
"""

import time
import numpy as np
import numba
from numba import njit, prange

print(f"Hardware Threads Detected: {numba.config.NUMBA_NUM_THREADS}")

@njit(parallel=True)
def monte_carlo_pi(n_samples):
    inside_circle = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x * x + y * y <= 1.0:
            inside_circle += 1
    return (4.0 * inside_circle) / n_samples

# JIT Warmup (excludes JIT compilation overhead from benchmarks)
_ = monte_carlo_pi(10_000)

SAMPLES = 120_000_000
thread_counts = [1, 2, 4, 8, numba.config.NUMBA_NUM_THREADS]
# Filter out duplicates and invalid counts
thread_counts = sorted(list(set([t for t in thread_counts if t <= numba.config.NUMBA_NUM_THREADS])))

print(f"{'Threads':<10} | {'Time (s)':<12} | {'Speedup':<10} | {'Efficiency (%)':<15}")
print("-" * 55)

t1_baseline = None
results = []
for t in thread_counts:
    numba.set_num_threads(t)
    start = time.perf_counter()
    pi_est = monte_carlo_pi(SAMPLES)
    elapsed = time.perf_counter() - start
    if t == 1:
        t1_baseline = elapsed
        speedup = 1.0
        efficiency = 100.0
    else:
        speedup = t1_baseline / elapsed
        efficiency = (speedup / t) * 100.0
    results.append((t, elapsed, speedup, efficiency, pi_est))
    print(f"{t:<10} | {elapsed:<12.4f} | {speedup:<10.2f}x | {efficiency:<15.1f}")

# Reset default threads
numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
