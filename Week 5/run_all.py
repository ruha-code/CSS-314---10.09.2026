"""
CSS 314: Parallel Computing - Week 5 Lab Practicum
Master Benchmark Runner & Telemetry Suite

Student Name: Ruslan Ussen
Student ID: 230103026
Date: October 1, 2026
Hardware: AMD Ryzen 7 7435HS (8 Cores / 16 Threads)
"""

import time
import os
import sys
import numpy as np
import numba
from numba import njit, prange
import matplotlib.pyplot as plt

def main():
    print("=" * 65)
    print("CSS 314: PARALLEL COMPUTING - WEEK 5 LAB PRACTICUM")
    print("Student: Ruslan Ussen | ID: 230103026 | Session: 02-N (Group 06-P)")
    print("Hardware: AMD Ryzen 7 7435HS (8 Cores / 16 Threads)")
    print(f"Numba Version: {numba.__version__} | NumPy: {np.__version__}")
    print(f"Hardware Threads Detected: {numba.config.NUMBA_NUM_THREADS}")
    print("=" * 65)

    # ----------------------------------------------------
    # Challenge 1: Monte Carlo Simulation
    # ----------------------------------------------------
    print("\n[CHALLENGE 1: The Core Speedometer & Amdahl's Law]")
    
    @njit(parallel=True)
    def monte_carlo_pi(n_samples):
        inside_circle = 0
        for i in prange(n_samples):
            x = np.random.uniform(0.0, 1.0)
            y = np.random.uniform(0.0, 1.0)
            if x * x + y * y <= 1.0:
                inside_circle += 1
        return (4.0 * inside_circle) / n_samples

    _ = monte_carlo_pi(10_000)

    SAMPLES = 120_000_000
    thread_counts = [1, 2, 4, 8, numba.config.NUMBA_NUM_THREADS]
    thread_counts = sorted(list(set([t for t in thread_counts if t <= numba.config.NUMBA_NUM_THREADS])))

    print(f"{'Threads':<10} | {'Time (s)':<12} | {'Speedup':<10} | {'Efficiency (%)':<15}")
    print("-" * 55)

    t1_baseline = None
    c1_table = []
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
        c1_table.append((t, elapsed, speedup, efficiency, pi_est))
        print(f"{t:<10} | {elapsed:<12.4f} | {speedup:<10.2f}x | {efficiency:<15.1f}")

    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    # ----------------------------------------------------
    # Challenge 2: Load Imbalance & Dynamic Scheduling
    # ----------------------------------------------------
    print("\n[CHALLENGE 2: Load Imbalance & Dynamic Scheduling]")

    @njit(parallel=True)
    def render_mandelbrot_rows(h, w, max_iter):
        img = np.zeros((h, w), dtype=np.int32)
        for r in prange(h):
            cy = -1.2 + (r / h) * 2.4
            for c in range(w):
                cx = -2.0 + (c / w) * 2.5
                z_real, z_imag = 0.0, 0.0
                it = 0
                while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                    next_real = z_real * z_real - z_imag * z_imag + cx
                    z_imag = 2.0 * z_real * z_imag + cy
                    z_real = next_real
                    it += 1
                img[r, c] = it
        return img

    @njit(parallel=True)
    def render_mandelbrot_cols(h, w, max_iter):
        img = np.zeros((h, w), dtype=np.int32)
        for c in prange(w):
            cx = -2.0 + (c / w) * 2.5
            for r in range(h):
                cy = -1.2 + (r / h) * 2.4
                z_real, z_imag = 0.0, 0.0
                it = 0
                while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                    next_real = z_real * z_real - z_imag * z_imag + cx
                    z_imag = 2.0 * z_real * z_imag + cy
                    z_real = next_real
                    it += 1
                img[r, c] = it
        return img

    _ = render_mandelbrot_rows(100, 100, 50)
    _ = render_mandelbrot_cols(100, 100, 50)

    H, W, MAX_IT = 2500, 2500, 1000

    t0 = time.perf_counter()
    grid_rows = render_mandelbrot_rows(H, W, MAX_IT)
    t_rows = time.perf_counter() - t0

    t1 = time.perf_counter()
    grid_cols = render_mandelbrot_cols(H, W, MAX_IT)
    t_cols = time.perf_counter() - t1

    print(f"Row-Parallel Render Time:    {t_rows:.4f} s")
    print(f"Column-Parallel Render Time: {t_cols:.4f} s")

    img_path = os.path.join(os.path.dirname(__file__), 'mandelbrot_output.png')
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_rows, cmap='magma', extent=[-2.0, 0.5, -1.2, 1.2])
    plt.title(f"Mandelbrot {H}x{W} (Render: {t_rows:.2f}s)")
    plt.axis('off')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visual artifact to: {img_path}")

    # ----------------------------------------------------
    # Challenge 3: Stencil Computation & Memory Bandwidth
    # ----------------------------------------------------
    print("\n[CHALLENGE 3: Stencil Computation & Memory Bandwidth]")

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

    # float64
    u64 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    u64_next = np.zeros_like(u64)
    u64[0, :] = 100.0
    u64[:, 0] = 100.0
    u64_next[0, :] = 100.0
    u64_next[:, 0] = 100.0
    heat_step(u64, u64_next)

    start = time.perf_counter()
    for step in range(STEPS):
        heat_step(u64, u64_next)
        u64, u64_next = u64_next, u64
    t_stencil_f64 = time.perf_counter() - start
    mcells_f64 = (GRID_SIZE * GRID_SIZE * STEPS) / t_stencil_f64 / 1e6

    print(f"Heat Diffusion Complete (float64): {t_stencil_f64:.4f} s")
    print(f"Throughput (float64): {mcells_f64:.2f} Megacells/sec")

    # float32
    u32 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    u32_next = np.zeros_like(u32)
    u32[0, :] = 100.0
    u32[:, 0] = 100.0
    u32_next[0, :] = 100.0
    u32_next[:, 0] = 100.0
    heat_step(u32, u32_next)

    start = time.perf_counter()
    for step in range(STEPS):
        heat_step(u32, u32_next)
        u32, u32_next = u32_next, u32
    t_stencil_f32 = time.perf_counter() - start
    mcells_f32 = (GRID_SIZE * GRID_SIZE * STEPS) / t_stencil_f32 / 1e6

    print(f"Heat Diffusion Complete (float32): {t_stencil_f32:.4f} s")
    print(f"Throughput (float32): {mcells_f32:.2f} Megacells/sec")
    f_speedup = t_stencil_f64 / t_stencil_f32
    print(f"Precision speedup factor (f64 / f32): {f_speedup:.2f}x")

    # ----------------------------------------------------
    # Summary Table Output
    # ----------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL SUMMARY DATA FOR TABLE 1")
    print("=" * 65)
    for t, el, sp, eff, _ in c1_table:
        print(f"C1 - Threads: {t:<2} | Time: {el:.4f}s | Speedup: {sp:.2f}x | Efficiency: {eff:.1f}%")
    print(f"C2 - Mandelbrot (Rows): {t_rows:.4f}s")
    print(f"C2 - Mandelbrot (Cols): {t_cols:.4f}s")
    print(f"C3 - Heat Stencil f64 : {t_stencil_f64:.4f}s ({mcells_f64:.2f} Mcells/s)")
    print(f"C3 - Heat Stencil f32 : {t_stencil_f32:.4f}s ({mcells_f32:.2f} Mcells/s)")
    print("=" * 65)

    # ----------------------------------------------------
    # Write Log File
    # ----------------------------------------------------
    log_path = os.path.join(os.path.dirname(__file__), 'results_230103026.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("CSS 314: PARALLEL COMPUTING - WEEK 5 LAB PRACTICUM RESULTS\n")
        f.write("Student: Ruslan Ussen | ID: 230103026 | Session: 02-N (Group 06-P)\n")
        f.write("Hardware: AMD Ryzen 7 7435HS (8 Cores / 16 Threads)\n")
        f.write("Date: 2026-10-01\n")
        f.write("=" * 60 + "\n\n")
        f.write("TABLE 1: BENCHMARK RESULTS\n")
        f.write(f"{'Challenge':<28} | {'Threads':<12} | {'Time (s)':<10} | {'Speedup':<8} | {'Efficiency':<10}\n")
        f.write("-" * 75 + "\n")
        for t, el, sp, eff, _ in c1_table:
            lbl = f"C1: Monte Carlo ({t}T)"
            f.write(f"{lbl:<28} | {t:<12} | {el:<10.4f} | {sp:<8.2f}x | {eff:.1f}%\n")
        f.write(f"{'C2: Mandelbrot (Rows)':<28} | {'16':<12} | {t_rows:<10.4f} | {'N/A':<8} | {'N/A':<10}\n")
        f.write(f"{'C2: Mandelbrot (Cols)':<28} | {'16':<12} | {t_cols:<10.4f} | {'N/A':<8} | {'N/A':<10}\n")
        f.write(f"{'C3: Heat Stencil (float64)':<28} | {'16':<12} | {t_stencil_f64:<10.4f} | {'N/A':<8} | {'N/A':<10}\n")
        f.write(f"{'C3: Heat Stencil (float32)':<28} | {'16':<12} | {t_stencil_f32:<10.4f} | {'N/A':<8} | {'N/A':<10}\n")
        f.write("\nThroughput Breakdown:\n")
        f.write(f"  float64 Stencil: {mcells_f64:.2f} Megacells/sec\n")
        f.write(f"  float32 Stencil: {mcells_f32:.2f} Megacells/sec\n")
        f.write(f"  Precision Speedup (f64 / f32): {f_speedup:.2f}x\n")
    print(f"Log written to: {log_path}")

    # ----------------------------------------------------
    # Write CSV File
    # ----------------------------------------------------
    csv_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Benchmark Challenge,Active Threads,Array Size / Samples,Time (seconds),Speedup Factor,Efficiency (%)\n")
        for t, el, sp, eff, _ in c1_table:
            lbl = f"Challenge 1: Monte Carlo ({t} Threads)" if t > 1 else "Challenge 1: Monte Carlo (1 Thread Base)"
            f.write(f'"{lbl}",{t},120000000,{el:.4f},{sp:.2f}x,{eff:.1f}%\n')
        f.write(f'"Challenge 2: Mandelbrot (Rows)",16,2500 x 2500,{t_rows:.4f},N/A,N/A\n')
        f.write(f'"Challenge 2: Mandelbrot (Cols)",16,2500 x 2500,{t_cols:.4f},N/A,N/A\n')
        f.write(f'"Challenge 3: Heat Stencil (float64)",16,1500 x 1500 x 300,{t_stencil_f64:.4f},N/A,N/A\n')
        f.write(f'"Challenge 3: Heat Stencil (float32)",16,1500 x 1500 x 300,{t_stencil_f32:.4f},N/A,N/A\n')
    print(f"CSV written to: {csv_path}")

if __name__ == '__main__':
    main()
