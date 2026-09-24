import os
import sys
import time
import math
import csv
import threading
import numpy as np
import numba
from numba import njit, prange
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Variant A: Naive Unsynchronized Race Condition (Native OpenMP threads)
# -------------------------------------------------------------
@njit(parallel=True)
def run_naive_race_numba(n_steps: int) -> float:
    shared_sum = np.zeros(1, dtype=np.float64)
    step = 1.0 / n_steps
    for i in prange(n_steps):
        x = (i + 0.5) * step
        # Unprotected shared memory write -> true hardware race condition
        shared_sum[0] += 4.0 / (1.0 + x * x)
    return shared_sum[0] * step

def run_naive_race(num_threads: int, n_steps: int = 10_000_000):
    numba.set_num_threads(num_threads)
    t0 = time.perf_counter()
    pi_approx = run_naive_race_numba(n_steps)
    t1 = time.perf_counter()
    return pi_approx, t1 - t0

# -------------------------------------------------------------
# Variant B: Synchronized Critical Section (Lock per step)
# -------------------------------------------------------------
def run_critical_section(num_threads: int, n_steps: int = 1_000_000):
    shared_sum = [0.0]
    step = 1.0 / n_steps
    chunk_size = n_steps // num_threads
    lock = threading.Lock()

    def worker(start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x = (i + 0.5) * step
            term = 4.0 / (1.0 + x * x)
            with lock: # Emulates #pragma omp critical
                shared_sum[0] += term

    threads = []
    t0 = time.perf_counter()
    for t in range(num_threads):
        start = t * chunk_size
        end = n_steps if t == num_threads - 1 else start + chunk_size
        th = threading.Thread(target=worker, args=(start, end))
        threads.append(th)
        th.start()

    for th in threads:
        th.join()
    t1 = time.perf_counter()

    pi_approx = shared_sum[0] * step
    return pi_approx, t1 - t0

# -------------------------------------------------------------
# Variant C: Parallel Tree Reduction (Numba OpenMP-style backend)
# -------------------------------------------------------------
@njit
def calc_pi_serial(num_steps: int) -> float:
    step = 1.0 / num_steps
    total_sum = 0.0
    for i in range(num_steps):
        x = (i + 0.5) * step
        total_sum += 4.0 / (1.0 + x * x)
    return total_sum * step

@njit(parallel=True)
def calc_pi_reduction(num_steps: int) -> float:
    step = 1.0 / num_steps
    total_sum = 0.0
    for i in prange(num_steps):
        x = (i + 0.5) * step
        total_sum += 4.0 / (1.0 + x * x) # Tree reduction
    return total_sum * step

# -------------------------------------------------------------
# Execution & Benchmarking
# -------------------------------------------------------------
def benchmark_lab2(base_dir: str):
    print("==================================================")
    print(" LAB 2: NUMERICAL INTEGRATION & REDUCTIONS")
    print("==================================================")

    # 1. Task 2.1: Race Condition Quantification
    print("\n--- Task 2.1: Race Condition Quantification ---")
    race_threads = [1, 2, 4, 8]
    n_race = 1_000_000
    race_rows = []
    print(f"{'Threads (P)':<12}{'Computed Pi':<18}{'Absolute Error':<18}{'Time (s)':<12}")
    print("-" * 60)
    for p in race_threads:
        pi_val, dt = run_naive_race(p, n_race)
        err = abs(pi_val - math.pi)
        print(f"{p:<12}{pi_val:<18.10f}{err:<18.2e}{dt:<12.4f}")
        race_rows.append({"threads": p, "pi_computed": f"{pi_val:.10f}", "error": f"{err:.6e}", "time_s": f"{dt:.4f}"})

    race_csv = os.path.join(base_dir, "task2_race.csv")
    with open(race_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threads", "pi_computed", "error", "time_s"])
        w.writeheader()
        w.writerows(race_rows)

    # 2. Task 2.2: Critical Section Overhead
    print("\n--- Task 2.2: Critical Section Overhead ---")
    n_crit = 500_000  # 500k steps with fine-grained lock per step
    # Fair baseline: single thread loop without lock
    t_start = time.perf_counter()
    step_c = 1.0 / n_crit
    base_sum = 0.0
    for i in range(n_crit):
        x = (i + 0.5) * step_c
        base_sum += 4.0 / (1.0 + x * x)
    t_base = time.perf_counter() - t_start
    pi_base = base_sum * step_c
    print(f"Single-thread Baseline (no lock): {t_base:.4f} s | Pi: {pi_base:.10f}")
    
    crit_rows = [{"config": "Baseline (Single Thread, No Lock)", "threads": 1, "time_s": f"{t_base:.4f}", "overhead_pct": "0.0%"}]
    for p in [1, 2, 4]:
        pi_crit, t_crit = run_critical_section(p, n_crit)
        overhead = ((t_crit - t_base) / t_base) * 100.0
        print(f"Critical Section (P={p}): {t_crit:.4f} s | Overhead: {overhead:+.1f}% | Pi: {pi_crit:.10f}")
        crit_rows.append({"config": f"Critical Section P={p}", "threads": p, "time_s": f"{t_crit:.4f}", "overhead_pct": f"{overhead:.1f}%"})

    crit_csv = os.path.join(base_dir, "task2_critical.csv")
    with open(crit_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "threads", "time_s", "overhead_pct"])
        w.writeheader()
        w.writerows(crit_rows)

    # 3. Task 2.3 & 2.4: Strong Scaling Benchmark for Parallel Reduction
    print("\n--- Task 2.3 & 2.4: Strong Scaling Benchmark (Numba Reduction) ---")
    N_REDUCTION = 100_000_000
    p_values = [1, 2, 4, 8, 16]
    trials = 5

    # Warm-up JIT
    _ = calc_pi_reduction(1000)

    # Serial baseline
    _ = calc_pi_serial(1000)
    t0 = time.perf_counter()
    pi_s = calc_pi_serial(N_REDUCTION)
    t_serial = time.perf_counter() - t0
    print(f"Serial Baseline N={N_REDUCTION:,}: {t_serial:.4f} s (Pi={pi_s:.10f})")

    scaling_rows = []
    avg_times = {}

    print(f"{'Threads (P)':<12}{'T_avg (s)':<14}{'Speedup S(P)':<16}{'Efficiency E(P)':<16}")
    print("-" * 60)

    for p in p_values:
        numba.set_num_threads(p)
        trial_times = []
        for _ in range(trials):
            t_start = time.perf_counter()
            _ = calc_pi_reduction(N_REDUCTION)
            trial_times.append(time.perf_counter() - t_start)
        
        t_avg = sum(trial_times) / trials
        avg_times[p] = t_avg
        speedup = avg_times[1] / t_avg
        efficiency = speedup / p
        print(f"{p:<12}{t_avg:<14.4f}{speedup:<16.2f}x{efficiency*100:<15.1f}%")
        scaling_rows.append({
            "threads": p,
            "t_avg_s": f"{t_avg:.5f}",
            "speedup": f"{speedup:.4f}",
            "efficiency_pct": f"{efficiency*100:.2f}"
        })

    scaling_csv = os.path.join(base_dir, "task2_scaling.csv")
    with open(scaling_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threads", "t_avg_s", "speedup", "efficiency_pct"])
        w.writeheader()
        w.writerows(scaling_rows)

    # 4. Generate Plot (Speedup vs Linear Ideal)
    plt.figure(figsize=(9, 5.5), dpi=300)
    threads = p_values
    emp_speedup = [avg_times[1] / avg_times[p] for p in threads]
    ideal_speedup = threads

    plt.plot(threads, ideal_speedup, 'r--', label='Ideal Linear Speedup ($S=P$)', linewidth=2)
    plt.plot(threads, emp_speedup, 'b-o', label='Measured Speedup (Numba Reduction)', linewidth=2.5, markersize=8)

    plt.title(r"Lab 2: Strong Scaling of $\pi$ Parallel Reduction ($N=10^8$)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Thread Count ($P$)", fontsize=11, fontweight='bold')
    plt.ylabel("Speedup $S(P) = T(1) / T(P)$", fontsize=11, fontweight='bold')
    plt.xticks(threads)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(base_dir, "speedup_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Speedup plot saved to {plot_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_lab2(base_dir)
