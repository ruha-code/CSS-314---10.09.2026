import os
import sys
import time
import math
import io
import csv
import threading
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------------------
# Starter implementation from manual
# -------------------------------------------------------------
def worker_task(thread_id: int, team_size: int, output_list=None):
    native_tid = threading.get_native_id()
    role = "Master" if thread_id == 0 else "Worker"
    time.sleep(0.001 * (thread_id % 3))
    line = f"[{role}] Logical Rank: {thread_id} of {team_size} | Native OS TID: {native_tid}"
    if output_list is not None:
        output_list.append(line)
    return line

def run_team(num_threads: int, capture=False):
    lines = []
    lines.append(f"--- Forking a team of {num_threads} threads ---")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, tid, num_threads, lines if capture else None) for tid in range(num_threads)]
        for f in futures:
            res = f.result()
            if not capture:
                print(res)
    lines.append("--- Joined thread team. Execution returned to serial master ---\n")
    if capture:
        return "\n".join(lines)

# -------------------------------------------------------------
# Task 1.1: Verification of Non-Determinism
# -------------------------------------------------------------
def run_task_1_1(output_filepath: str):
    print("=== Task 1.1: Verification of Non-Determinism ===")
    results = []
    for run_idx in range(1, 11):
        header = f"=== Run {run_idx} of 10 ==="
        run_output = run_team(4, capture=True)
        full_block = f"{header}\n{run_output}\n"
        results.append(full_block)
        print(f"Completed run {run_idx}/10")
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.writelines(results)
    print(f"Task 1.1 output saved to {output_filepath}\n")

# -------------------------------------------------------------
# Task 1.2: Thread Oversubscription Sweep
# -------------------------------------------------------------
def run_task_1_2(csv_filepath: str):
    print("=== Task 1.2: Thread Oversubscription Sweep ===")
    thread_counts = [1, 2, 4, 8, 16, 32, 64]
    trials = 5
    
    def minimal_worker(tid):
        return threading.get_native_id()

    rows = []
    print(f"{'Threads (P)':<12}{'Mean Time (ms)':<16}{'Min Time (ms)':<16}{'Max Time (ms)':<16}")
    print("-" * 60)
    for p in thread_counts:
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=p) as executor:
                futures = [executor.submit(minimal_worker, tid) for tid in range(p)]
                for f in futures:
                    f.result()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0) # milliseconds
        
        mean_t = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        print(f"{p:<12}{mean_t:<16.3f}{min_t:<16.3f}{max_t:<16.3f}")
        rows.append({"threads": p, "mean_ms": f"{mean_t:.4f}", "min_ms": f"{min_t:.4f}", "max_ms": f"{max_t:.4f}"})

    with open(csv_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threads", "mean_ms", "min_ms", "max_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Task 1.2 sweep saved to {csv_filepath}\n")

# -------------------------------------------------------------
# Task 1.3: CPU Core Saturation Analysis
# -------------------------------------------------------------
def heavy_cpu_worker(tid: int, iters: int):
    # Artificial intensive compute workload: square roots & arithmetic
    acc = 0.0
    for i in range(iters):
        acc += math.sqrt(i + 1.0)
    return acc

def run_task_1_3(csv_filepath: str):
    print("=== Task 1.3: CPU Core Saturation Analysis ===")
    # 2,000,000 math operations per thread to balance runtime and CPU saturation
    iters_per_thread = 2_000_000
    thread_counts = [1, 2, 4, 8, 16, 32, 64]
    
    rows = []
    print(f"{'Threads (P)':<12}{'Workload/Thread':<18}{'Wall Time (s)':<16}")
    print("-" * 50)
    for p in thread_counts:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=p) as executor:
            futures = [executor.submit(heavy_cpu_worker, tid, iters_per_thread) for tid in range(p)]
            for f in futures:
                f.result()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"{p:<12}{iters_per_thread:<18}{elapsed:<16.4f}")
        rows.append({"threads": p, "iters_per_thread": iters_per_thread, "wall_time_s": f"{elapsed:.4f}"})

    with open(csv_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threads", "iters_per_thread", "wall_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Task 1.3 workload benchmark saved to {csv_filepath}\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_1_1 = os.path.join(base_dir, "task1_1_nondeterminism.txt")
    out_1_2 = os.path.join(base_dir, "task1_2_oversubscription.csv")
    out_1_3 = os.path.join(base_dir, "task1_3_saturation.csv")
    
    run_task_1_1(out_1_1)
    run_task_1_2(out_1_2)
    run_task_1_3(out_1_3)
