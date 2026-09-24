import os
import sys
import time
import csv
import threading
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

WIDTH, HEIGHT = 1920, 1080
MAX_ITER = 1000

@njit(nogil=True)
def render_row_range(img, start_row, end_row, width, height, max_iter):
    total_iters = 0
    for y in range(start_row, end_row):
        y0 = (y - height / 2.0) * 4.0 / height
        for x in range(width):
            x0 = (x - width / 2.0) * 4.0 / width
            cur_x = 0.0
            cur_y = 0.0
            it = 0
            while cur_x * cur_x + cur_y * cur_y <= 4.0 and it < max_iter:
                xtemp = cur_x * cur_x - cur_y * cur_y + x0
                cur_y = 2.0 * cur_x * cur_y + y0
                cur_x = xtemp
                it += 1
            img[y, x] = it
            total_iters += it
    return total_iters

# -------------------------------------------------------------
# Dynamic Work-Stealing Loop Scheduler
# -------------------------------------------------------------
class DynamicQueue:
    def __init__(self, total_rows, chunk_size):
        self.total_rows = total_rows
        self.chunk_size = chunk_size
        self.next_row = 0
        self.lock = threading.Lock()

    def get_chunk(self):
        with self.lock:
            if self.next_row >= self.total_rows:
                return None, None
            start = self.next_row
            end = min(self.next_row + self.chunk_size, self.total_rows)
            self.next_row = end
            return start, end

def run_dynamic_schedule(num_threads: int, chunk_size: int, record_work=False):
    img = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    queue = DynamicQueue(HEIGHT, chunk_size)
    thread_work = [0] * num_threads

    def worker(tid):
        local_work = 0
        while True:
            start, end = queue.get_chunk()
            if start is None:
                break
            w = render_row_range(img, start, end, WIDTH, HEIGHT, MAX_ITER)
            local_work += w
        thread_work[tid] = local_work

    threads = []
    t0 = time.perf_counter()
    for t in range(num_threads):
        th = threading.Thread(target=worker, args=(t,))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    if record_work:
        return elapsed, thread_work
    return elapsed

# -------------------------------------------------------------
# Static Round-Robin Chunk Scheduler
# -------------------------------------------------------------
def run_static_schedule(num_threads: int, chunk_size: int, record_work=False):
    img = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    thread_work = [0] * num_threads

    def worker(tid):
        local_work = 0
        k = 0
        while True:
            start = (k * num_threads + tid) * chunk_size
            if start >= HEIGHT:
                break
            end = min(start + chunk_size, HEIGHT)
            w = render_row_range(img, start, end, WIDTH, HEIGHT, MAX_ITER)
            local_work += w
            k += 1
        thread_work[tid] = local_work

    threads = []
    t0 = time.perf_counter()
    for t in range(num_threads):
        th = threading.Thread(target=worker, args=(t,))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    if record_work:
        return elapsed, thread_work
    return elapsed

# -------------------------------------------------------------
# Benchmarking & Visualization
# -------------------------------------------------------------
def benchmark_lab3(base_dir: str):
    print("==================================================")
    print(" LAB 3: WORK-SHARING & LOOP SCHEDULING (MANDELBROT)")
    print("==================================================")

    # Warm-up JIT
    dummy_img = np.zeros((10, 10), dtype=np.int32)
    render_row_range(dummy_img, 0, 10, 10, 10, 10)

    thread_counts = [2, 4, 8, 16]
    chunk_sizes = [1, 16, 64, 256]
    trials = 3

    print("\n--- Task 3.2: 4x4 Parameter Sweep Matrix (Dynamic Scheduling) ---")
    matrix_results = np.zeros((len(thread_counts), len(chunk_sizes)))
    csv_rows = []

    print(f"{'Threads (P)':<12}{'Chunk (C)':<12}{'Run 1 (s)':<12}{'Run 2 (s)':<12}{'Run 3 (s)':<12}{'Mean (s)':<12}")
    print("-" * 72)

    for i, p in enumerate(thread_counts):
        for j, c in enumerate(chunk_sizes):
            run_times = []
            for _ in range(trials):
                t_run = run_dynamic_schedule(p, c)
                run_times.append(t_run)
            mean_t = sum(run_times) / trials
            matrix_results[i, j] = mean_t
            print(f"{p:<12}{c:<12}{run_times[0]:<12.4f}{run_times[1]:<12.4f}{run_times[2]:<12.4f}{mean_t:<12.4f}")
            csv_rows.append({
                "threads": p,
                "chunk_size": c,
                "run1_s": f"{run_times[0]:.4f}",
                "run2_s": f"{run_times[1]:.4f}",
                "run3_s": f"{run_times[2]:.4f}",
                "mean_s": f"{mean_t:.4f}"
            })

    sweep_csv = os.path.join(base_dir, "task3_sweep.csv")
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threads", "chunk_size", "run1_s", "run2_s", "run3_s", "mean_s"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nParameter sweep data saved to {sweep_csv}")

    # Task 3.3: Data Visualization (2D Performance Heatmap)
    print("\n--- Task 3.3: Generating 2D Performance Heatmap ---")
    plt.figure(figsize=(8, 6), dpi=300)
    plt.imshow(matrix_results, cmap="viridis_r", aspect="auto")
    plt.colorbar(label="Execution Time (seconds)")
    plt.xticks(range(len(chunk_sizes)), chunk_sizes)
    plt.yticks(range(len(thread_counts)), thread_counts)
    plt.xlabel("Chunk Size ($C$)", fontsize=11, fontweight="bold")
    plt.ylabel("Thread Count ($P$)", fontsize=11, fontweight="bold")
    plt.title("Lab 3: Mandelbrot Dynamic Loop Scheduling Heatmap", fontsize=13, fontweight="bold", pad=12)

    # Annotate cell values
    for i in range(len(thread_counts)):
        for j in range(len(chunk_sizes)):
            val = matrix_results[i, j]
            text_color = "white" if val > np.mean(matrix_results) else "black"
            plt.text(j, i, f"{val:.3f}s", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

    plt.tight_layout()
    heatmap_path = os.path.join(base_dir, "heatmap_plot.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")

    # Task 3.4: Load Imbalance Quantification
    print("\n--- Task 3.4: Load Imbalance Quantification (P=8 Physical Cores) ---")
    P_TEST = 8
    # Static schedule with 1 contiguous chunk per thread (C = HEIGHT // P)
    static_chunk = HEIGHT // P_TEST
    t_static, work_static = run_static_schedule(P_TEST, static_chunk, record_work=True)
    
    # Dynamic schedule with fine chunk (C = 16)
    t_dyn, work_dyn = run_dynamic_schedule(P_TEST, 16, record_work=True)

    def calc_imbalance(work_list):
        max_w = max(work_list)
        min_w = min(work_list)
        avg_w = sum(work_list) / len(work_list)
        return (max_w - min_w) / avg_w, max_w, min_w, avg_w

    imb_stat, max_s, min_s, avg_s = calc_imbalance(work_static)
    imb_dyn, max_d, min_d, avg_d = calc_imbalance(work_dyn)

    print(f"Static (Block):  Time = {t_static:.4f}s | Imbalance Metric = {imb_stat:.4f} (Max={max_s:,}, Min={min_s:,})")
    print(f"Dynamic (C=16):  Time = {t_dyn:.4f}s | Imbalance Metric = {imb_dyn:.4f} (Max={max_d:,}, Min={min_d:,})")
    print(f"Load Imbalance Reduction: {(1.0 - imb_dyn/imb_stat)*100:.1f}%")

    imbalance_csv = os.path.join(base_dir, "task3_imbalance.csv")
    with open(imbalance_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Policy", "Threads", "Chunk_Size", "Time_s", "Imbalance_Ratio", "Max_Work", "Min_Work", "Avg_Work"])
        w.writerow(["Static_Block", P_TEST, static_chunk, f"{t_static:.4f}", f"{imb_stat:.4f}", max_s, min_s, f"{avg_s:.1f}"])
        w.writerow(["Dynamic_C16", P_TEST, 16, f"{t_dyn:.4f}", f"{imb_dyn:.4f}", max_d, min_d, f"{avg_d:.1f}"])
    print(f"Imbalance metrics saved to {imbalance_csv}\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_lab3(base_dir)
