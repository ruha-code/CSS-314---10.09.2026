import threading
import time
import statistics

ITERATIONS = 5_000_000

def worker_adjacent(shared_list, index):
    for _ in range(ITERATIONS):
        shared_list[index] += 1

def worker_padded(shared_list, index):
    # Offset by 16 integers (16 * 8 bytes = 128 bytes > 64-byte cache line)
    padded_idx = index * 16
    for _ in range(ITERATIONS):
        shared_list[padded_idx] += 1

def run_test(target_fn, size):
    arr = [0] * size
    threads = [threading.Thread(target=target_fn, args=(arr, i)) for i in range(4)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start

if __name__ == "__main__":
    trials_adj = []
    trials_pad = []
    
    print("Running Task 2 Trials...")
    for i in range(3):
        t_adj = run_test(worker_adjacent, size=4)
        t_pad = run_test(worker_padded, size=64)
        trials_adj.append(t_adj)
        trials_pad.append(t_pad)
        print(f"Trial {i+1}: Adjacent = {t_adj:.4f}s | Padded = {t_pad:.4f}s")
    
    med_adj = statistics.median(trials_adj)
    med_pad = statistics.median(trials_pad)
    
    print("\nSummary:")
    print(f"Adjacent Indices (False Sharing) Median: {med_adj:.4f}s")
    print(f"Padded Indices (Cache-Aligned) Median: {med_pad:.4f}s")
    print(f"Slowdown Factor: {med_adj / med_pad:.2f}x")
