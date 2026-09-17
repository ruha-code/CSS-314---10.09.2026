import multiprocessing as mp
import random
import time

TOTAL_POINTS = 5_000_000
NUM_THREADS = 4


def worker(n, hits, lock):
    rng = random.Random()
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            with lock:
                hits.value += 1


if __name__ == "__main__":
    print("=== PART 2: THE SYNCHRONIZATION TRAP ===")
    print(f"Points: {TOTAL_POINTS:,} | Threads: {NUM_THREADS}")
    print("-" * 50)

    rng = random.Random()
    t0 = time.perf_counter()
    single_hits = 0
    for _ in range(TOTAL_POINTS):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            single_hits += 1
    t_single = time.perf_counter() - t0
    pi_single = 4.0 * single_hits / TOTAL_POINTS
    print(f"Single Thread: Pi = {pi_single:.5f} | Time: {t_single*1000:.0f} ms")

    hits = mp.Value("l", 0)
    lock = mp.Lock()
    chunk = TOTAL_POINTS // NUM_THREADS
    procs = [mp.Process(target=worker, args=(chunk, hits, lock)) for _ in range(NUM_THREADS)]

    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    t_sync = time.perf_counter() - t0
    pi_sync = 4.0 * hits.value / TOTAL_POINTS
    print(f"Synchronized:  Pi = {pi_sync:.5f} | Time: {t_sync*1000:.0f} ms")
    print(f"Slowdown vs single thread: {t_sync / t_single:.2f}x")
