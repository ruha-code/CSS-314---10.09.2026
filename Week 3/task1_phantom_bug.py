import multiprocessing as mp
import random
import time

TOTAL_POINTS = 50_000_000
NUM_THREADS = 4


def worker(n, hits):
    rng = random.Random()
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            hits.value += 1


if __name__ == "__main__":
    print("=== PART 1: THE PHANTOM BUG ===")
    print(f"Points: {TOTAL_POINTS:,} | Threads: {NUM_THREADS}")
    print("-" * 50)

    chunk = TOTAL_POINTS // NUM_THREADS

    for run in range(1, 6):
        hits = mp.Value("l", 0, lock=False)
        procs = [mp.Process(target=worker, args=(chunk, hits)) for _ in range(NUM_THREADS)]

        t0 = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        t1 = time.perf_counter()

        pi = 4.0 * hits.value / TOTAL_POINTS
        print(f"Run {run}: Pi = {pi:.5f} | Hits = {hits.value:,} | Time: {(t1 - t0)*1000:.0f} ms")
