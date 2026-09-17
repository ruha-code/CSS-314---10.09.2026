import multiprocessing as mp
import random
import time

TOTAL_POINTS = 100_000_000
THREAD_COUNTS = [1, 2, 4, 8, 16, 32]


def worker(n):
    rng = random.Random()
    local_hits = 0
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            local_hits += 1
    return local_hits


if __name__ == "__main__":
    print("=== PART 3: OPENMP-STYLE REDUCTION ===")
    print(f"Points: {TOTAL_POINTS:,}")
    print("-" * 60)
    print(f"{'Threads':<10} {'Runtime (ms)':<16} {'Speedup':<12} {'Efficiency'}")
    print("-" * 60)

    t1 = None
    for t in THREAD_COUNTS:
        chunk = TOTAL_POINTS // t
        leftover = TOTAL_POINTS % t
        args = [chunk + leftover if i == 0 else chunk for i in range(t)]

        start = time.perf_counter()
        with mp.Pool(t) as pool:
            hits = sum(pool.map(worker, args))
        elapsed = (time.perf_counter() - start) * 1000

        if t == 1:
            t1 = elapsed

        speedup = t1 / elapsed
        efficiency = speedup / t * 100
        print(f"{t:<10} {elapsed:<16.0f} {speedup:<12.2f}x {efficiency:.1f}%")
