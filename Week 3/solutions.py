import multiprocessing as mp
import random
import time
import os

# CSS-314 Week 3
# ID: 230103026 | Group: 06-P


def worker_unsafe(n, hits):
    rng = random.Random()
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            hits.value += 1


def worker_locked(n, hits, lock):
    rng = random.Random()
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            with lock:
                hits.value += 1


def worker_reduction(n):
    rng = random.Random()
    local_hits = 0
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            local_hits += 1
    return local_hits


def part1():
    print(">>> RUNNING task1_phantom_bug.py ...")
    total = 50_000_000
    n_threads = 4
    chunk = total // n_threads

    for run in range(1, 6):
        hits = mp.Value("l", 0, lock=False)
        procs = [mp.Process(target=worker_unsafe, args=(chunk, hits)) for _ in range(n_threads)]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        dt = (time.perf_counter() - t0) * 1000
        pi = 4.0 * hits.value / total
        print(f"Run {run}: Pi = {pi:.5f} | Hits = {hits.value:,} | Time: {dt:.0f} ms")
    print()


def part2():
    print(">>> RUNNING task2_sync_trap.py ...")
    total = 5_000_000
    n_threads = 4

    rng = random.Random()
    t0 = time.perf_counter()
    single_hits = 0
    for _ in range(total):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            single_hits += 1
    t_single = time.perf_counter() - t0
    print(f"Single Thread: Pi = {4.0 * single_hits / total:.5f} | Time: {t_single*1000:.0f} ms")

    hits = mp.Value("l", 0)
    lock = mp.Lock()
    chunk = total // n_threads
    procs = [mp.Process(target=worker_locked, args=(chunk, hits, lock)) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    t_sync = time.perf_counter() - t0
    print(f"Synchronized:  Pi = {4.0 * hits.value / total:.5f} | Time: {t_sync*1000:.0f} ms")
    print(f"Slowdown: {t_sync / t_single:.2f}x")
    print()


def part3():
    print(">>> RUNNING task3_reduction.py ...")
    total = 100_000_000
    print(f"{'Threads':<10} {'Runtime (ms)':<16} {'Speedup':<12} {'Efficiency'}")
    print("-" * 50)

    t1 = None
    for t in [1, 2, 4, 8, 16, 32]:
        chunk = total // t
        leftover = total % t
        args = [chunk + leftover if i == 0 else chunk for i in range(t)]

        start = time.perf_counter()
        with mp.Pool(t) as pool:
            pool.map(worker_reduction, args)
        elapsed = (time.perf_counter() - start) * 1000

        if t == 1:
            t1 = elapsed
        speedup = t1 / elapsed
        print(f"{t:<10} {elapsed:<16.0f} {speedup:<12.2f}x {speedup / t * 100:.1f}%")
    print()


if __name__ == "__main__":
    print("=" * 55)
    print("   CSS-314 WEEK 3  |  ID: 230103026  |  Group: 06-P")
    print(f"   CPU cores: {os.cpu_count()}")
    print("=" * 55)
    print()
    part1()
    part2()
    part3()
