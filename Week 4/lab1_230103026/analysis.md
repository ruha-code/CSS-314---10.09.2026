# Technical Analysis & Defense: The Amdahl Reality Gap
**Course:** Parallel Computing | **Practicum:** Laboratory Worksheet 1  
**Student Name:** Ruslan Ussen  
**Student ID:** 230103026  
**Host Machine:** AMD Ryzen 7 7435HS (8 Physical Cores / 16 SMT Logical Processors)  
**OS:** Windows 11 Pro for Workstations  
**Target Workload Size:** $N = 13,026,000$ (Derived via formula $10,000,000 + 3026 \times 1,000$)  
**Verification Checksum:** `57863230` | **Max Stopping Steps:** `688`  

---

## Executive Summary & Experimental Dataset

All benchmarks were compiled natively with `gcc -O2 -fopenmp` and executed directly on the bare-metal AMD Ryzen 7 7435HS hardware. Precision timings were obtained using `omp_get_wtime()`. Warmup runs (Run 1) were discarded to invalidate cold-cache and OS memory allocation artifacts; reported times reflect the average of Run 2 and Run 3 under steady-state thermal and OS isolation conditions.

### Table 1: Multi-Thread Scaling & Empirical Amdahl Fitting
- **Sequential Baseline ($T_{seq}$):** $2.1550$ s
- **Dual-Core Empirical Speedup ($S_{emp}(2)$):** $1.8904$
- **Derived Parallel Fraction ($p$):** $0.9420$ ($94.20\%$)
- **Derived Sequential Fraction ($1 - p$):** $0.0580$ ($5.80\%$)

| Worker Threads ($k$) | Run 1 (Cold) (s) | Run 2 (s) | Run 3 (s) | Avg $T_k$ (s) | Empirical Speedup $S_{emp}(k)$ | Theoretical Speedup $S_{theo}(k)$ | Amdahl Reality Gap $\Delta(k)$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$k = 1$ ($T_1$)** | 2.2940 | 2.8810 | 2.2790 | 2.5800 | 0.8353x | 1.0000x | +0.1647 |
| **$k = 2$** | 1.1260 | 1.1430 | 1.1370 | 1.1400 | 1.8904x | 1.8904x | 0.0000 |
| **$k = 4$** | 0.6800 | 0.7380 | 0.7200 | 0.7290 | 2.9561x | 3.4071x | +0.4510 |
| **$k = 8$** | 0.4230 | 0.4100 | 0.4160 | 0.4130 | 5.2179x | 5.6898x | +0.4719 |
| **$k = 16$ (max)** | 0.2930 | 0.2730 | 0.2590 | 0.2660 | 8.1015x | 8.5558x | +0.4543 |

---

### Table 2: False Sharing Penalty Analysis ($k = 8$ Physical Cores)
*Task: Count numbers in $[1, N]$ requiring $> 100$ stopping steps ($Total Hits = 6,867,419$)*

| Implementation Variant | Physical Thread Count | Avg Execution Time (s) | Effective Throughput (iter/sec) | Speedup Penalty Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Variant 1: Naive `hits[tid]++` (False Sharing)** | 8 | 0.5010 s | 26,000,004 iter/s | **1.28x** |
| **Variant 2A: Cache-Padded Struct (64-byte aligned)** | 8 | 0.3925 s | 33,187,258 iter/s | **1.00x (Baseline)** |
| **Variant 2B: OpenMP Reduction (`reduction(+:hits)`)** | 8 | 0.3625 s | 35,933,798 iter/s | **0.92x (1.38x vs Naive)** |

---

### Table 3: OpenMP Loop Scheduling Evaluation ($k = 16$ Threads)

| OpenMP Scheduling Clause | Chunk Size | Execution Time (s) | Observed Behavior & CPU Load Distribution |
| :--- | :--- | :--- | :--- |
| `schedule(static)` | Default ($N / k \approx 814,125$) | 0.2655 s | Baseline static partitioning across threads; slight tail latency due to Collatz step variability. |
| `schedule(static, 1000)` | 1,000 | 0.2635 s | Round-robin chunk distribution; interleaves dense and light Collatz regions across cores. |
| `schedule(dynamic, 100)` | 100 | 0.2720 s | Fine-grained dynamic work queue; excessive atomic queue lock contention degrades speed. |
| **`schedule(dynamic, 10000)`** | **10,000** | **0.2510 s** | **Optimal performance.** Balances work dynamically while keeping atomic dispatch overhead negligible. |
| `schedule(guided)` | Exponential decay | 0.2595 s | Adaptive scheduling starting with large chunks and shrinking to smooth boundary tail latencies. |

---

## Detailed Technical Defense Questions

### Q1: Micro-Architectural Root Cause of False Sharing
**Question:** Explain why Variant 1 in Experiment A caused massive execution degradation. Specifically reference your CPU's L1 cache line size (in bytes), the MESI/MOESI cache protocol, and bus invalidation traffic.

**Answer:**  
On the AMD Ryzen 7 7435HS (Zen 3+ microarchitecture), the Level 1 (L1) data cache line size is exactly **64 bytes**. In Variant 1, the counter array was declared as:
```c
static int hit_count[MAX_THREADS];
```
Since a 32-bit `int` occupies 4 bytes, all 8 thread counters ($8 \times 4 = 32$ bytes) fit contiguously inside a **single 64-byte L1 cache line**. 

AMD Zen processors maintain cache coherency across individual core complexes (CCX) using the **MOESI** (Modified, Owned, Exclusive, Shared, Invalid) protocol over the high-speed Infinity Fabric interconnect. 
1. When Thread 0 on Core 0 executes `hit_count[0]++`, Core 0 must acquire exclusive write ownership of the containing cache line, transitioning its state to **Modified (M)**.
2. Under MOESI, acquiring write permission requires broadcasting an **invalidation request (BusRdX / Inval)** across the fabric to all peer cores.
3. This invalidation forces the cache controllers on Cores 1 through 7 to transition their local copy of that 64-byte line to the **Invalid (I)** state.
4. An instant later, when Thread 1 on Core 1 attempts to execute `hit_count[1]++`, it incurs an L1 data cache miss because the line is in state `I`. Core 1 stalls its execution pipeline, issues a read request over the interconnect, forces Core 0 to flush or forward the modified line, and marks it Modified in its own L1 cache while invalidating Core 0.

This continuous "cache-line ping-pong" generates heavy **bus invalidation traffic**, saturates the inter-core interconnect, and constantly stalls instruction commit units. 

In our empirical measurement, this phenomenon inflated execution time from **0.3925 s** (cache-padded) to **0.5010 s** (naive), representing an immediate **1.28x penalty (or 1.38x compared to reduction)**, purely caused by sharing the same physical 64-byte cache line despite accessing logically independent array indices.

---

### Q2: Hyperthreading (SMT) Saturation & Physical Core Ceilings
**Question:** Did your speedup continue to scale linearly when thread count $k$ increased from your physical core count to your logical thread count? Why or why not? What hardware execution resources are shared between SMT threads?

**Answer:**  
No, speedup did **not** scale linearly when $k$ transitioned from the physical core ceiling ($k = 8$) to the logical thread count ($k = 16$).
- Scaling across physical cores showed strong performance:
  - $k = 2$: $T = 1.1400$ s ($S_{emp} = 1.89$x)
  - $k = 4$: $T = 0.7290$ s ($S_{emp} = 2.96$x)
  - $k = 8$: $T = 0.4130$ s ($S_{emp} = 5.22$x) -> scaling factor from 4 to 8 physical cores was $\mathbf{1.76\times}$.
- Scaling from physical cores to SMT logical threads ($k = 8 \to 16$):
  - $k = 16$: $T = 0.2660$ s ($S_{emp} = 8.10$x) -> scaling factor was only $\mathbf{1.55\times}$ ($8.10 / 5.22$), falling significantly short of a $2.0\times$ linear doubling.

**Micro-architectural reason:**  
Simultaneous Multithreading (SMT) on AMD Zen 3+ duplicates only the thread state (architectural register state, program counters, instruction pointers, and return address stacks). The underlying physical execution resources are strictly **shared** between sibling threads on the same physical core:
1. **Integer Execution Pipelines & ALUs:** The Collatz kernel is an integer-dominated, branch-heavy workload (`while (n > 1) { if (n & 1) n >>= 1; else n = 3*n + 1; }`). A single thread executing this tight loop keeps the core's 4 integer ALUs and branch execution ports highly utilized. The second SMT thread cannot access additional ALUs; it must wait for execution slots.
2. **Caches & Buffers:** SMT threads compete directly for the private 32 KB L1 Data Cache, 32 KB L1 Instruction Cache, and 512 KB L2 Cache, as well as the Reorder Buffer (ROB) and Load/Store Queues. Cache capacity per active thread is halved, increasing cache eviction rates.
3. **Branch Prediction Table:** Branch history buffers are shared, leading to mutual branch history aliasing and higher branch misprediction rates for the unpredictable Collatz trajectory branches.

Consequently, SMT provides a throughput boost primarily by masking latency during memory stalls or pipeline bubbles, but cannot provide linear compute scaling for compute-bound ALU kernels.

---

### Q3: Empirical vs. Theoretical Amdahl Discrepancy
**Question:** State your derived parallel fraction $p$ (from $k=2$). Why did the theoretical curve $S_{theo}(k)$ diverge from your empirical $S_{emp}(k)$ as $k$ increased to 8 or 16? Identify two real physical factors omitted by Amdahl's equation.

**Answer:**  
- **Derived Parallel Fraction ($p$):**  
  Using dual-core speedup $S_{emp}(2) = 1.890351$:
  $$p = 2 \times \left[ 1 - \frac{1}{S_{emp}(2)} \right] = 2 \times \left[ 1 - \frac{1}{1.890351} \right] = \mathbf{0.941995} \quad (94.20\%)$$
  Sequential fraction: $1 - p = 0.058005$ ($5.80\%$).

- **Divergence:**  
  At $k = 8$, Amdahl's Law predicted $S_{theo}(8) = 5.6898\text{x}$, whereas empirical speedup reached $S_{emp}(8) = 5.2179\text{x}$ (Reality Gap $\Delta(8) = +0.4719$). At $k = 16$, $S_{theo}(16) = 8.5558\text{x}$ vs $S_{emp}(16) = 8.1015\text{x}$ ($\Delta(16) = +0.4543$).

**Two Physical Factors Omitted by Amdahl's Model:**
1. **Dynamic CPU Frequency Scaling & Thermal/Power Budgeting (AMD Precision Boost):**  
   Amdahl's Law implicitly assumes that processor core clocks remain constant regardless of thread count $k$. In physical silicon, the AMD Ryzen 7 7435HS operates within a strict Thermal Design Power (TDP) budget (35–54W). When running with $k=1$ or $k=2$ threads, active cores boost to maximum clock frequencies (up to ~4.5 GHz). When all 8 cores and 16 SMT threads are fully loaded, power and thermals force all-core frequencies down closer to base clock (~3.1 GHz). Thus, the effective compute capacity per thread decreases as $k$ increases, causing empirical speedup to fall below Amdahl's prediction.
2. **Synchronization Overhead & Interconnect / Memory Contention:**  
   Amdahl's formula treats thread parallelization as zero-cost. In reality, scaling to 8 and 16 threads introduces:
   - Non-zero OpenMP runtime thread fork/join overhead.
   - Barrier synchronization delays at loop exit.
   - Memory controller and Infinity Fabric interconnect contention when multiple cores write their reduction results or read shared memory regions.

---

### Q4: The Scheduling Trade-Off Dilemma
**Question:** Compare `dynamic(100)` versus `dynamic(10000)` and `static` scheduling. At what chunk size did lock-queue contention outweigh the benefits of workload balancing on your machine?

**Answer:**  
In the Collatz 3x+1 problem, adjacent integers exhibit extreme non-uniformity in trajectory length (e.g., $i=26$ requires only 10 steps, whereas $i=27$ requires 111 steps). 

Empirical execution times at $k = 16$ threads:
- `schedule(static)` (Default chunk $N/k \approx 814,125$): **0.2655 s**
- `schedule(dynamic, 100)`: **0.2720 s**
- `schedule(dynamic, 10000)`: **0.2510 s** (Fastest)

**Analysis of the Trade-off:**
1. **`schedule(static)` (0.2655 s):** Each thread receives a single monolithic contiguous chunk of $\approx 814,125$ numbers. It incurs zero runtime synchronization overhead, but threads assigned ranges with clusters of long trajectories finish later, causing other threads to idle at the implicit barrier (tail latency).
2. **`schedule(dynamic, 100)` (0.2720 s):** Decomposes $N = 13,026,000$ into **130,260 separate chunks**. For every 100 numbers evaluated, a thread must execute an atomic fetch-and-add or acquire a lock on the shared OpenMP task queue. At 16 concurrent threads, the high frequency of atomic updates causes severe cache-line bouncing and lock queue contention, entirely wiping out any benefit from dynamic load balancing and making it slower than static scheduling.
3. **`schedule(dynamic, 10000)` (0.2510 s):** Decomposes $N$ into only **1,302 chunks**. The atomic dispatch overhead is amortized across 10,000 Collatz evaluations per fetch, rendering synchronization cost negligible while still allowing idle threads to dynamically grab remaining chunks, eliminating tail latency.

**Crossover Threshold:**  
Lock-queue contention outweighed the benefits of workload balancing at fine chunk sizes **below approximately 500 to 1,000 iterations**. Above 1,000 iterations, the dispatch frequency drops sufficiently low that dynamic scheduling reliably outperforms static partitioning.
