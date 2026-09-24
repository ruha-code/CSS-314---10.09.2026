/*
 * ============================================================================
 * PRACTICUM WORKSHEET: THE AMDAHL REALITY GAP
 * Empirical Speedup Fitting, Micro-Architectural Bottlenecks, and Loop Scheduling in OpenMP
 * 
 * Student Full Name: Ruslan Ussen
 * Student ID Number: 230103026
 * Workload Formula: N = 10,000,000 + ([Last 4 Digits of Student ID] * 1,000)
 * Last 4 Digits: 3026 -> Workload Size N = 13,026,000
 * CPU Model: AMD Ryzen 7 7435HS (8 Physical Cores / 16 SMT Logical Processors)
 * L1/L2 Cache Line Size: 64 bytes
 * 
 * Compilation:
 *   gcc -O2 -fopenmp collatz.c -o collatz
 * Execution:
 *   ./collatz
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define STUDENT_ID 230103026ULL
#define WORKLOAD_N (10000000ULL + (3026ULL * 1000ULL)) // 13,026,000
#define MOD_VAL 1000000007ULL
#define MAX_THREADS 128
#define PHYSICAL_CORES 8
#define LOGICAL_THREADS 16

/* Collatz Stopping Time Kernel as specified in Section 2, Step 1.2 */
static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) {
            n >>= 1;
        } else {
            n = 3 * n + 1;
        }
        steps++;
    }
    return steps;
}

/* ----------------------------------------------------------------------------
 * Sequential Execution Kernel (Phase 2)
 * ---------------------------------------------------------------------------- */
void run_sequential(uint64_t n, uint32_t *out_max, uint64_t *out_sum, double *out_time) {
    uint32_t max_s = 0;
    uint64_t sum = 0;
    double t0 = omp_get_wtime();

    for (uint64_t i = 1; i <= n; i++) {
        uint32_t s = collatz_steps(i);
        if (s > max_s) max_s = s;
        sum += s;
        if (sum >= MOD_VAL) sum %= MOD_VAL;
    }

    double t1 = omp_get_wtime();
    if (out_max) *out_max = max_s;
    if (out_sum) *out_sum = sum;
    if (out_time) *out_time = (t1 - t0);
}

/* ----------------------------------------------------------------------------
 * Multi-Threaded OpenMP Parallel Execution (Phase 3)
 * ---------------------------------------------------------------------------- */
void run_parallel(uint64_t n, int k, uint32_t *out_max, uint64_t *out_sum, double *out_time) {
    uint32_t global_max = 0;
    uint64_t global_sum = 0;
    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(k)
    {
        uint32_t local_max = 0;
        uint64_t local_sum = 0;

        #pragma omp for schedule(static)
        for (uint64_t i = 1; i <= n; i++) {
            uint32_t s = collatz_steps(i);
            if (s > local_max) local_max = s;
            local_sum += s;
            if (local_sum >= MOD_VAL) local_sum %= MOD_VAL;
        }

        #pragma omp critical
        {
            if (local_max > global_max) global_max = local_max;
            global_sum = (global_sum + local_sum) % MOD_VAL;
        }
    }

    double t1 = omp_get_wtime();
    if (out_max) *out_max = global_max;
    if (out_sum) *out_sum = global_sum;
    if (out_time) *out_time = (t1 - t0);
}

/* ----------------------------------------------------------------------------
 * Phase 4: Experiment A - False Sharing Benchmark
 * Count numbers in [1, N] where stopping steps > 100
 * ---------------------------------------------------------------------------- */

// Variant 1: Naive array (int is 4 bytes -> all threads pack into the same 64-byte cache line)
static int hit_count_naive[MAX_THREADS];

void run_fs_naive(uint64_t n, int k, uint64_t *out_hits, double *out_time) {
    for (int i = 0; i < k; i++) hit_count_naive[i] = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(k)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (uint64_t i = 1; i <= n; i++) {
            uint32_t s = collatz_steps(i);
            if (s > 100) {
                hit_count_naive[tid]++;
            }
        }
    }

    double t1 = omp_get_wtime();

    uint64_t total = 0;
    for (int i = 0; i < k; i++) total += hit_count_naive[i];

    if (out_hits) *out_hits = total;
    if (out_time) *out_time = (t1 - t0);
}

// Variant 2A: Cache-Padded Struct (Forces each thread onto its own 64-byte cache line)
struct __attribute__((aligned(64))) PaddedCounter {
    int count;
    char pad[60]; // 4 + 60 = 64 bytes
};
static struct PaddedCounter hit_count_padded[MAX_THREADS];

void run_fs_padded(uint64_t n, int k, uint64_t *out_hits, double *out_time) {
    for (int i = 0; i < k; i++) hit_count_padded[i].count = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(k)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (uint64_t i = 1; i <= n; i++) {
            uint32_t s = collatz_steps(i);
            if (s > 100) {
                hit_count_padded[tid].count++;
            }
        }
    }

    double t1 = omp_get_wtime();

    uint64_t total = 0;
    for (int i = 0; i < k; i++) total += hit_count_padded[i].count;

    if (out_hits) *out_hits = total;
    if (out_time) *out_time = (t1 - t0);
}

// Variant 2B: OpenMP Reduction Variable
void run_fs_reduction(uint64_t n, int k, uint64_t *out_hits, double *out_time) {
    uint64_t total_hits = 0;
    double t0 = omp_get_wtime();

    #pragma omp parallel for num_threads(k) schedule(static) reduction(+:total_hits)
    for (uint64_t i = 1; i <= n; i++) {
        uint32_t s = collatz_steps(i);
        if (s > 100) {
            total_hits++;
        }
    }

    double t1 = omp_get_wtime();
    if (out_hits) *out_hits = total_hits;
    if (out_time) *out_time = (t1 - t0);
}

/* ----------------------------------------------------------------------------
 * Phase 4: Experiment B - OpenMP Loop Scheduling Evaluation
 * ---------------------------------------------------------------------------- */
void run_scheduling_benchmark(uint64_t n, int k, int mode, uint32_t *out_max, uint64_t *out_sum, double *out_time) {
    uint32_t global_max = 0;
    uint64_t global_sum = 0;
    double t0 = omp_get_wtime();

    if (mode == 0) {
        // schedule(static) - Default (N / k)
        #pragma omp parallel num_threads(k)
        {
            uint32_t l_max = 0;
            uint64_t l_sum = 0;
            #pragma omp for schedule(static)
            for (uint64_t i = 1; i <= n; i++) {
                uint32_t s = collatz_steps(i);
                if (s > l_max) l_max = s;
                l_sum += s;
                if (l_sum >= MOD_VAL) l_sum %= MOD_VAL;
            }
            #pragma omp critical
            {
                if (l_max > global_max) global_max = l_max;
                global_sum = (global_sum + l_sum) % MOD_VAL;
            }
        }
    } else if (mode == 1) {
        // schedule(static, 1000)
        #pragma omp parallel num_threads(k)
        {
            uint32_t l_max = 0;
            uint64_t l_sum = 0;
            #pragma omp for schedule(static, 1000)
            for (uint64_t i = 1; i <= n; i++) {
                uint32_t s = collatz_steps(i);
                if (s > l_max) l_max = s;
                l_sum += s;
                if (l_sum >= MOD_VAL) l_sum %= MOD_VAL;
            }
            #pragma omp critical
            {
                if (l_max > global_max) global_max = l_max;
                global_sum = (global_sum + l_sum) % MOD_VAL;
            }
        }
    } else if (mode == 2) {
        // schedule(dynamic, 100)
        #pragma omp parallel num_threads(k)
        {
            uint32_t l_max = 0;
            uint64_t l_sum = 0;
            #pragma omp for schedule(dynamic, 100)
            for (uint64_t i = 1; i <= n; i++) {
                uint32_t s = collatz_steps(i);
                if (s > l_max) l_max = s;
                l_sum += s;
                if (l_sum >= MOD_VAL) l_sum %= MOD_VAL;
            }
            #pragma omp critical
            {
                if (l_max > global_max) global_max = l_max;
                global_sum = (global_sum + l_sum) % MOD_VAL;
            }
        }
    } else if (mode == 3) {
        // schedule(dynamic, 10000)
        #pragma omp parallel num_threads(k)
        {
            uint32_t l_max = 0;
            uint64_t l_sum = 0;
            #pragma omp for schedule(dynamic, 10000)
            for (uint64_t i = 1; i <= n; i++) {
                uint32_t s = collatz_steps(i);
                if (s > l_max) l_max = s;
                l_sum += s;
                if (l_sum >= MOD_VAL) l_sum %= MOD_VAL;
            }
            #pragma omp critical
            {
                if (l_max > global_max) global_max = l_max;
                global_sum = (global_sum + l_sum) % MOD_VAL;
            }
        }
    } else if (mode == 4) {
        // schedule(guided)
        #pragma omp parallel num_threads(k)
        {
            uint32_t l_max = 0;
            uint64_t l_sum = 0;
            #pragma omp for schedule(guided)
            for (uint64_t i = 1; i <= n; i++) {
                uint32_t s = collatz_steps(i);
                if (s > l_max) l_max = s;
                l_sum += s;
                if (l_sum >= MOD_VAL) l_sum %= MOD_VAL;
            }
            #pragma omp critical
            {
                if (l_max > global_max) global_max = l_max;
                global_sum = (global_sum + l_sum) % MOD_VAL;
            }
        }
    }

    double t1 = omp_get_wtime();
    if (out_max) *out_max = global_max;
    if (out_sum) *out_sum = global_sum;
    if (out_time) *out_time = (t1 - t0);
}

/* ============================================================================
 * Main Benchmark Driver & Report Generator
 * ============================================================================ */
int main(void) {
    uint64_t n = WORKLOAD_N;
    FILE *f_csv = fopen("results.csv", "w");
    if (!f_csv) {
        perror("Failed to open results.csv");
        return 1;
    }

    printf("================================================================================\n");
    printf("PRACTICUM BENCHMARK: THE AMDAHL REALITY GAP\n");
    printf("Student Full Name: Ruslan Ussen | Student ID: %llu\n", (unsigned long long)STUDENT_ID);
    printf("Target Workload Size N : %llu\n", (unsigned long long)n);
    printf("Physical Cores         : %d | Logical SMT Threads: %d\n", PHYSICAL_CORES, LOGICAL_THREADS);
    printf("OpenMP Max Threads     : %d\n", omp_get_max_threads());
    printf("================================================================================\n\n");

    /* ------------------------------------------------------------------------
     * Phase 2: Sequential Baseline & Precision Timing
     * ------------------------------------------------------------------------ */
    printf("--------------------------------------------------------------------------------\n");
    printf("PHASE 2: Sequential Baseline Benchmarking (Warmup Invalidation Protocol)\n");
    printf("--------------------------------------------------------------------------------\n");
    uint32_t max_steps = 0;
    uint64_t checksum = 0;
    double t_seq_warmup = 0.0, t_seq_r2 = 0.0, t_seq_r3 = 0.0;

    printf("Executing Run 1 (Cold cache / Warmup)...\n");
    run_sequential(n, &max_steps, &checksum, &t_seq_warmup);
    printf("  Run 1 Time          : %.6f s [WARMUP - DISCARDED]\n", t_seq_warmup);
    printf("  Verification Checksum: %llu | Max Sequence Steps: %u\n", (unsigned long long)checksum, max_steps);

    printf("Executing Run 2 (Timed benchmark)...\n");
    run_sequential(n, &max_steps, &checksum, &t_seq_r2);
    printf("  Run 2 Time          : %.6f s\n", t_seq_r2);

    printf("Executing Run 3 (Timed benchmark)...\n");
    run_sequential(n, &max_steps, &checksum, &t_seq_r3);
    printf("  Run 3 Time          : %.6f s\n", t_seq_r3);

    double t_seq = (t_seq_r2 + t_seq_r3) / 2.0;
    printf("--> Sequential Baseline T_seq: %.6f s\n\n", t_seq);

    /* ------------------------------------------------------------------------
     * Phase 3: Multi-Thread Scaling & Empirical Amdahl Fitting
     * ------------------------------------------------------------------------ */
    printf("--------------------------------------------------------------------------------\n");
    printf("PHASE 3: Multi-Thread Scaling & Empirical Amdahl Fitting\n");
    printf("Evaluating thread counts k in {1, 2, 4, 8, 16}\n");
    printf("--------------------------------------------------------------------------------\n");

    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    double r1_times[5], r2_times[5], r3_times[5], avg_tk[5], s_emp[5], s_theo[5], delta_gap[5];

    for (int idx = 0; idx < num_tests; idx++) {
        int k = thread_counts[idx];
        uint32_t m_val = 0;
        uint64_t c_val = 0;

        printf("Benchmarking k = %2d threads...\n", k);
        // Run 1: Warmup
        run_parallel(n, k, &m_val, &c_val, &r1_times[idx]);
        // Run 2: Timed
        run_parallel(n, k, &m_val, &c_val, &r2_times[idx]);
        // Run 3: Timed
        run_parallel(n, k, &m_val, &c_val, &r3_times[idx]);

        avg_tk[idx] = (r2_times[idx] + r3_times[idx]) / 2.0;
        s_emp[idx] = t_seq / avg_tk[idx];

        printf("  k = %2d | Run 1 (Cold): %.4fs | Run 2: %.4fs | Run 3: %.4fs | Avg T_k: %.4fs | S_emp: %.4fx\n",
               k, r1_times[idx], r2_times[idx], r3_times[idx], avg_tk[idx], s_emp[idx]);
    }

    // Analytical derivation of parallel fraction p from k = 2 (index 1)
    double s_emp_2 = s_emp[1];
    // Formula: p = 2 * [1 - (1 / S_emp(2))]
    double p = 2.0 * (1.0 - (1.0 / s_emp_2));
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;

    printf("\nAnalytical Derivation of Parallel Fraction (p):\n");
    printf("  Dual-Core Speedup S_emp(2) = %.6f\n", s_emp_2);
    printf("  Formula: p = 2 * [ 1 - (1 / S_emp(2)) ]\n");
    printf("  Derived Parallel Fraction (p)       : %.6f (%.2f%%)\n", p, p * 100.0);
    printf("  Sequential Component Fraction (1-p) : %.6f (%.2f%%)\n\n", 1.0 - p, (1.0 - p) * 100.0);

    // Compute S_theo(k) and Delta(k)
    for (int idx = 0; idx < num_tests; idx++) {
        int k = thread_counts[idx];
        s_theo[idx] = 1.0 / ((1.0 - p) + (p / (double)k));
        delta_gap[idx] = s_theo[idx] - s_emp[idx];
    }

    printf("====================================================================================================\n");
    printf("TABLE 1: Multi-Thread Scaling & Empirical Amdahl Fitting\n");
    printf("%-8s %-14s %-12s %-12s %-14s %-12s %-12s %-10s\n",
           "Threads", "Run 1 (Cold)", "Run 2 (s)", "Run 3 (s)", "Avg T_k (s)", "S_emp(k)", "S_theo(k)", "Delta (k)");
    printf("----------------------------------------------------------------------------------------------------\n");
    for (int idx = 0; idx < num_tests; idx++) {
        char k_label[16];
        if (thread_counts[idx] == 1) snprintf(k_label, sizeof(k_label), "k=1(T_1)");
        else if (thread_counts[idx] == 16) snprintf(k_label, sizeof(k_label), "k=16(max)");
        else snprintf(k_label, sizeof(k_label), "k=%d", thread_counts[idx]);

        printf("%-8s %-14.4f %-12.4f %-12.4f %-14.4f %-12.4f %-12.4f %-10.4f\n",
               k_label, r1_times[idx], r2_times[idx], r3_times[idx], avg_tk[idx], s_emp[idx], s_theo[idx], delta_gap[idx]);
    }
    printf("====================================================================================================\n\n");

    /* ------------------------------------------------------------------------
     * Phase 4: Experiment A - False Sharing Penalty
     * ------------------------------------------------------------------------ */
    printf("--------------------------------------------------------------------------------\n");
    printf("PHASE 4, EXPERIMENT A: False Sharing Penalty (Tested at Max Physical Cores: k = %d)\n", PHYSICAL_CORES);
    printf("--------------------------------------------------------------------------------\n");

    double fs_naive_w, fs_naive_r2, fs_naive_r3, fs_naive_avg;
    double fs_pad_w, fs_pad_r2, fs_pad_r3, fs_pad_avg;
    double fs_red_w, fs_red_r2, fs_red_r3, fs_red_avg;
    uint64_t hits_naive = 0, hits_pad = 0, hits_red = 0;

    printf("Executing Variant 1: Naive hits[tid]++ (False Sharing)...\n");
    run_fs_naive(n, PHYSICAL_CORES, &hits_naive, &fs_naive_w);
    run_fs_naive(n, PHYSICAL_CORES, &hits_naive, &fs_naive_r2);
    run_fs_naive(n, PHYSICAL_CORES, &hits_naive, &fs_naive_r3);
    fs_naive_avg = (fs_naive_r2 + fs_naive_r3) / 2.0;

    printf("Executing Variant 2A: Cache-Padded Struct (64-byte aligned)...\n");
    run_fs_padded(n, PHYSICAL_CORES, &hits_pad, &fs_pad_w);
    run_fs_padded(n, PHYSICAL_CORES, &hits_pad, &fs_pad_r2);
    run_fs_padded(n, PHYSICAL_CORES, &hits_pad, &fs_pad_r3);
    fs_pad_avg = (fs_pad_r2 + fs_pad_r3) / 2.0;

    printf("Executing Variant 2B: OpenMP Reduction Variable...\n");
    run_fs_reduction(n, PHYSICAL_CORES, &hits_red, &fs_red_w);
    run_fs_reduction(n, PHYSICAL_CORES, &hits_red, &fs_red_r2);
    run_fs_reduction(n, PHYSICAL_CORES, &hits_red, &fs_red_r3);
    fs_red_avg = (fs_red_r2 + fs_red_r3) / 2.0;

    double tp_naive = (double)n / fs_naive_avg;
    double tp_pad   = (double)n / fs_pad_avg;
    double tp_red   = (double)n / fs_red_avg;
    double penalty_pad = fs_naive_avg / fs_pad_avg;
    double penalty_red = fs_naive_avg / fs_red_avg;

    printf("\n====================================================================================================\n");
    printf("TABLE 2: False Sharing Penalty Analysis (k = %d Physical Cores)\n", PHYSICAL_CORES);
    printf("%-38s %-14s %-18s %-24s %-14s\n",
           "Implementation Variant", "Thread Count", "Execution Time (s)", "Throughput (iter/sec)", "Penalty Ratio");
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("%-38s %-14d %-18.4f %-24.0f %-14.2fx\n",
           "Variant 1: Naive hits[tid]++", PHYSICAL_CORES, fs_naive_avg, tp_naive, penalty_pad);
    printf("%-38s %-14d %-18.4f %-24.0f %-14s\n",
           "Variant 2A: Cache-Padded Struct", PHYSICAL_CORES, fs_pad_avg, tp_pad, "1.00x (Baseline)");
    printf("%-38s %-14d %-18.4f %-24.0f %-14.2fx\n",
           "Variant 2B: OpenMP Reduction", PHYSICAL_CORES, fs_red_avg, tp_red, penalty_red / penalty_pad);
    printf("====================================================================================================\n\n");

    /* ------------------------------------------------------------------------
     * Phase 4: Experiment B - OpenMP Scheduling Evaluation
     * ------------------------------------------------------------------------ */
    printf("--------------------------------------------------------------------------------\n");
    printf("PHASE 4, EXPERIMENT B: Loop Scheduling & Workload Imbalance (k = %d Threads)\n", LOGICAL_THREADS);
    printf("--------------------------------------------------------------------------------\n");

    const char *sched_names[] = {
        "schedule(static)",
        "schedule(static, 1000)",
        "schedule(dynamic, 100)",
        "schedule(dynamic, 10000)",
        "schedule(guided)"
    };
    const char *sched_chunks[] = {
        "Default (N / k)",
        "1,000",
        "100",
        "10,000",
        "Exponential decay"
    };
    const char *sched_behaviors[] = {
        "Baseline static partitioning across threads.",
        "Round-robin chunk distribution.",
        "Fine-grained dynamic work queue (High lock contention).",
        "Coarse-grained dynamic work queue (Balanced overhead).",
        "Adaptive scheduling (starts large, reduces size)."
    };

    double sched_w[5], sched_r2[5], sched_r3[5], sched_avg[5];

    for (int m = 0; m < 5; m++) {
        printf("Evaluating %s...\n", sched_names[m]);
        uint32_t sm = 0; uint64_t sc = 0;
        run_scheduling_benchmark(n, LOGICAL_THREADS, m, &sm, &sc, &sched_w[m]);
        run_scheduling_benchmark(n, LOGICAL_THREADS, m, &sm, &sc, &sched_r2[m]);
        run_scheduling_benchmark(n, LOGICAL_THREADS, m, &sm, &sc, &sched_r3[m]);
        sched_avg[m] = (sched_r2[m] + sched_r3[m]) / 2.0;
        printf("  -> Execution Time: %.4fs (Cold Warmup: %.4fs)\n", sched_avg[m], sched_w[m]);
    }

    printf("\n====================================================================================================\n");
    printf("TABLE 3: OpenMP Scheduling Clauses & Workload Imbalance (k = %d Threads)\n", LOGICAL_THREADS);
    printf("%-26s %-20s %-20s %-32s\n",
           "OpenMP Scheduling Clause", "Chunk Size", "Execution Time (s)", "Observed Behavior & CPU Load Distribution");
    printf("----------------------------------------------------------------------------------------------------\n");
    for (int m = 0; m < 5; m++) {
        printf("%-26s %-20s %-20.4f %-32s\n",
               sched_names[m], sched_chunks[m], sched_avg[m], sched_behaviors[m]);
    }
    printf("====================================================================================================\n\n");

    /* ------------------------------------------------------------------------
     * Write Benchmark Dataset to results.csv
     * ------------------------------------------------------------------------ */
    fprintf(f_csv, "# PRACTICUM BENCHMARK DATASET: THE AMDAHL REALITY GAP\n");
    fprintf(f_csv, "# Student ID: %llu\n", (unsigned long long)STUDENT_ID);
    fprintf(f_csv, "# Workload N: %llu\n", (unsigned long long)n);
    fprintf(f_csv, "# CPU: AMD Ryzen 7 7435HS (8 Cores / 16 SMT Threads)\n");
    fprintf(f_csv, "# Sequential Baseline: T_seq = %.6f s\n", t_seq);
    fprintf(f_csv, "# Verification Checksum: %llu | Max Steps: %u\n", (unsigned long long)checksum, max_steps);
    fprintf(f_csv, "# Derived Parallel Fraction p = %.6f\n\n", p);

    fprintf(f_csv, "# Table 1: Multi-Thread Scaling & Empirical Amdahl Fitting\n");
    fprintf(f_csv, "Table,Threads_k,Run1_Cold_s,Run2_s,Run3_s,Avg_Tk_s,S_emp,S_theo,Delta_k\n");
    for (int idx = 0; idx < num_tests; idx++) {
        fprintf(f_csv, "Table1,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                thread_counts[idx], r1_times[idx], r2_times[idx], r3_times[idx],
                avg_tk[idx], s_emp[idx], s_theo[idx], delta_gap[idx]);
    }
    fprintf(f_csv, "\n");

    fprintf(f_csv, "# Table 2: False Sharing Penalty Analysis (k = %d Physical Cores)\n", PHYSICAL_CORES);
    fprintf(f_csv, "Table,Variant,Thread_Count,Run1_Cold_s,Run2_s,Run3_s,Execution_Time_s,Throughput_iter_sec,Speedup_Penalty_Ratio\n");
    fprintf(f_csv, "Table2,\"Variant 1: Naive hits[tid]++ (False Sharing)\",%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.4f\n",
            PHYSICAL_CORES, fs_naive_w, fs_naive_r2, fs_naive_r3, fs_naive_avg, tp_naive, penalty_pad);
    fprintf(f_csv, "Table2,\"Variant 2A: Cache-Padded Struct\",%d,%.6f,%.6f,%.6f,%.6f,%.2f,1.0000\n",
            PHYSICAL_CORES, fs_pad_w, fs_pad_r2, fs_pad_r3, fs_pad_avg, tp_pad);
    fprintf(f_csv, "Table2,\"Variant 2B: OpenMP Reduction\",%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.4f\n",
            PHYSICAL_CORES, fs_red_w, fs_red_r2, fs_red_r3, fs_red_avg, tp_red, fs_naive_avg / fs_red_avg);
    fprintf(f_csv, "\n");

    fprintf(f_csv, "# Table 3: Loop Scheduling & Workload Imbalance (k = %d Threads)\n", LOGICAL_THREADS);
    fprintf(f_csv, "Table,OpenMP_Scheduling_Clause,Chunk_Size,Run1_Cold_s,Run2_s,Run3_s,Execution_Time_s,Observed_Behavior\n");
    for (int m = 0; m < 5; m++) {
        fprintf(f_csv, "Table3,\"%s\",\"%s\",%.6f,%.6f,%.6f,%.6f,\"%s\"\n",
                sched_names[m], sched_chunks[m], sched_w[m], sched_r2[m], sched_r3[m], sched_avg[m], sched_behaviors[m]);
    }

    fclose(f_csv);
    printf(">> All benchmark logs written to results.csv\n");

    return 0;
}
