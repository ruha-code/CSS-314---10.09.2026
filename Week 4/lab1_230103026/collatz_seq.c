/*
 * Practicum: The Amdahl Reality Gap
 * Student ID: 230103026
 * Workload N: 13,026,000 (Formula: 10,000,000 + 3026 * 1000)
 * 
 * Compilation:
 *   gcc -O2 -fopenmp collatz_seq.c -o collatz_seq
 * Execution:
 *   ./collatz_seq
 */

#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#define STUDENT_ID 230103026ULL
#define WORKLOAD_N (10000000ULL + (3026ULL * 1000ULL))
#define MOD_VAL 1000000007ULL

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

void run_collatz_seq(uint64_t n, uint32_t *out_max, uint64_t *out_checksum) {
    uint32_t max_s = 0;
    uint64_t sum = 0;
    for (uint64_t i = 1; i <= n; i++) {
        uint32_t s = collatz_steps(i);
        if (s > max_s) {
            max_s = s;
        }
        sum += s;
        if (sum >= MOD_VAL) {
            sum %= MOD_VAL;
        }
    }
    *out_max = max_s;
    *out_checksum = sum;
}

int main(void) {
    uint64_t n = WORKLOAD_N;
    printf("====================================================\n");
    printf("Phase 2: Sequential Baseline & Precision Timing\n");
    printf("Student ID: %llu\n", (unsigned long long)STUDENT_ID);
    printf("Target Workload Size N: %llu\n", (unsigned long long)n);
    printf("====================================================\n");

    uint32_t max_steps = 0;
    uint64_t checksum = 0;

    // Run 1: Warmup (Discarded)
    printf("Running Run 1 (Warmup / Cold Cache)...\n");
    double start_warmup = omp_get_wtime();
    run_collatz_seq(n, &max_steps, &checksum);
    double t_warmup = omp_get_wtime() - start_warmup;
    printf("Run 1 (Warmup) Time : %.6f s [DISCARDED]\n", t_warmup);
    printf("Verification Checksum : %llu | Max Steps: %u\n\n",
           (unsigned long long)checksum, max_steps);

    // Run 2: Benchmark Run 2
    printf("Running Run 2 (Timed)...\n");
    double start_run2 = omp_get_wtime();
    run_collatz_seq(n, &max_steps, &checksum);
    double t_run2 = omp_get_wtime() - start_run2;
    printf("Run 2 Time          : %.6f s\n", t_run2);

    // Run 3: Benchmark Run 3
    printf("Running Run 3 (Timed)...\n");
    double start_run3 = omp_get_wtime();
    run_collatz_seq(n, &max_steps, &checksum);
    double t_run3 = omp_get_wtime() - start_run3;
    printf("Run 3 Time          : %.6f s\n\n", t_run3);

    double t_seq = (t_run2 + t_run3) / 2.0;
    printf("====================================================\n");
    printf("RESULTS SUMMARY:\n");
    printf("Max Stopping Steps  : %u\n", max_steps);
    printf("Checksum (mod 10^9+7): %llu\n", (unsigned long long)checksum);
    printf("Run 1 (Warmup)      : %.6f s\n", t_warmup);
    printf("Run 2               : %.6f s\n", t_run2);
    printf("Run 3               : %.6f s\n", t_run3);
    printf("T_seq (Average 2&3) : %.6f s\n", t_seq);
    printf("====================================================\n");

    return 0;
}
