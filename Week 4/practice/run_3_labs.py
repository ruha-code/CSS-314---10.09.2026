import os
import sys
import subprocess
import time
import datetime

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "results_230103026.log")

    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    logger = Logger(log_path)
    sys.stdout = logger

    print("=" * 70)
    print("PARALLEL COMPUTING LABORATORY BENCHMARK LOG (LABS 1–3)")
    print(f"Student: Ruslan Ussen | ID: 230103026")
    print(f"Host CPU: AMD Ryzen 7 7435HS (8 Physical Cores / 16 SMT Threads)")
    print(f"OS: Windows 11 Pro for Workstations")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    def run_lab(folder, script):
        cwd = os.path.join(base_dir, folder)
        print(f"\n>>> EXECUTING {folder}/{script}...")
        t0 = time.perf_counter()
        res = subprocess.run([sys.executable, script], cwd=cwd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print("STDERR:", res.stderr.strip())
        print(f">>> Finished {folder}/{script} in {dt:.2f}s (Exit Code: {res.returncode})\n")
        print("-" * 70)

    t_total = time.perf_counter()
    run_lab("lab1", "lab1.py")
    run_lab("lab2", "lab2.py")
    run_lab("lab3", "lab3.py")
    total_time = time.perf_counter() - t_total

    print("=" * 70)
    print(f"ALL 3 LABS COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print(f"Log written to: {log_path}")
    print("=" * 70)

    logger.log.close()
    sys.stdout = logger.terminal
    print(f"\nExecution log successfully saved to {log_path}")

if __name__ == "__main__":
    main()
