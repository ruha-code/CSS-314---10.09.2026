"""
High-Resolution Plotting Script for Amdahl Reality Gap
Student ID: 230103026
"""

import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "results.csv")
PLOT_PATH = os.path.join(BASE_DIR, "speedup_plot.png")

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found!")
    exit(1)

ks = []
s_emps = []
s_theos = []
p_val = 0.0

with open(CSV_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("# Derived Parallel Fraction p ="):
            p_val = float(line.split("=")[1].strip())
        elif line.startswith("Table1,"):
            parts = line.split(",")
            k = int(parts[1])
            s_emp = float(parts[6])
            s_theo = float(parts[7])
            ks.append(k)
            s_emps.append(s_emp)
            s_theos.append(s_theo)

s_ideals = ks

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(ks, s_ideals, 'k--', label='Linear Ideal Speedup $S(k) = k$', linewidth=1.5, alpha=0.7)
plt.plot(ks, s_theos, 'b-o', label=f'Amdahl Theoretical $S_{{theo}}(k)$ ($p = {p_val:.3f}$)', linewidth=2.0, markersize=7)
plt.plot(ks, s_emps, 'r-s', label='Empirical Measured $S_{{emp}}(k)$ (AMD Ryzen 7 7435HS)', linewidth=2.5, markersize=8)

# Shaded Reality Gap
plt.fill_between(ks, s_emps, s_theos, color='#ff9999', alpha=0.35, label='Amdahl Reality Gap $\\Delta(k) = S_{theo} - S_{emp}$')

# Mark physical cores
plt.axvline(x=8, color='#660099', linestyle=':', linewidth=2.0, label='Physical Cores Limit ($k = 8$)')
plt.text(8.2, max(s_emps) * 0.55, 'SMT / Hyperthreading\nContention Regime ($k > 8$)', color='#660099', fontsize=9, fontweight='bold')

plt.title('Amdahl Reality Gap & Empirical Multi-Core Scalability\nAMD Ryzen 7 7435HS (8 Cores / 16 SMT Threads) | Student ID: 230103026', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Number of Worker Threads ($k$)', fontsize=11, fontweight='bold')
plt.ylabel('Speedup Factor ($S$)', fontsize=11, fontweight='bold')
plt.xticks(ks)
plt.xlim(0.8, 16.5)
plt.ylim(0.5, max(s_theos) * 1.15)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', framealpha=0.95, fontsize=10)
plt.tight_layout()

plt.savefig(PLOT_PATH, dpi=300)
plt.close()
print(f"High-resolution speedup plot saved successfully to {PLOT_PATH}")
