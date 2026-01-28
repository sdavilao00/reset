# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 23:40:05 2026

@author: sdavilao
"""
#%% necessary packages for code

import os
import pandas as pd
import glob
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%% Combined modeled Recurrence Interval data
 
# Path to your CSV folder
folder_path = r"C:\Users\sdavilao\Documents\newcodesoil\results\new"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Regex to extract trailing number and "extX"
run_value_pattern = re.compile(r"(\d+)(?=\.csv$)", re.IGNORECASE)
extent_pattern = re.compile(r"(ext\d+)", re.IGNORECASE)

dfs = []

for file in csv_files:
    filename = os.path.basename(file)

    # Extract Run_Value from end of filename
    run_match = run_value_pattern.search(filename)
    # Extract extent (e.g., ext15)
    ext_match = extent_pattern.search(filename)

    if run_match and ext_match:
        run_value = int(run_match.group(1))
        extent = ext_match.group(1)
        df = pd.read_csv(file)
        df['Cohesion'] = run_value
        df['Extent'] = extent
        dfs.append(df)
    else:
        print(f"⚠️ Skipping file: {filename} (missing Run_Value or Extent)")

# Combine all into a single big DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Filter for slope > 28
filtered_df = combined_df[combined_df['Avg_Slope_deg'] > 25]

#remove bad hollows
filtered_df = filtered_df.drop([38,40,44])

filtered_df = filtered_df.dropna(subset=['slope_mid'])
filtered_df['slope_mid'] = filtered_df['slope_mid'].astype(float)

print("✅ Combined and filtered files with Extent and Run_Value.")

#%% Plot Recurrence Interval as a function of hollow slope
## Use an inverse model to fit data
# Include critical hollow calculation

# Load data
df = filtered_df

# Log-space inverse model
def inverse_model_log(theta, loga, b):
    return loga - np.log10(theta + b)

# Color-blind-friendly palette
cbf_colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7', '#0072B2']

plt.figure(figsize=(9, 6))
ax = plt.gca()
ax.set_facecolor("#f0f0f0")
ax.grid(True, alpha=0.4)

legend_entries = []

cohesions = sorted(df["Cohesion"].unique(), reverse=True)

for i, coh in enumerate(cohesions):
    g = df[df["Cohesion"] == coh]
    x = g["Avg_Slope_deg"].values
    y = g["Year"].values
    color = cbf_colors[i % len(cbf_colors)]

    # Scatter
    ax.scatter(x, y, color=color, s=60, alpha=0.75)

    if len(x) < 3:
        continue

    # Fit in log space
    popt, _ = curve_fit(
        inverse_model_log,
        x,
        np.log10(y),
        p0=[3.0, -25]
    )

    loga_fit, b_fit = popt
    a_fit = 10**loga_fit

    # Log-space R²
    y_log = np.log10(y)
    y_log_pred = inverse_model_log(x, loga_fit, b_fit)
    ss_res = np.sum((y_log - y_log_pred) ** 2)
    ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
    r2_log = 1 - ss_res / ss_tot

    # Smooth curve
    x_fit = np.linspace(min(x) - 0.5, max(x) + 0.5, 300)
    y_fit = a_fit / (x_fit + b_fit)

    line, = ax.plot(x_fit, y_fit, color=color, lw=2.5)

    label = (
        f"{int(coh)} Pa: "
        f"$RI=\\frac{{{a_fit:.1f}}}{{\\theta_H+{b_fit:.2f}}}$, "
        f"$R^2_{{\\log}}={r2_log:.2f}$"
    )
    legend_entries.append((line, label))

# Axes
ax.set_yscale("log")
# ax.set_xscale('log')
# ax.set_xlabel("Hollow slope, $\\theta_H$ (°)", fontsize=13, fontweight="bold")
# ax.set_ylabel("Recurrence interval, RI (years)", fontsize=13, fontweight="bold")

# Legend
if legend_entries:
    handles, labels = zip(*legend_entries)
    ax.legend(handles, labels, title="Cohesion (log-space fit)",
              fontsize=10, title_fontsize=11,
              loc="upper right", frameon=True)

# X ticks
xticks = np.arange(
    np.floor(df["Avg_Slope_deg"].min() / 3) * 3,
    df["Avg_Slope_deg"].max() + 1, 3
)
ax.set_xticks(xticks)

plt.tight_layout()
plt.savefig("C:/Users/sdavilao/Documents/RI_3D_Fit_updates.png", dpi=450, bbox_inches='tight')
plt.show()