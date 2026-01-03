import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# 1. STYLE CONFIGURATION (Presentation Ready)
# ═══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="white", context="poster", font_scale=1.1) # 'poster' context = HUGE fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 300 

# High Contrast Colors
COLOR_SEPARATION = "#D62728"  # Strong Red
COLOR_ZERO = "#333333"        # Dark Grey

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def get_separation_data(filepath, window_size=30):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    risk_data = []
    rand_data = []
    
    if "Risk_Dishes" in data:
        for seed, history in data["Risk_Dishes"].items():
            for entry in history:
                if entry['step'] > 20: 
                    risk_data.append({'step': entry['step'], 'val': entry['grad_alignment']})

    if "Random_Dishes" in data:
        for seed, history in data["Random_Dishes"].items():
            for entry in history:
                if entry['step'] > 20: 
                    rand_data.append({'step': entry['step'], 'val': entry['grad_alignment']})
    
    df_risk = pd.DataFrame(risk_data)
    df_rand = pd.DataFrame(rand_data)
    
    risk_mean = df_risk.groupby("step")['val'].mean()
    rand_mean = df_rand.groupby("step")['val'].mean()
    
    common_steps = risk_mean.index.intersection(rand_mean.index)
    
    gap = risk_mean.loc[common_steps] - rand_mean.loc[common_steps]
    
    df_gap = pd.DataFrame({'step': common_steps, 'gap': gap})
    df_gap['gap_smooth'] = df_gap['gap'].rolling(window=window_size, center=True).mean()
    
    return df_gap

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLOTTING (Widescreen & Bold)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_separation(df_gap):
    # UPDATED: Huge 16x9 Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 1. Zero Line
    ax.axhline(0, color=COLOR_ZERO, linestyle='--', linewidth=2.5, alpha=0.6, label="No Difference")
    
    # 2. Raw Shadow
    ax.plot(df_gap['step'], df_gap['gap'], color=COLOR_SEPARATION, alpha=0.12, linewidth=1.5)
    
    # 3. Trend Line (Thicker)
    ax.plot(df_gap['step'], df_gap['gap_smooth'], color=COLOR_SEPARATION, linewidth=5, label="Risk Retention Signal")
    
    # 4. Danger Zone Fill
    ax.fill_between(df_gap['step'], df_gap['gap_smooth'], 0, 
                    where=(df_gap['gap_smooth'] > 0),
                    color=COLOR_SEPARATION, alpha=0.15)

    # ════════════════════════════════════════════════════════════
    # ANNOTATIONS
    # ════════════════════════════════════════════════════════════
    
    # Increased font sizes for big screen
    ax.set_title("Gradient Subspace Overlap Separation Over Time", 
                 fontweight="bold", fontsize=28, pad=30, loc='left')
    ax.set_xlabel("Training Steps", fontweight="bold", fontsize=22)
    ax.set_ylabel("Net Risk Alignment (Risk - Control)", fontweight="bold", fontsize=22)
    
    # Max Divergence Callout
    max_idx = df_gap['gap_smooth'].idxmax()
    if not pd.isna(max_idx):
        max_x = df_gap.loc[max_idx, 'step']
        max_y = df_gap.loc[max_idx, 'gap_smooth']
        
        ax.annotate(f"Max Divergence\n(Risk 'Sticks' while Control Fades)", 
                    xy=(max_x, max_y), xytext=(max_x + 20, max_y + 0.0005),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10),
                    fontsize=18, fontweight='bold',
                    backgroundcolor="white") # Added background to text for readability

    # Tight Zoom
    y_vals = df_gap['gap_smooth'].dropna()
    if not y_vals.empty:
        y_max = y_vals.max()
        y_min = min(0, y_vals.min()) 
        padding = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.legend(loc="upper left", frameon=False, fontsize=18)
    
    plt.tight_layout()
    plt.savefig("plot_signal_separation_large.png")
    print("Graph generated: plot_signal_separation_large.png (16x9)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Path to aggregated_results.json")
    args = parser.parse_args()
    
    df = get_separation_data(args.results_file)
    plot_separation(df)