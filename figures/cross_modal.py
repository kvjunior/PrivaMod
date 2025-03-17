import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Set up the matplotlib parameters for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 600,
})

# Create a professional color palette
dark_blue = "#1F4E79"
dark_red = "#C0392B"
light_blue = "#3498DB"
light_red = "#E74C3C"

# Create figure with subplots - increased size and spacing
fig = plt.figure(figsize=(9, 8))
gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

# Data for subplot (a): Modality contribution to price prediction across NFT categories
categories = ['Most Rare\n(Alien, Ape)', 'Medium Rare\n(Zombie)', 'Common\n(Male, Female)']
visual_contrib = [0.718, 0.684, 0.457]
transaction_contrib = [0.282, 0.316, 0.543]
error_bars = [0.024, 0.027, 0.021]  # standard error

# Subplot (a): Modality contribution
ax1 = fig.add_subplot(gs[0, 0])

# Create stacked bars
bar_width = 0.7
x = np.arange(len(categories))
b1 = ax1.bar(x, visual_contrib, bar_width, label='Visual Features', color=dark_blue, edgecolor='white', linewidth=0.5)
b2 = ax1.bar(x, transaction_contrib, bar_width, bottom=visual_contrib, label='Transaction Features', 
            color=dark_red, edgecolor='white', linewidth=0.5)

# Add error bars
for i, (v, e) in enumerate(zip(visual_contrib, error_bars)):
    ax1.errorbar(i, v, yerr=e, fmt='none', color='black', capsize=4, capthick=1, elinewidth=1)
    ax1.errorbar(i, v+transaction_contrib[i], yerr=e, fmt='none', color='black', capsize=4, capthick=1, elinewidth=1)

# Format axes
ax1.set_ylim(0, 1.05)
ax1.set_ylabel('Contribution to Price Prediction')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.set_title('(a) Modality Contribution by NFT Category', fontsize=11, pad=10)

# Add percentage labels inside bars with improved positioning
for i, (v, t) in enumerate(zip(visual_contrib, transaction_contrib)):
    # Visual percentage label
    ax1.text(i, v/2, f'{v:.1%}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    # Transaction percentage label
    ax1.text(i, v + t/2, f'{t:.1%}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)

# Improve axes appearance
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.set_axisbelow(True)

# Create custom legend below plot
handles = [mpatches.Patch(color=dark_blue, label='Visual Features'),
           mpatches.Patch(color=dark_red, label='Transaction Features')]
ax1.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Data for subplot (b): Fusion information gain
approaches = ['Visual Only', 'Transaction Only', 'PrivaMod Fusion']
gain_values = [0.0, 15.9, 33.4]  # Percentage gain compared to visual only
confidence = [0.0, 1.3, 1.4]  # 95% confidence intervals

# Subplot (b): Fusion information gain
ax2 = fig.add_subplot(gs[0, 1])

# Create bars with distinct colors
colors = [dark_blue, dark_red, light_blue]
bars = ax2.bar(approaches, gain_values, width=0.65, yerr=confidence, 
               color=colors, edgecolor='white', linewidth=0.5,
               capsize=4, error_kw={'elinewidth': 1, 'capthick': 1})

# Format axes
ax2.set_ylabel('Information Gain (%)')
ax2.set_title('(b) Fusion Information Gain', fontsize=11, pad=10)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.set_axisbelow(True)
ax2.set_ylim(0, 38)  # Give room for label and error bars

# Add value labels above bars with better positioning
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Add annotation inside the PrivaMod fusion bar
ax2.text(2, gain_values[2]/2, '+27.6%\ninformation\ngain', ha='center', 
         va='center', color='white', fontweight='bold', fontsize=8)

# Improve axes appearance
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Data for subplot (c): Impact of different fusion mechanisms
fusion_types = ['Simple\nConcatenation', 'Weighted\nAveraging', 'Attention\nBased', 'Bayesian\n(Ours)']
metrics = {
    'Market Efficiency': [0.843, 0.859, 0.867, 0.874],
    'R² Score': [0.834, 0.862, 0.878, 0.912],
    'MAE Reduction': [11.3, 19.7, 22.4, 27.6]
}

# Subplot (c): Fusion mechanisms impact
ax3 = fig.add_subplot(gs[1, 0])

# Set width of bars and positions
bar_width = 0.25
x = np.arange(len(fusion_types))
offsets = [-bar_width, 0, bar_width]
colors = [dark_blue, dark_red, light_blue]

# Create three groups of bars for each metric
for i, (metric, values) in enumerate(list(metrics.items())[:2]):  # First two metrics
    ax3.bar(x + offsets[i], values, bar_width, label=metric, color=colors[i], 
           edgecolor='white', linewidth=0.5)

# Create a twin axis for the MAE Reduction
ax3_right = ax3.twinx()
ax3_right.bar(x + offsets[2], metrics['MAE Reduction'], bar_width, label='MAE Reduction', 
              color=colors[2], edgecolor='white', linewidth=0.5)
ax3_right.set_ylabel('MAE Reduction (%)')

# Format main axes
ax3.set_ylim(0, 1.0)
ax3.set_ylabel('Performance Value')
ax3.set_xticks(x)
ax3.set_xticklabels(fusion_types)
ax3.set_title('(c) Impact of Fusion Mechanisms', fontsize=11, pad=10)
ax3.spines['top'].set_visible(False)
ax3.grid(axis='y', linestyle='--', alpha=0.3)
ax3.set_axisbelow(True)

# Format secondary y-axis
ax3_right.set_ylim(0, 100)
ax3_right.spines['top'].set_visible(False)

# Create combined legend below the plot
handles1, labels1 = ax3.get_legend_handles_labels()
handles2, labels2 = ax3_right.get_legend_handles_labels()
ax3.legend(handles1 + handles2, labels1 + labels2, 
          loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

# Data for subplot (d): Uncertainty calibration
expected_error = np.linspace(0, 0.1, 20)
observed_privamod = np.array([0.0, 0.005, 0.011, 0.016, 0.020, 0.025, 0.031, 0.036, 0.042, 
                             0.048, 0.053, 0.058, 0.064, 0.069, 0.075, 0.080, 0.085, 0.090, 0.095, 0.10])
observed_concat = np.array([0.023, 0.028, 0.032, 0.035, 0.042, 0.044, 0.052, 0.055, 0.060, 
                           0.065, 0.075, 0.076, 0.075, 0.084, 0.087, 0.088, 0.092, 0.094, 0.096, 0.10])

# Subplot (d): Uncertainty calibration
ax4 = fig.add_subplot(gs[1, 1])

# Create shaded regions
ax4.fill_between([0, 0.1], [0, 0.1], [0, 0], color='#E8F0FF', alpha=0.6, label='Over-confidence')
ax4.fill_between([0, 0.1], [0, 0.1], [0.1, 0.1], color='#FFEFEF', alpha=0.6, label='Under-confidence')

# Plot ideal calibration line
ax4.plot([0, 0.1], [0, 0.1], '--', color='gray', linewidth=1.5, label='Ideal Calibration')
ax4.plot([0, 0.1], [0, 0.11], ':', color='gray', alpha=0.7, linewidth=1)
ax4.plot([0, 0.1], [0, 0.09], ':', color='gray', alpha=0.7, linewidth=1)

# Plot observed vs expected error for both methods
ax4.plot(expected_error, observed_privamod, '-o', markersize=5, color=dark_blue, 
         label='PrivaMod (Bayesian)', linewidth=1.5)
ax4.plot(expected_error, observed_concat, '-s', markersize=5, color=dark_red, 
         label='Baseline (Concat)', linewidth=1.5)

# Format axes
ax4.set_xlim(0, 0.1)
ax4.set_ylim(0, 0.1)
ax4.set_xlabel('Expected Error Rate')
ax4.set_ylabel('Observed Error Rate')
ax4.set_title('(d) Uncertainty Calibration', fontsize=11, pad=10)
ax4.grid(linestyle='--', alpha=0.3)
ax4.set_axisbelow(True)

# Add annotation explaining calibration
ax4.text(0.06, 0.025, 'Better\nCalibration', color=dark_blue, fontsize=9, 
         ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, 
                                            edgecolor='none', boxstyle='round,pad=0.2'))

# Add legend
ax4.legend(loc='upper left', fontsize=8, framealpha=0.9)

# Improve axes appearance
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Ensure tight layout and save
plt.tight_layout()
plt.savefig('cross_modal.pdf', format='pdf', bbox_inches='tight', pad_inches=0.15)
plt.close()

print("Figure 'cross_modal.pdf' has been generated successfully.")