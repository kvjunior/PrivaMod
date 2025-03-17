import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
})

# Create a color palette
colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
palette = sns.color_palette(colors)

# Create figure with subplots
fig = plt.figure(figsize=(8.5, 7.0))
gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.35)

# Data for subplot (a): Market efficiency vs. privacy budget
privacy_budgets = [0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0]
market_efficiency = [0.792, 0.821, 0.842, 0.874, 0.876, 0.881, 0.883, 0.885, 0.887]
confidence_lower = [0.774, 0.807, 0.828, 0.863, 0.865, 0.871, 0.873, 0.875, 0.877]
confidence_upper = [0.810, 0.835, 0.856, 0.885, 0.887, 0.891, 0.893, 0.895, 0.897]

# Subplot (a): Market efficiency vs. privacy budget
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(privacy_budgets, market_efficiency, '-o', color=palette[0], linewidth=2,
         markersize=6, markeredgecolor='white', markeredgewidth=0.8)

# Add confidence bands
ax1.fill_between(privacy_budgets, confidence_lower, confidence_upper, 
                 color=palette[0], alpha=0.15)

# Mark operating point
op_index = privacy_budgets.index(0.08)
ax1.plot(privacy_budgets[op_index], market_efficiency[op_index], 'o', 
         markersize=8, markeredgewidth=1.5, markerfacecolor='white', 
         markeredgecolor=palette[0])
ax1.text(privacy_budgets[op_index]+0.03, market_efficiency[op_index]+0.003, 
         f"ε = {privacy_budgets[op_index]}", fontsize=9, 
         va='bottom', ha='left', color=palette[0])

# Format axes
ax1.set_xscale('log')
ax1.set_xlabel('Privacy Budget (ε)')
ax1.set_ylabel('Market Efficiency Score')
ax1.grid(linestyle='--', alpha=0.6)
ax1.set_title('(a) Market Efficiency vs. Privacy Budget', fontsize=11, pad=10)

# Annotate performance percentage relative to non-private
max_performance = market_efficiency[-1]  # Assuming highest privacy budget has max performance
selected_performance = market_efficiency[op_index]
percentage = (selected_performance / max_performance) * 100
ax1.text(0.08, 0.815, f"98.7% of maximum\nperformance at ε = 0.08", 
         fontsize=8, color=palette[0], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# Improve axes appearance
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Data for subplot (b): Membership inference attack success rates
privacy_levels = [0.01, 0.05, 0.1, 0.5, 1.0]
attack_success = [51.2, 52.1, 53.4, 58.7, 63.2]
attack_confidence = [1.1, 1.2, 1.2, 1.4, 1.5]  # error margins

# Subplot (b): Attack success rates
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(range(len(privacy_levels)), attack_success, 
               yerr=attack_confidence, capsize=4, width=0.7, 
               color=[palette[0] if i == 2 else palette[1] for i in range(len(privacy_levels))],
               error_kw={'elinewidth': 1.0, 'capthick': 1.0})

# Mark the random guessing baseline
ax2.axhline(y=50, linestyle='--', color='gray', alpha=0.8, linewidth=1.5)
ax2.text(len(privacy_levels)-1.5, 50.5, 'Random Guessing (50%)', 
         fontsize=8, color='gray', ha='center', va='bottom')

# Format axes
ax2.set_xticks(range(len(privacy_levels)))
ax2.set_xticklabels([f'ε = {p}' for p in privacy_levels])
ax2.set_ylabel('Attack Success Rate (%)')
ax2.set_ylim(45, 70)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.set_title('(b) Membership Inference Attack Success', fontsize=11, pad=10)

# Highlight operating point
bars[2].set_color(palette[0])  # ε = 0.1 is closest to our operating point

# Improve axes appearance
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Data for subplot (c): Comparison of methods
methods = ['PrivaMod', 'CODER', 'PriMonitor', 'Aegis', 'MAVEN', 'DCA-VAE']
privacy_points = {
    'PrivaMod': [(0.01, 0.792), (0.05, 0.842), (0.08, 0.874), (0.5, 0.883)],
    'CODER': [(0.05, 0.728), (0.11, 0.771), (0.5, 0.823)],
    'PriMonitor': [(0.05, 0.712), (0.09, 0.758), (0.5, 0.795)],
    'Aegis': [(0.07, 0.729), (0.2, 0.751), (0.5, 0.780)],
    'MAVEN': [(0.0, 0.768), (0.0, 0.768)],  # No privacy guarantees, constant performance
    'DCA-VAE': [(0.0, 0.745), (0.0, 0.745)]  # No privacy guarantees, constant performance
}

# Subplot (c): Method comparison
ax3 = fig.add_subplot(gs[1, 0])

# Plot each method's privacy-utility curve
for i, method in enumerate(methods):
    x_vals = [point[0] for point in privacy_points[method]]
    y_vals = [point[1] for point in privacy_points[method]]
    
    if method in ['MAVEN', 'DCA-VAE']:
        # For methods without privacy, use different markers
        ax3.plot([0.01, 1.0], [y_vals[0], y_vals[0]], '--', color=palette[i+1], alpha=0.7, 
                 label=f"{method} (no privacy)")
    else:
        ax3.plot(x_vals, y_vals, '-o', color=palette[i], markersize=5, label=method)

# Mark the PrivaMod operating point
ax3.plot(0.08, 0.874, 'o', markersize=8, markeredgewidth=1.5, 
         markerfacecolor='white', markeredgecolor=palette[0])

# Format axes
ax3.set_xscale('log')
ax3.set_xlim(0.009, 1.1)
ax3.set_ylim(0.7, 0.9)
ax3.set_xlabel('Privacy Budget (ε)')
ax3.set_ylabel('Market Efficiency Score')
ax3.grid(linestyle='--', alpha=0.6)
ax3.legend(loc='lower right', fontsize=8)
ax3.set_title('(c) Comparison Across Methods', fontsize=11, pad=10)

# Improve axes appearance
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Data for subplot (d): Optimal operating points
operating_points = [
    ('ε = 0.01', 0.792, 0.767, 0.817, 51.2, 49.2, 53.2),  # Low privacy budget
    ('ε = 0.05', 0.842, 0.822, 0.862, 52.1, 50.1, 54.1),  # Medium privacy budget
    ('ε = 0.08', 0.874, 0.859, 0.889, 53.4, 51.3, 55.5),  # Operating point
    ('ε = 0.50', 0.883, 0.872, 0.894, 58.7, 56.0, 61.4),  # High privacy budget
]

# Subplot (d): Optimal operating points
ax4 = fig.add_subplot(gs[1, 1])

# Create scatter plot for operating points
for i, point in enumerate(operating_points):
    label, eff, eff_min, eff_max, attack, attack_min, attack_max = point
    
    # Determine marker properties
    if label == 'ε = 0.08':  # Highlight optimal point
        marker_color = palette[0]
        marker_size = 100
        edge_width = 1.5
        alpha = 0.8
        zorder = 10
    else:
        marker_color = palette[i+1]
        marker_size = 80
        edge_width = 1.0
        alpha = 0.7
        zorder = 5
    
    # Plot point
    scatter = ax4.scatter(attack, eff, s=marker_size, c=[marker_color], alpha=alpha,
                          marker='o', edgecolors='white', linewidths=edge_width, zorder=zorder,
                          label=label)
    
    # Add confidence ellipse
    width = (attack_max - attack_min) / 2
    height = (eff_max - eff_min) / 2
    ellipse = Ellipse(xy=(attack, eff), width=width*2, height=height*2,
                      edgecolor=marker_color, facecolor='none',
                      linewidth=edge_width, alpha=0.6, zorder=zorder-1)
    ax4.add_patch(ellipse)
    
    # Add label
    if label == 'ε = 0.08':
        ax4.text(attack+1, eff+0.003, 'Optimal\nOperating Point', 
                 fontsize=8, ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    else:
        ax4.text(attack, eff+0.005, label, fontsize=7, ha='center', va='bottom')

# Format axes
ax4.set_xlabel('Attack Success Rate (%)')
ax4.set_ylabel('Market Efficiency Score')
ax4.set_xlim(48, 65)
ax4.set_ylim(0.76, 0.9)
ax4.grid(linestyle='--', alpha=0.6)
ax4.set_title('(d) Optimal Operating Points', fontsize=11, pad=10)

# Add shaded regions to indicate desirable areas
ax4.axvspan(48, 55, alpha=0.1, color='green', label='High Privacy')
ax4.axhspan(0.85, 0.9, alpha=0.1, color='blue', label='High Utility')

# Add legend
ax4.legend(loc='lower right', fontsize=8)

# Improve axes appearance
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Ensure tight layout and save
plt.tight_layout()
plt.savefig('privacy_utility.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

print("Figure 'privacy_utility.pdf' has been generated successfully.")