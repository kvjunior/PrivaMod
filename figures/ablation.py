import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Increase figure size and set up improved parameters
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

# Create a color palette based on professional academic standards
colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
palette = sns.color_palette(colors)

# Create figure with improved dimensions and spacing
fig = plt.figure(figsize=(8.5, 7.0))
gs = GridSpec(2, 2, figure=fig, wspace=0.32, hspace=0.4)

# Data for subplot (a): Impact of removing key components
components = ['Full\nPrivaMod', 'No Privacy\nMechanisms', 'No Contrastive\nLearning', 
              'No Temporal\nGNN', 'No Bayesian\nFusion', 'No Graph\nComponents']
market_efficiency = [0.874, 0.891, 0.856, 0.859, 0.821, 0.862]
confidence_intervals = [0.011, 0.010, 0.012, 0.012, 0.013, 0.012]

# Subplot (a): Impact of removing key components with improved spacing
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(np.arange(len(components)), market_efficiency, color=palette, 
               yerr=confidence_intervals, capsize=4, width=0.65, 
               error_kw={'elinewidth': 1.0, 'capthick': 1.0})

# Apply distinct colors
bars[0].set_color(palette[0])  # Full PrivaMod
bars[1].set_color(palette[1])  # No Privacy Mechanisms
for i in range(2, len(bars)):
    bars[i].set_color(palette[2])  # Other components

ax1.set_ylim(0.79, 0.92)
ax1.set_ylabel('Market Efficiency Score')
ax1.set_xticks(np.arange(len(components)))
ax1.set_xticklabels(components, rotation=45, ha='center', fontsize=8)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.set_title('(a) Impact of Removing Key Components', fontsize=11, pad=10)

# Add significance markers with improved positioning
for i in [1, 2, 4]:  # p < 0.01 for these components
    ax1.text(i, market_efficiency[i] + confidence_intervals[i] + 0.007, '**', 
             ha='center', fontsize=9, fontweight='bold')
for i in [3, 5]:  # p < 0.05 for these components
    ax1.text(i, market_efficiency[i] + confidence_intervals[i] + 0.007, '*', 
             ha='center', fontsize=9, fontweight='bold')

# Improve axes appearance
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='both', which='both', direction='out')

# Data for subplot (b): Relative contribution of architectural elements
elements = ['Bayesian\nFusion', 'Vision\nTransformer', 'Contrastive\nLearning', 
            'Temporal GNN', 'Graph\nComponents', 'Privacy\nMechanisms']
contributions = [31.4, 21.8, 18.6, 14.2, 9.5, 4.5]

# Subplot (b): Relative contribution with improved legend
ax2 = fig.add_subplot(gs[0, 1])
wedges, texts, autotexts = ax2.pie(contributions, labels=None, autopct='%1.1f%%', 
                                   startangle=90, counterclock=False,
                                   colors=palette[:len(elements)], 
                                   wedgeprops={'edgecolor': 'w', 'linewidth': 1.2})

# Improve percentage labels
for autotext in autotexts:
    autotext.set_fontsize(8)
    autotext.set_color('white')
    autotext.set_weight('bold')

# Add legend with improved positioning
ax2.legend(wedges, elements, title=None,
           loc="center left", bbox_to_anchor=(1.05, 0.5), 
           frameon=False, fontsize=9)
ax2.set_title('(b) Relative Contribution of Architectural Elements', fontsize=11, pad=10)

# Data for subplot (c): Parameter sensitivity analysis - using the same data
param_names = ['Contrastive\nTemperature', 'Privacy Clipping\nThreshold', 
               'Fusion Weight\nBalancing', 'Learning\nRate']
param_values = {
    'Contrastive\nTemperature': [0.03, 0.05, 0.07, 0.09, 0.11],
    'Privacy Clipping\nThreshold': [0.4, 0.7, 1.0, 1.3, 1.6],
    'Fusion Weight\nBalancing': [0.01, 0.05, 0.1, 0.15, 0.2],
    'Learning\nRate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
}
efficiency_scores = {
    'Contrastive\nTemperature': [0.841, 0.862, 0.874, 0.856, 0.830],
    'Privacy Clipping\nThreshold': [0.834, 0.858, 0.874, 0.865, 0.843],
    'Fusion Weight\nBalancing': [0.855, 0.868, 0.874, 0.865, 0.847],
    'Learning\nRate': [0.832, 0.860, 0.874, 0.851, 0.820]
}
std_errors = {
    'Contrastive\nTemperature': [0.013, 0.012, 0.011, 0.012, 0.014],
    'Privacy Clipping\nThreshold': [0.014, 0.012, 0.011, 0.011, 0.013],
    'Fusion Weight\nBalancing': [0.012, 0.011, 0.011, 0.011, 0.012],
    'Learning\nRate': [0.014, 0.012, 0.011, 0.012, 0.015]
}

# Subplot (c): Parameter sensitivity analysis with improved legend
ax3 = fig.add_subplot(gs[1, 0])
for i, param in enumerate(param_names):
    x_vals = param_values[param]
    y_vals = efficiency_scores[param]
    errors = std_errors[param]
    
    # Normalize x-axis to a standard scale
    x_norm = np.linspace(0, 1, len(x_vals))
    
    ax3.errorbar(x_norm, y_vals, yerr=errors, fmt='-o', 
                 color=palette[i], label=param, capsize=3,
                 markersize=5, markeredgecolor='white', markeredgewidth=0.5)

ax3.set_ylim(0.81, 0.89)
ax3.set_xlabel('Normalized Parameter Value')
ax3.set_ylabel('Market Efficiency Score')
ax3.grid(linestyle='--', alpha=0.6)
ax3.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax3.set_xticklabels(['Min', '25%', '50%', '75%', 'Max'])

# Improved legend positioning
ax3.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='gray',
           loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
ax3.set_title('(c) Parameter Sensitivity Analysis', fontsize=11, pad=10)

# Improve axes appearance
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Data for subplot (d): Generalization performance
data_sizes = [10, 25, 50, 75, 100]
performance = [0.783, 0.842, 0.889, 0.912, 0.942]
confidence_lower = [0.748, 0.817, 0.871, 0.894, 0.924]
confidence_upper = [0.818, 0.867, 0.907, 0.930, 0.960]

# Subplot (d): Generalization performance with improved layout
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(data_sizes, performance, '-o', color=palette[0], 
         markersize=6, markeredgecolor='white', markeredgewidth=0.8, 
         label='Performance')

# Add confidence region
ax4.fill_between(data_sizes, confidence_lower, confidence_upper, 
                 color=palette[0], alpha=0.2, label='95% Confidence Region')

# Add percentage of maximum performance with improved positioning
for i, size in enumerate(data_sizes):
    percentage = (performance[i] / max(performance)) * 100
    ax4.text(size, performance[i] + 0.022, f"{percentage:.1f}%", 
             fontsize=8, ha='center')

ax4.set_xlim(5, 105)
ax4.set_ylim(0.7, 1.0)
ax4.set_xlabel('Training Dataset Size (%)')
ax4.set_ylabel('Normalized Performance')
ax4.grid(linestyle='--', alpha=0.6)
ax4.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='gray',
           loc='lower right')
ax4.set_title('(d) Generalization Performance', fontsize=11, pad=10)

# Improve axes appearance
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Add annotation with improved positioning
fig.text(0.01, 0.01, 
         "* p < 0.05, ** p < 0.01 compared to full PrivaMod", 
         fontsize=4, style='italic')

# Ensure tight layout and save
plt.tight_layout()
plt.savefig('ablation.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()