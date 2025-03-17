import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import pandas as pd
from datetime import datetime, timedelta

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

# Create a color palette based on the paper's style
colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
community_colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", 
                    "#34495E", "#D35400", "#27AE60", "#8E44AD", "#16A085", "#2980B9"]
palette = sns.color_palette(colors)

# Create figure with subplots
fig = plt.figure(figsize=(8.5, 7.0))
gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.35)

# --- Subplot (a): Community Structure with Modularity-Based Coloring ---
ax1 = fig.add_subplot(gs[0, 0])

# Create a graph with community structure
# For demonstration, we'll create a synthetic network with communities
# In a real scenario, you would load actual transaction network data
np.random.seed(42)  # For reproducibility
G = nx.random_partition_graph([12, 15, 10, 8, 5], 0.25, 0.01)

# Get community assignments
communities = {node: 0 for node in range(0, 12)}
communities.update({node: 1 for node in range(12, 27)})
communities.update({node: 2 for node in range(27, 37)})
communities.update({node: 3 for node in range(37, 45)})
communities.update({node: 4 for node in range(45, 50)})

# Compute node size based on degree (can be replaced with actual trading volume)
node_size = [20 + 5 * G.degree(n) for n in G.nodes()]

# Compute layout for graph visualization
pos = nx.spring_layout(G, seed=42)

# Draw nodes colored by community
for comm_id in set(communities.values()):
    node_list = [node for node in G.nodes() if communities[node] == comm_id]
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=node_list,
        node_color=community_colors[comm_id],
        node_size=[node_size[n] for n in node_list],
        alpha=0.8,
        label=f"Community {comm_id+1}",
        ax=ax1
    )

# Draw edges with low alpha for clarity
nx.draw_networkx_edges(
    G, pos, 
    alpha=0.15, 
    width=0.5,
    ax=ax1
)

# Add modularity score annotation
modularity_score = 0.684  # From the paper
ax1.text(0.05, 0.95, f"Modularity Score: {modularity_score:.3f}", 
         transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add legend and formatting
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9, markerscale=0.8)
ax1.set_title('(a) Community Structure in Transaction Network', fontsize=11, pad=10)
ax1.set_axis_off()

# --- Subplot (b): Temporal Trading Patterns ---
ax2 = fig.add_subplot(gs[0, 1])

# Generate sample temporal data
# Daily pattern - 24 hours
hours = np.arange(24)
daily_activity = 100 + 30 * np.sin(np.pi * hours / 12) + 50 * np.exp(-(hours - 20)**2 / 20)
daily_std = 10 + 5 * np.sin(np.pi * hours / 12)

# Weekly pattern - 7 days
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
weekly_activity = [142, 135, 128, 145, 162, 180, 155]
weekly_std = [12, 10, 11, 13, 15, 16, 14]

# Plot daily pattern
ax2.plot(hours, daily_activity, 'o-', color=palette[0], label='Daily Trading Pattern', 
         markersize=4, markeredgecolor='white', markeredgewidth=0.5)
ax2.fill_between(hours, daily_activity - daily_std, daily_activity + daily_std, 
                 color=palette[0], alpha=0.2)

# Format axes for daily pattern
ax2.set_xlabel('Hour of Day (UTC)')
ax2.set_ylabel('Transaction Volume')
ax2.set_xlim(-0.5, 23.5)
ax2.set_xticks(np.arange(0, 24, 4))
ax2.grid(linestyle='--', alpha=0.6)

# Add twin axis for weekly pattern
ax2_twin = ax2.twiny()
ax2_twin.plot(range(7), weekly_activity, 's-', color=palette[1], 
              label='Weekly Trading Pattern', markersize=5, 
              markeredgecolor='white', markeredgewidth=0.5)
ax2_twin.fill_between(range(7), 
                      [w - s for w, s in zip(weekly_activity, weekly_std)], 
                      [w + s for w, s in zip(weekly_activity, weekly_std)], 
                      color=palette[1], alpha=0.2)

# Format twin axis for weekly pattern
ax2_twin.set_xlim(-0.5, 6.5)
ax2_twin.set_xticks(range(7))
ax2_twin.set_xticklabels(days)
ax2_twin.tick_params(axis='x', colors=palette[1])
ax2_twin.spines['top'].set_color(palette[1])

# Highlight peak trading times
ax2.axvline(x=20, color=palette[0], linestyle='--', alpha=0.6)
ax2.text(20.2, daily_activity[20] - 40, 'Peak Hour\n(20:00 UTC)', 
         fontsize=8, color=palette[0])

ax2_twin.axvline(x=5, color=palette[1], linestyle='--', alpha=0.6)
ax2_twin.text(5.1, weekly_activity[5] - 35, 'Peak Day\n(Saturday)', 
              fontsize=8, color=palette[1])

# Add legend and formatting
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)
ax2.set_title('(b) Temporal Trading Patterns', fontsize=11, pad=10)

# --- Subplot (c): Node Centrality vs Trading Volume ---
ax3 = fig.add_subplot(gs[1, 0])

# Generate sample data for centrality vs volume relationship
np.random.seed(42)
n_points = 40
centrality = np.random.gamma(2, 0.25, n_points)
volume = 1000 * centrality + 500 * np.random.normal(0, 0.5, n_points)
error = 200 + 100 * np.random.random(n_points)

# Sort for better visualization
idx = np.argsort(centrality)
centrality = centrality[idx]
volume = volume[idx]
error = error[idx]

# Calculate best fit line
z = np.polyfit(centrality, volume, 1)
p = np.poly1d(z)
x_line = np.linspace(min(centrality), max(centrality), 100)
y_line = p(x_line)

# Plot data points with error bars
ax3.errorbar(centrality, volume, yerr=error, fmt='o', color=palette[0], 
             markersize=6, markeredgecolor='white', markeredgewidth=0.5,
             capsize=3, elinewidth=0.8, alpha=0.7)

# Plot trend line
ax3.plot(x_line, y_line, '-', color=palette[0], alpha=0.8, linewidth=1.5)

# Add correlation coefficient
correlation = 0.793  # From the paper
ax3.text(0.05, 0.95, f"Correlation: {correlation:.3f}", 
         transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Format axes
ax3.set_xlabel('Node Degree Centrality')
ax3.set_ylabel('Trading Volume (ETH)')
ax3.grid(linestyle='--', alpha=0.6)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_title('(c) Node Centrality vs Trading Volume', fontsize=11, pad=10)

# --- Subplot (d): Feature Importance ---
ax4 = fig.add_subplot(gs[1, 1])

# Sample feature importance data
features = [
    'Recency',
    'Price Trend',
    'Network Centrality',
    'Trading Frequency',
    'Community Position',
    'Transaction Count',
    'Temporal Pattern',
    'Address Diversity'
]

# Feature importance values and confidence intervals
importance = [0.721, 0.684, 0.632, 0.598, 0.543, 0.487, 0.432, 0.376]
confidence = [0.019, 0.021, 0.023, 0.022, 0.025, 0.026, 0.024, 0.027]

# Sort features by importance
sorted_indices = np.argsort(importance)
features = [features[i] for i in sorted_indices]
importance = [importance[i] for i in sorted_indices]
confidence = [confidence[i] for i in sorted_indices]

# Plot horizontal bar chart with confidence intervals
y_pos = np.arange(len(features))
ax4.barh(y_pos, importance, xerr=confidence, align='center', 
         height=0.6, color=palette[0], alpha=0.7, 
         error_kw=dict(ecolor='black', lw=1, capsize=3, capthick=1))

# Format axes
ax4.set_yticks(y_pos)
ax4.set_yticklabels(features)
ax4.invert_yaxis()  # Most important at top
ax4.set_xlabel('Feature Importance')
ax4.grid(axis='x', linestyle='--', alpha=0.6)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set_title('(d) Transaction Feature Importance', fontsize=11, pad=10)

# Add importance values next to bars
for i, v in enumerate(importance):
    ax4.text(v + 0.03, i, f"{v:.3f}", va='center', fontsize=8)

# Save the figure
plt.tight_layout()
plt.savefig('transaction_network.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

print("Figure 'transaction_network.pdf' has been generated successfully.")