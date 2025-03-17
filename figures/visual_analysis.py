import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.decomposition import PCA
from scipy import stats

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

# Create a color palette consistent with the paper style
colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
palette = sns.color_palette(colors)

# Function to create confidence ellipse
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radius.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # Same for y
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)

# Create figure with subplots
fig = plt.figure(figsize=(8.5, 7.0))
gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.35)

# --- Subplot (a): Dimensionality Reduction with Price-Colored Points ---
ax1 = fig.add_subplot(gs[0, 0])

# Generate sample data representing dimensionality-reduced NFT features
np.random.seed(42)
n_points = 200

# Create clusters representing different NFT types
groups = {
    "Alien": {"center": (-5, 5), "n": 15, "price_range": (800, 4200), "std": 0.7},
    "Ape": {"center": (-3, 2), "n": 25, "price_range": (500, 1500), "std": 0.8},
    "Zombie": {"center": (0, 4), "n": 40, "price_range": (250, 800), "std": 1.0},
    "Common Female": {"center": (2, 0), "n": 60, "price_range": (10, 50), "std": 1.5},
    "Common Male": {"center": (4, -2), "n": 60, "price_range": (5, 40), "std": 1.5},
}

# Generate points for each group
points_x, points_y, prices, groups_list = [], [], [], []
for group_name, group_info in groups.items():
    # Generate cluster points
    x = np.random.normal(group_info["center"][0], group_info["std"], group_info["n"])
    y = np.random.normal(group_info["center"][1], group_info["std"], group_info["n"])
    
    # Generate prices within range
    price_min, price_max = group_info["price_range"]
    p = np.random.uniform(price_min, price_max, group_info["n"])
    
    points_x.extend(x)
    points_y.extend(y)
    prices.extend(p)
    groups_list.extend([group_name] * group_info["n"])

# Convert to numpy arrays
points_x = np.array(points_x)
points_y = np.array(points_y)
prices = np.array(prices)
groups_array = np.array(groups_list)

# Create scatter plot with prices mapped to colors
sc = ax1.scatter(points_x, points_y, c=np.log10(prices), cmap='viridis', 
                 alpha=0.7, s=30, edgecolor='w', linewidth=0.5)

# Add 95% confidence ellipses for each group
for group_name in groups.keys():
    mask = groups_array == group_name
    if np.sum(mask) > 2:  # Need at least 3 points for covariance
        confidence_ellipse(
            points_x[mask], points_y[mask], ax1, n_std=2.0,
            edgecolor=palette[list(groups.keys()).index(group_name) % len(palette)],
            alpha=0.6, linewidth=1.5, linestyle='--'
        )
        
        # Add group label at center of ellipse
        center_x = np.mean(points_x[mask])
        center_y = np.mean(points_y[mask])
        
        # Adjust text position for better legibility
        if group_name == "Alien":
            text_x, text_y = center_x + 0.5, center_y - 0.5
        elif group_name == "Zombie":
            text_x, text_y = center_x + 0.5, center_y
        else:
            text_x, text_y = center_x, center_y
            
        ax1.text(text_x, text_y, group_name, fontsize=8, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Add colorbar
cbar = plt.colorbar(sc, ax=ax1, pad=0.01)
cbar.set_label('Price (ETH, log scale)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Format axes
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.grid(linestyle='--', alpha=0.3)
ax1.set_title('(a) Dimensionality Reduction Visualization', fontsize=11, pad=10)

# Add variance explained annotation
ax1.text(0.05, 0.95, "PC1: 18.7% variance\nPC2: 12.3% variance", 
         transform=ax1.transAxes, fontsize=8, va='top',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# Improve axes appearance
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Subplot (b): Feature Importance for Top Visual Attributes ---
ax2 = fig.add_subplot(gs[0, 1])

# Define visual attributes and their importance scores
attributes = [
    "Skin Tone",
    "Accessories",
    "Background Type",
    "Hair Style",
    "Facial Expression",
    "Clothing",
    "Hat Type",
    "Glasses",
    "Beard",
    "Earrings"
]

# Feature importance scores and standard errors
importance_scores = [0.683, 0.612, 0.587, 0.542, 0.501, 0.478, 0.453, 0.432, 0.401, 0.387]
std_errors = [0.024, 0.027, 0.025, 0.028, 0.026, 0.029, 0.031, 0.027, 0.030, 0.028]

# Sort data by importance score
sorted_indices = np.argsort(importance_scores)
sorted_attributes = [attributes[i] for i in sorted_indices]
sorted_importance = [importance_scores[i] for i in sorted_indices]
sorted_errors = [std_errors[i] for i in sorted_indices]

# Create horizontal bar chart
y_pos = np.arange(len(sorted_attributes))
bars = ax2.barh(y_pos, sorted_importance, xerr=sorted_errors, 
                align='center', color=palette[0], alpha=0.7, height=0.6,
                error_kw=dict(ecolor='black', lw=1, capsize=3, capthick=1))

# Format axes
ax2.set_yticks(y_pos)
ax2.set_yticklabels(sorted_attributes)
ax2.invert_yaxis()  # Highest importance at the top
ax2.set_xlabel('Correlation with Price')
ax2.grid(axis='x', linestyle='--', alpha=0.3)
ax2.set_xlim(0, 0.8)
ax2.set_title('(b) Feature Importance of Visual Attributes', fontsize=11, pad=10)

# Add importance scores as text
for i, v in enumerate(sorted_importance):
    ax2.text(v + 0.03, i, f"{v:.3f}", va='center', fontsize=8)

# Improve axes appearance
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Subplot (c): Clustering Results Revealing NFT Visual Styles ---
ax3 = fig.add_subplot(gs[1, 0])

# Generate sample data for clusters
np.random.seed(43)
n_clusters = 8
n_points_per_cluster = 200

# Create coordinates for cluster centers
cluster_centers = [
    (-5, 4),    # Cluster 1 (Aliens)
    (-3, 0),    # Cluster 2 (Apes)
    (-1, -4),   # Cluster 3 (Zombies)
    (1, 1),     # Cluster 4 (Female 1)
    (3, -2),    # Cluster 5 (Female 2)
    (4, 2),     # Cluster 6 (Male 1)
    (0, -2),    # Cluster 7 (Male 2)
    (2, 4)      # Cluster 8 (Male 3)
]

cluster_stds = [0.7, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 0.9]
cluster_prices = [1500, 800, 600, 25, 30, 22, 20, 35]
cluster_names = ["Alien", "Ape", "Zombie", "Female Type 1", "Female Type 2", 
                 "Male Type 1", "Male Type 2", "Male Type 3"]

# Generate points
all_x, all_y, all_clusters, all_prices = [], [], [], []
for i, (center, std, price) in enumerate(zip(cluster_centers, cluster_stds, cluster_prices)):
    # Scale number of points based on relative rarity
    n_points = n_points_per_cluster if i >= 3 else int(n_points_per_cluster * (0.2 if i == 0 else 0.4))
    
    x = np.random.normal(center[0], std, n_points)
    y = np.random.normal(center[1], std, n_points)
    
    all_x.extend(x)
    all_y.extend(y)
    all_clusters.extend([i] * n_points)
    
    # Add some price variation
    price_variation = price * 0.3
    p = np.random.normal(price, price_variation, n_points)
    all_prices.extend(p)

# Create scatter plot with cluster coloring
for i in range(n_clusters):
    mask = np.array(all_clusters) == i
    ax3.scatter(np.array(all_x)[mask], np.array(all_y)[mask], 
                color=palette[i % len(palette)], alpha=0.5, s=20, label=cluster_names[i])

# Add cluster centers
for i, center in enumerate(cluster_centers):
    ax3.scatter(center[0], center[1], s=100, marker='*', 
                color=palette[i % len(palette)], edgecolor='black', linewidth=1,
                alpha=0.9, zorder=10)

# Format axes
ax3.set_xlabel('Feature Dimension 1')
ax3.set_ylabel('Feature Dimension 2')
ax3.grid(linestyle='--', alpha=0.3)
ax3.set_title('(c) NFT Visual Style Clusters', fontsize=11, pad=10)

# Add silhouette score annotation
ax3.text(0.05, 0.95, "Silhouette Score: 0.728", transform=ax3.transAxes,
         fontsize=9, va='top',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# Add legend with two columns
legend = ax3.legend(fontsize=8, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True)

# Improve axes appearance
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# --- Subplot (d): Correlation between Visual Features and Pricing ---
ax4 = fig.add_subplot(gs[1, 1])

# Generate sample data for feature-price correlation
np.random.seed(44)
n_points = 80

# Generate feature values (feature #17 from the paper)
feature_values = np.random.normal(0, 1, n_points)

# Generate prices correlated with feature
correlation = 0.683  # From the paper
z = np.random.normal(0, np.sqrt(1 - correlation**2), n_points)
prices = correlation * feature_values + z
prices = np.exp(prices + 3)  # Transform to price-like values

# Create scatter plot
scatter = ax4.scatter(feature_values, prices, c=palette[0], alpha=0.7, s=40)

# Fit regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(feature_values, np.log(prices))
x_line = np.linspace(min(feature_values), max(feature_values), 100)
y_line = np.exp(slope * x_line + intercept)

# Add regression line
ax4.plot(x_line, y_line, color=palette[1], linewidth=2, alpha=0.8)

# Add prediction intervals
def prediction_band(x, y, x_new, alpha=0.05):
    """Calculate prediction bands for linear regression."""
    n = len(x)
    x_mean = np.mean(x)
    y_log = np.log(y)
    
    # Compute linear regression on log-transformed prices
    slope, intercept, _, _, std_err = stats.linregress(x, y_log)
    
    # Compute residuals and standard error
    y_pred = slope * x + intercept
    residuals = y_log - y_pred
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    # Compute prediction interval
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    y_pred_new = slope * x_new + intercept
    
    # Prediction interval factors
    se_fit = s_err * np.sqrt(1 + 1/n + ((x_new - x_mean)**2) / np.sum((x - x_mean)**2))
    
    # Compute upper and lower prediction intervals
    lower = y_pred_new - t_val * se_fit
    upper = y_pred_new + t_val * se_fit
    
    return np.exp(lower), np.exp(upper)

# Add 95% prediction bands
lower_band, upper_band = prediction_band(feature_values, prices, x_line)
ax4.fill_between(x_line, lower_band, upper_band, alpha=0.2, color=palette[1], 
                 label='95% Prediction Interval')

# Format axes with log scale for prices
ax4.set_yscale('log')
ax4.set_xlabel('Visual Feature #17 Value')
ax4.set_ylabel('Price (ETH)')
ax4.grid(linestyle='--', alpha=0.3)
ax4.set_title('(d) Visual Feature-Price Correlation', fontsize=11, pad=10)

# Add correlation coefficient annotation
ax4.text(0.05, 0.95, f"Correlation: {correlation:.3f}", transform=ax4.transAxes,
         fontsize=9, va='top',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# Improve axes appearance
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Save the figure
plt.tight_layout()
plt.savefig('visual_analysis.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

print("Figure 'visual_analysis.pdf' has been generated successfully.")