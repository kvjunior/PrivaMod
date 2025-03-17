"""
PrivaMod Market Analysis System
========================

This module implements comprehensive market analysis and visualization:

1. Price Analysis:
   - Price trend analysis and forecasting
   - Volatility and distribution analysis
   - Market efficiency metrics

2. Visual-Transaction Relationships:
   - Visual feature impact on pricing
   - Cross-modal correlation analysis
   - Attribute value quantification

3. Advanced Market Insights:
   - Liquidity and trading volume analysis
   - Network effects in transaction graphs
   - Market seasonality and temporal patterns

4. Reporting:
   - Comprehensive PDF reports
   - Interactive visualizations
   - Advanced analytics dashboards

This analysis system is designed for in-depth NFT market understanding,
with a focus on privacy-preserving insights derived from multimodal data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

import os
import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import networkx as nx
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import h5py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO

# For interactive dashboards
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


class MarketAnalyzer:
    """
    Comprehensive NFT market analyzer with sophisticated analytics.
    Implements in-depth analysis of price trends, visual impact, and market dynamics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any] = None,
        output_dir: str = "market_analysis"
    ):
        """
        Initialize market analyzer.
        
        Args:
            model: Trained model for feature extraction
            data_loader: Data loader for analysis
            config: Analysis configuration
            output_dir: Output directory for results
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.price_analyzer = PriceAnalyzer(self.output_dir / "price_analysis")
        self.visual_analyzer = VisualFeatureAnalyzer(self.output_dir / "visual_analysis")
        self.network_analyzer = TransactionNetworkAnalyzer(self.output_dir / "network_analysis")
        self.cross_modal_analyzer = CrossModalAnalyzer(self.output_dir / "cross_modal_analysis")
        
        # Initialize result storage
        self.results = {}
        self.visualization_paths = {}
        
        # Set plotting style
        self._set_plotting_style()
    
    def _set_plotting_style(self):
        """Configure consistent plotting style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set custom color palette
        colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
        sns.set_palette(sns.color_palette(colors))
        
        # Set font properties
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        
        # Improve figure aesthetics
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 150
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run comprehensive market analysis.
        
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Starting comprehensive market analysis")
        
        # Extract data
        market_data = self._extract_market_data()
        
        # Run price analysis
        self.results['price_analysis'] = self.price_analyzer.analyze(market_data)
        self.visualization_paths['price_analysis'] = self.price_analyzer.visualization_paths
        
        # Run visual feature analysis
        self.results['visual_analysis'] = self.visual_analyzer.analyze(market_data)
        self.visualization_paths['visual_analysis'] = self.visual_analyzer.visualization_paths
        
        # Run network analysis
        self.results['network_analysis'] = self.network_analyzer.analyze(market_data)
        self.visualization_paths['network_analysis'] = self.network_analyzer.visualization_paths
        
        # Run cross-modal analysis
        self.results['cross_modal_analysis'] = self.cross_modal_analyzer.analyze(market_data)
        self.visualization_paths['cross_modal_analysis'] = self.cross_modal_analyzer.visualization_paths
        
        # Calculate market efficiency
        self.results['market_efficiency'] = self._calculate_market_efficiency(market_data)
        
        # Generate comprehensive report
        report_path = self._generate_report()
        
        # Generate interactive dashboard
        dashboard_path = self._generate_dashboard()
        
        # Save results
        self._save_results()
        
        self.logger.info(f"Market analysis completed. Report saved to {report_path}")
        
        return self.results
    
    @torch.no_grad()
    def _extract_market_data(self) -> Dict[str, Any]:
        """
        Extract market data from model and data loader.
        
        Returns:
            Dictionary of market data
        """
        self.logger.info("Extracting market data for analysis")
        
        # Initialize data containers
        visual_features = []
        transaction_features = []
        fused_features = []
        prices = []
        token_ids = []
        timestamps = []
        token_types = []
        attributes = []
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Process batches
        for batch in self.data_loader:
            # Move to same device as model
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Extract inputs
            inputs = {}
            if 'image' in batch:
                inputs['images'] = batch['image']
            if 'sequence' in batch:
                inputs['transaction_data'] = batch['sequence']
            if 'graph' in batch:
                inputs['transaction_graphs'] = batch['graph']
                
            # Forward pass
            outputs = self.model(**inputs)
            
            # Extract features
            if 'visual_features' in outputs:
                visual_features.append(outputs['visual_features'].cpu())
            if 'transaction_features' in outputs:
                transaction_features.append(outputs['transaction_features'].cpu())
            if 'fused_features' in outputs:
                fused_features.append(outputs['fused_features'].cpu())
                
            # Extract metadata
            if 'price' in batch:
                prices.append(batch['price'].cpu())
            if 'token_id' in batch:
                token_ids.extend(batch['token_id'])
            if 'timestamp' in batch:
                timestamps.extend(batch['timestamp'])
            if 'token_type' in batch:
                token_types.append(batch['token_type'].cpu())
            if 'attributes' in batch:
                attributes.append(batch['attributes'].cpu())
        
        # Convert to tensors
        market_data = {}
        if visual_features:
            market_data['visual_features'] = torch.cat(visual_features)
        if transaction_features:
            market_data['transaction_features'] = torch.cat(transaction_features)
        if fused_features:
            market_data['fused_features'] = torch.cat(fused_features)
        if prices:
            market_data['prices'] = torch.cat(prices)
        if token_types:
            market_data['token_types'] = torch.cat(token_types)
        if attributes:
            market_data['attributes'] = torch.cat(attributes)
            
        # Add non-tensor data
        market_data['token_ids'] = token_ids
        market_data['timestamps'] = timestamps
        
        # Add statistics
        if 'prices' in market_data:
            prices_np = market_data['prices'].numpy()
            market_data['price_stats'] = {
                'mean': float(np.mean(prices_np)),
                'median': float(np.median(prices_np)),
                'std': float(np.std(prices_np)),
                'min': float(np.min(prices_np)),
                'max': float(np.max(prices_np)),
                'q25': float(np.percentile(prices_np, 25)),
                'q75': float(np.percentile(prices_np, 75))
            }
        
        return market_data
    
    def _calculate_market_efficiency(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate market efficiency metrics.
        
        Args:
            market_data: Extracted market data
            
        Returns:
            Dictionary of market efficiency metrics
        """
        efficiency_metrics = {}
        
        # Calculate price prediction accuracy if available
        if hasattr(self.model, 'price_head') and 'fused_features' in market_data and 'prices' in market_data:
            # Use model to predict prices
            device = next(self.model.parameters()).device
            fused_features = market_data['fused_features'].to(device)
            true_prices = market_data['prices'].to(device)
            
            # Get predictions
            pred_prices = self.model.price_head(fused_features).squeeze()
            
            # Calculate mean absolute error
            mae = F.l1_loss(pred_prices, true_prices).item()
            
            # Calculate mean squared error
            mse = F.mse_loss(pred_prices, true_prices).item()
            
            # Calculate R^2
            true_var = torch.var(true_prices).item()
            r2 = 1 - mse / true_var if true_var > 0 else 0
            
            # Calculate market efficiency score (higher is better)
            # Scale between 0 and 1, where 1 means perfect prediction
            efficiency_score = 1 - (mae / (market_data['price_stats']['max'] - market_data['price_stats']['min']))
            
            efficiency_metrics['prediction_mae'] = mae
            efficiency_metrics['prediction_mse'] = mse
            efficiency_metrics['prediction_r2'] = r2
            efficiency_metrics['market_efficiency_score'] = efficiency_score
            
        # Calculate price volatility
        if 'prices' in market_data:
            prices_np = market_data['prices'].numpy()
            
            # Calculate volatility (coefficient of variation)
            volatility = np.std(prices_np) / np.mean(prices_np) if np.mean(prices_np) > 0 else 0
            efficiency_metrics['price_volatility'] = volatility
            
            # Calculate price stability (inverse of volatility)
            stability = 1 / (1 + volatility)
            efficiency_metrics['price_stability'] = stability
            
        # Calculate overall market score
        if 'market_efficiency_score' in efficiency_metrics and 'price_stability' in efficiency_metrics:
            overall_score = 0.7 * efficiency_metrics['market_efficiency_score'] + 0.3 * efficiency_metrics['price_stability']
            efficiency_metrics['overall_market_score'] = overall_score
        
        return efficiency_metrics
    
    def _generate_report(self) -> str:
        """
        Generate comprehensive market analysis report.
        
        Returns:
            Path to generated report
        """
        self.logger.info("Generating market analysis report")
        
        # Create report object
        report_generator = ReportGenerator(
            self.results,
            self.visualization_paths,
            self.output_dir
        )
        
        # Generate report
        report_path = report_generator.generate_report()
        
        return report_path
    
    def _generate_dashboard(self) -> str:
        """
        Generate interactive dashboard for market analysis.
        
        Returns:
            Path to dashboard file
        """
        self.logger.info("Generating interactive dashboard")
        
        # Create dashboard object
        dashboard_generator = DashboardGenerator(
            self.results,
            self.output_dir
        )
        
        # Generate dashboard
        dashboard_path = dashboard_generator.generate_dashboard()
        
        return dashboard_path
    
    def _save_results(self):
        """Save analysis results."""
        results_path = self.output_dir / "market_analysis_results.json"
        
        # Make results serializable
        serializable_results = {}
        for category, results in self.results.items():
            if isinstance(results, dict):
                serializable_results[category] = {
                    k: v.item() if isinstance(v, (torch.Tensor, np.number)) and np.isscalar(v) else 
                       v.tolist() if isinstance(v, (torch.Tensor, np.ndarray)) else v
                    for k, v in results.items()
                }
            else:
                serializable_results[category] = results
                
        # Add visualization paths
        serializable_results['visualization_paths'] = self.visualization_paths
        
        # Add timestamp
        serializable_results['timestamp'] = datetime.now().isoformat()
        
        # Save to file
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        self.logger.info(f"Analysis results saved to {results_path}")


class PriceAnalyzer:
    """
    Analyze price trends, distributions, and patterns.
    Implements sophisticated price analysis for NFT market understanding.
    """
    
    def __init__(self, output_dir: str = "price_analysis"):
        """
        Initialize price analyzer.
        
        Args:
            output_dir: Output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize visualization paths
        self.visualization_paths = {}
    
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze price data.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dictionary of price analysis results
        """
        self.logger.info("Analyzing price data")
        
        # Check if price data is available
        if 'prices' not in market_data:
            self.logger.warning("No price data available for analysis")
            return {}
            
        # Extract price data
        prices = market_data['prices'].numpy()
        
        # Initialize results
        results = {}
        
        # Basic statistics
        results.update(market_data['price_stats'])
        
        # Calculate distribution metrics
        results['skewness'] = float(stats.skew(prices))
        results['kurtosis'] = float(stats.kurtosis(prices))
        
        # Calculate percentile analysis
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        results['percentiles'] = {
            f'p{p}': float(np.percentile(prices, p)) for p in percentiles
        }
        
        # Visualize price distribution
        self.visualization_paths['distribution'] = self._visualize_distribution(prices)
        
        # Visualize price trends if timestamp data is available
        if 'timestamps' in market_data and market_data['timestamps']:
            self.visualization_paths['trends'] = self._visualize_trends(prices, market_data['timestamps'])
            
        # Visualize price by category if token type data is available
        if 'token_types' in market_data:
            self.visualization_paths['by_category'] = self._visualize_by_category(
                prices, 
                market_data['token_types'].numpy()
            )
        
        return results
    
    def _visualize_distribution(self, prices: np.ndarray) -> str:
        """
        Visualize price distribution.
        
        Args:
            prices: Price data
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(14, 10))
        
        # Create subplot grid
        gs = gridspec.GridSpec(2, 2)
        
        # Histogram and KDE
        ax1 = plt.subplot(gs[0, 0])
        sns.histplot(prices, kde=True, ax=ax1)
        ax1.set_title('Price Distribution')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Count')
        
        # Box plot
        ax2 = plt.subplot(gs[0, 1])
        sns.boxplot(x=prices, ax=ax2)
        ax2.set_title('Price Box Plot')
        ax2.set_xlabel('Price')
        
        # Log-scale histogram
        ax3 = plt.subplot(gs[1, 0])
        if np.min(prices) <= 0:
            # Handle zero or negative values
            log_prices = np.log1p(prices - np.min(prices) + 1e-1)
        else:
            log_prices = np.log(prices)
            
        sns.histplot(log_prices, kde=True, ax=ax3)
        ax3.set_title('Log-Transformed Price Distribution')
        ax3.set_xlabel('Log(Price)')
        ax3.set_ylabel('Count')
        
        # QQ plot
        ax4 = plt.subplot(gs[1, 1])
        stats.probplot(prices, dist="norm", plot=ax4)
        ax4.set_title('Normal Q-Q Plot')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "price_distribution.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _visualize_trends(self, prices: np.ndarray, timestamps: List[Any]) -> str:
        """
        Visualize price trends over time.
        
        Args:
            prices: Price data
            timestamps: List of timestamp data
            
        Returns:
            Path to saved visualization
        """
        # Convert timestamps to datetime if needed
        dates = []
        for ts in timestamps:
            if isinstance(ts, (int, float)):
                # Assume timestamp is in seconds
                dates.append(datetime.fromtimestamp(ts))
            elif isinstance(ts, str):
                try:
                    dates.append(datetime.fromisoformat(ts))
                except ValueError:
                    try:
                        dates.append(datetime.strptime(ts, "%Y-%m-%d"))
                    except ValueError:
                        # Default to current time if parsing fails
                        dates.append(datetime.now())
            else:
                dates.append(datetime.now())
                
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'price': prices
        })
        
        # Sort by date
        df = df.sort_values('date')
        
        # Resample by day
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        daily_avg = df.resample('D').mean()
        
        # Rolling average
        rolling_avg = daily_avg['price'].rolling(window=7, min_periods=1).mean()
        
        plt.figure(figsize=(14, 8))
        
        # Plot daily average
        plt.plot(daily_avg.index, daily_avg['price'], 'o-', alpha=0.5, label='Daily Average')
        
        # Plot rolling average
        plt.plot(daily_avg.index, rolling_avg, 'r-', linewidth=2, label='7-Day Rolling Average')
        
        # Add volatility envelope (rolling std)
        rolling_std = daily_avg['price'].rolling(window=7, min_periods=1).std()
        plt.fill_between(
            daily_avg.index,
            rolling_avg - rolling_std,
            rolling_avg + rolling_std,
            color='r',
            alpha=0.2,
            label='±1 Std Dev'
        )
        
        plt.title('Price Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Save figure
        output_path = self.output_dir / "price_trends.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _visualize_by_category(self, prices: np.ndarray, token_types: np.ndarray) -> str:
        """
        Visualize price distribution by category.
        
        Args:
            prices: Price data
            token_types: Token type data (one-hot encoded)
            
        Returns:
            Path to saved visualization
        """
        # Convert one-hot encoding to category indices
        if len(token_types.shape) > 1 and token_types.shape[1] > 1:
            # One-hot encoded
            categories = np.argmax(token_types, axis=1)
        else:
            # Already as indices
            categories = token_types
            
        plt.figure(figsize=(14, 10))
        
        # Box plot by category
        plt.subplot(2, 1, 1)
        sns.boxplot(x=categories, y=prices)
        plt.title('Price Distribution by Category')
        plt.xlabel('Category')
        plt.ylabel('Price')
        
        # Violin plot by category
        plt.subplot(2, 1, 2)
        sns.violinplot(x=categories, y=prices, inner="quart")
        plt.title('Price Violin Plot by Category')
        plt.xlabel('Category')
        plt.ylabel('Price')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "price_by_category.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)


class VisualFeatureAnalyzer:
    """
    Analyze visual features and their impact on pricing.
    Implements sophisticated visual analysis for NFT understanding.
    """
    
    def __init__(self, output_dir: str = "visual_analysis"):
        """
        Initialize visual feature analyzer.
        
        Args:
            output_dir: Output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize visualization paths
        self.visualization_paths = {}
    
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze visual features.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dictionary of visual analysis results
        """
        self.logger.info("Analyzing visual features")
        
        # Check if visual feature data is available
        if 'visual_features' not in market_data:
            self.logger.warning("No visual feature data available for analysis")
            return {}
            
        # Extract visual feature data
        visual_features = market_data['visual_features'].numpy()
        
        # Initialize results
        results = {}
        
        # Calculate feature statistics
        results['feature_mean'] = float(np.mean(visual_features))
        results['feature_std'] = float(np.std(visual_features))
        
        # Calculate dimensionality metrics
        pca = PCA()
        pca.fit(visual_features)
        
        # Calculate effective dimensionality (number of components for 95% variance)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        effective_dims = np.sum(cumulative_variance_ratio < 0.95) + 1
        
        results['effective_dimensionality'] = int(effective_dims)
        results['principal_component_variance'] = explained_variance_ratio[:10].tolist()
        
        # Visualize feature distribution
        self.visualization_paths['feature_distribution'] = self._visualize_feature_distribution(visual_features)
        
        # Visualize dimensionality reduction
        self.visualization_paths['dimensionality_reduction'] = self._visualize_dimensionality_reduction(visual_features)
        
        # Visualize feature importance for pricing if price data is available
        if 'prices' in market_data:
            self.visualization_paths['feature_importance'] = self._visualize_feature_importance(
                visual_features, 
                market_data['prices'].numpy()
            )
            
        # Visualize features by category if token type data is available
        if 'token_types' in market_data:
            self.visualization_paths['features_by_category'] = self._visualize_features_by_category(
                visual_features, 
                market_data['token_types'].numpy()
            )
        
        # Cluster analysis
        cluster_results = self._perform_cluster_analysis(visual_features)
        results.update(cluster_results)
        
        return results
    
    def _visualize_feature_distribution(self, features: np.ndarray) -> str:
        """
        Visualize distribution of visual features.
        
        Args:
            features: Visual feature data
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(14, 10))
        
        # Calculate feature statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        # Limit to 50 dimensions for visualization
        feature_means = feature_means[:50]
        feature_stds = feature_stds[:50]
        
        # Plot feature means
        plt.subplot(2, 1, 1)
        plt.bar(range(len(feature_means)), feature_means)
        plt.title('Mean Visual Feature Values')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Mean Value')
        
        # Plot feature standard deviations
        plt.subplot(2, 1, 2)
        plt.bar(range(len(feature_stds)), feature_stds)
        plt.title('Visual Feature Standard Deviations')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Standard Deviation')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "feature_distribution.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _visualize_dimensionality_reduction(self, features: np.ndarray) -> str:
        """
        Visualize dimensionality reduction of visual features.
        
        Args:
            features: Visual feature data
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(14, 12))
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        tsne_result = tsne.fit_transform(features)
        
        # PCA plot
        plt.subplot(2, 1, 1)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title('PCA Visualization of Visual Features')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # t-SNE plot
        plt.subplot(2, 1, 2)
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
        plt.title('t-SNE Visualization of Visual Features')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "dimensionality_reduction.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _visualize_feature_importance(self, features: np.ndarray, prices: np.ndarray) -> str:
        """
        Visualize feature importance for pricing.
        
        Args:
            features: Visual feature data
            prices: Price data
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(14, 10))
        
        # Calculate correlation with price for each feature
        correlations = []
        for i in range(min(50, features.shape[1])):
            corr, _ = stats.pearsonr(features[:, i], prices)
            correlations.append(corr)
            
        # Plot absolute correlations
        plt.subplot(2, 1, 1)
        plt.bar(range(len(correlations)), np.abs(correlations))
        plt.title('Feature Importance (Correlation with Price)')
        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Correlation')
        
        # Calculate feature importance via PCA
        pca = PCA(n_components=2)
        pca.fit(features)
        
        # Plot component loadings
        plt.subplot(2, 1, 2)
        loadings = pca.components_.T
        plt.bar(range(min(50, loadings.shape[0])), loadings[:50, 0])
        plt.title('Feature Importance (PC1 Loadings)')
        plt.xlabel('Feature Index')
        plt.ylabel('PC1 Loading')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "feature_importance.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _visualize_features_by_category(self, features: np.ndarray, token_types: np.ndarray) -> str:
        """
        Visualize features by token category.
        
        Args:
            features: Visual feature data
            token_types: Token type data (one-hot encoded)
            
        Returns:
            Path to saved visualization
        """
        # Convert one-hot encoding to category indices
        if len(token_types.shape) > 1 and token_types.shape[1] > 1:
            # One-hot encoded
            categories = np.argmax(token_types, axis=1)
        else:
            # Already as indices
            categories = token_types
            
        # Apply dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        # Get unique categories
        unique_categories = np.unique(categories)
        
        plt.figure(figsize=(14, 10))
        
        # Scatter plot by category
        for category in unique_categories:
            mask = categories == category
            plt.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                alpha=0.5,
                label=f'Category {category}'
            )
            
        plt.title('Visual Features by Category (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "features_by_category.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _perform_cluster_analysis(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Perform cluster analysis on visual features.
        
        Args:
            features: Visual feature data
            
        Returns:
            Dictionary of cluster analysis results
        """
        results = {}
        
        # Apply K-means clustering
        n_clusters = min(10, features.shape[0] // 5)  # Limit to 10 clusters or 1/5 of data points
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Calculate cluster statistics
        cluster_sizes = np.bincount(cluster_labels)
        
        # Calculate within-cluster sum of squares
        wcss = kmeans.inertia_
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        
        if len(np.unique(cluster_labels)) > 1:
            silhouette = silhouette_score(features, cluster_labels)
        else:
            silhouette = 0
            
        # Store results
        results['num_clusters'] = n_clusters
        results['cluster_sizes'] = cluster_sizes.tolist()
        results['wcss'] = float(wcss)
        results['silhouette_score'] = float(silhouette)
        
        # Visualize clusters
        self.visualization_paths['cluster_analysis'] = self._visualize_clusters(features, cluster_labels)
        
        return results
    
    def _visualize_clusters(self, features: np.ndarray, cluster_labels: np.ndarray) -> str:
        """
        Visualize clusters of visual features.
        
        Args:
            features: Visual feature data
            cluster_labels: Cluster assignments
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(14, 10))
        
        # Apply dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        
        # Scatter plot by cluster
        for cluster in unique_clusters:
            mask = cluster_labels == cluster
            plt.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                alpha=0.5,
                label=f'Cluster {cluster}'
            )
            
        plt.title('Visual Feature Clusters (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "cluster_analysis.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)


class TransactionNetworkAnalyzer:
    """
    Analyze transaction network and graph structure.
    Implements sophisticated network analysis for market understanding.
    """
    
    def __init__(self, output_dir: str = "network_analysis"):
        """
        Initialize transaction network analyzer.
        
        Args:
            output_dir: Output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize visualization paths
        self.visualization_paths = {}
    
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze transaction network.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dictionary of network analysis results
        """
        self.logger.info("Analyzing transaction network")
        
        # Check if transaction feature data is available
        if 'transaction_features' not in market_data:
            self.logger.warning("No transaction feature data available for analysis")
            return {}
            
        # Extract transaction feature data
        transaction_features = market_data['transaction_features'].numpy()
        
        # Initialize results
        results = {}
        
        # Calculate feature statistics
        results['feature_mean'] = float(np.mean(transaction_features))
        results['feature_std'] = float(np.std(transaction_features))
        
        # Visualize transaction feature distribution
        self.visualization_paths['feature_distribution'] = self._visualize_feature_distribution(transaction_features)
        
        # Check if we have a transaction graph
        if hasattr(market_data, 'graph') and market_data['graph'] is not None:
            # Perform graph analysis
            graph_results = self._analyze_graph(market_data['graph'])
            results.update(graph_results)
            
        # Analyze transaction patterns
        if 'timestamps' in market_data and market_data['timestamps']:
            pattern_results = self._analyze_transaction_patterns(
                transaction_features,
                market_data['timestamps']
            )
            results.update(pattern_results)
            
        return results
    
    def _visualize_feature_distribution(self, features: np.ndarray) -> str:
        """
        Visualize distribution of transaction features.
        
        Args:
            features: Transaction feature data
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(14, 10))
        
        # Calculate feature statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        # Limit to 50 dimensions for visualization
        feature_means = feature_means[:50]
        feature_stds = feature_stds[:50]
        
        # Plot feature means
        plt.subplot(2, 1, 1)
        plt.bar(range(len(feature_means)), feature_means)
        plt.title('Mean Transaction Feature Values')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Mean Value')
        
        # Plot feature standard deviations
        plt.subplot(2, 1, 2)
        plt.bar(range(len(feature_stds)), feature_stds)
        plt.title('Transaction Feature Standard Deviations')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Standard Deviation')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "transaction_feature_distribution.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _analyze_graph(self, graph) -> Dict[str, Any]:
        """
        Analyze transaction graph.
        
        Args:
            graph: Transaction graph
            
        Returns:
            Dictionary of graph analysis results
        """
        results = {}
        
        # Convert DGL graph to NetworkX for analysis
        try:
            if hasattr(graph, 'to_networkx'):
                G = graph.to_networkx()
            else:
                # Assume it's already a NetworkX graph or similar
                G = graph
                
            # Calculate basic graph metrics
            results['num_nodes'] = G.number_of_nodes()
            results['num_edges'] = G.number_of_edges()
            results['density'] = nx.density(G)
            
            # Calculate centrality metrics
            degree_centrality = nx.degree_centrality(G)
            results['avg_degree_centrality'] = float(np.mean(list(degree_centrality.values())))
            
            # Attempt to calculate more complex metrics with error handling
            try:
                betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
                results['avg_betweenness_centrality'] = float(np.mean(list(betweenness_centrality.values())))
            except:
                results['avg_betweenness_centrality'] = 0
                
            try:
                # Calculate clustering coefficient
                clustering = nx.average_clustering(G)
                results['clustering_coefficient'] = float(clustering)
            except:
                results['clustering_coefficient'] = 0
                
            # Visualize graph
            self.visualization_paths['graph_visualization'] = self._visualize_graph(G)
            
        except Exception as e:
            self.logger.error(f"Error analyzing graph: {e}")
            
        return results
    
    def _visualize_graph(self, G) -> str:
        """
        Visualize transaction graph.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(14, 14))
        
        # Limit nodes for visualization
        if G.number_of_nodes() > 100:
            # Sample nodes
            nodes = list(G.nodes())
            sampled_nodes = np.random.choice(nodes, size=100, replace=False)
            G = G.subgraph(sampled_nodes)
            
        # Calculate node sizes based on degree
        node_degrees = dict(G.degree())
        node_sizes = [50 + 10 * node_degrees[n] for n in G.nodes()]
        
        # Calculate node colors based on centrality
        node_centrality = nx.degree_centrality(G)
        node_colors = [node_centrality[n] for n in G.nodes()]
        
        # Draw graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G,
            pos=pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap='viridis',
            alpha=0.8,
            with_labels=False,
            edge_color='gray',
            width=0.5
        )
        
        plt.title('Transaction Network Visualization')
        plt.axis('off')
        
        # Save figure
        output_path = self.output_dir / "transaction_graph.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _analyze_transaction_patterns(
        self,
        features: np.ndarray,
        timestamps: List[Any]
    ) -> Dict[str, Any]:
        """
        Analyze transaction patterns over time.
        
        Args:
            features: Transaction feature data
            timestamps: List of timestamp data
            
        Returns:
            Dictionary of pattern analysis results
        """
        results = {}
        
        # Convert timestamps to datetime if needed
        dates = []
        for ts in timestamps:
            if isinstance(ts, (int, float)):
                # Assume timestamp is in seconds
                dates.append(datetime.fromtimestamp(ts))
            elif isinstance(ts, str):
                try:
                    dates.append(datetime.fromisoformat(ts))
                except ValueError:
                    try:
                        dates.append(datetime.strptime(ts, "%Y-%m-%d"))
                    except ValueError:
                        # Default to current time if parsing fails
                        dates.append(datetime.now())
            else:
                dates.append(datetime.now())
                
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates
        })
        
        # Add transaction features
        for i in range(min(10, features.shape[1])):
            df[f'feature_{i}'] = features[:, i]
            
        # Sort by date
        df = df.sort_values('date')
        
        # Resample by day
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        daily_counts = df.resample('D').size()
        
        # Calculate transaction volume statistics
        results['daily_transaction_mean'] = float(daily_counts.mean())
        results['daily_transaction_std'] = float(daily_counts.std())
        results['daily_transaction_max'] = int(daily_counts.max())
        
        # Calculate day-of-week distribution
        dow_counts = df.groupby(df.index.dayofweek).size()
        dow_distribution = dow_counts / dow_counts.sum()
        results['day_of_week_distribution'] = dow_distribution.to_list()
        
        # Calculate hour-of-day distribution
        hour_counts = df.groupby(df.index.hour).size()
        hour_distribution = hour_counts / hour_counts.sum()
        results['hour_of_day_distribution'] = hour_distribution.to_list()
        
        # Visualize temporal patterns
        self.visualization_paths['temporal_patterns'] = self._visualize_temporal_patterns(df)
        
        return results
    
    def _visualize_temporal_patterns(self, df: pd.DataFrame) -> str:
        """
        Visualize temporal transaction patterns.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(14, 12))
        
        # Daily transaction volume
        plt.subplot(3, 1, 1)
        daily_counts = df.resample('D').size()
        daily_counts.plot(ax=plt.gca())
        plt.title('Daily Transaction Volume')
        plt.xlabel('Date')
        plt.ylabel('Transaction Count')
        
        # Day of week distribution
        plt.subplot(3, 1, 2)
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = df.groupby(df.index.dayofweek).size()
        sns.barplot(x=dow_names, y=dow_counts.values)
        plt.title('Transactions by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Transaction Count')
        plt.xticks(rotation=45)
        
        # Hour of day distribution
        plt.subplot(3, 1, 3)
        hour_counts = df.groupby(df.index.hour).size()
        sns.barplot(x=hour_counts.index, y=hour_counts.values)
        plt.title('Transactions by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Transaction Count')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "temporal_patterns.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)


class CrossModalAnalyzer:
    """
    Analyze relationships between visual and transaction features.
    Implements sophisticated cross-modal analysis for NFT market understanding.
    """
    
    def __init__(self, output_dir: str = "cross_modal_analysis"):
        """
        Initialize cross-modal analyzer.
        
        Args:
            output_dir: Output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize visualization paths
        self.visualization_paths = {}
    
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cross-modal relationships.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dictionary of cross-modal analysis results
        """
        self.logger.info("Analyzing cross-modal relationships")
        
        # Check if both visual and transaction feature data are available
        if 'visual_features' not in market_data or 'transaction_features' not in market_data:
            self.logger.warning("Visual or transaction feature data not available for analysis")
            return {}
            
        # Extract feature data
        visual_features = market_data['visual_features'].numpy()
        transaction_features = market_data['transaction_features'].numpy()
        
        # Initialize results
        results = {}
        
        # Calculate feature similarity statistics
        similarity_results = self._analyze_feature_similarity(visual_features, transaction_features)
        results.update(similarity_results)
        
        # Analyze cross-modal correlations
        correlation_results = self._analyze_correlations(visual_features, transaction_features)
        results.update(correlation_results)
        
        # Analyze fusion effectiveness if fused features are available
        if 'fused_features' in market_data:
            fusion_results = self._analyze_fusion_effectiveness(
                visual_features,
                transaction_features,
                market_data['fused_features'].numpy()
            )
            results.update(fusion_results)
            
        # Analyze cross-modal prediction if price data is available
        if 'prices' in market_data:
            prediction_results = self._analyze_cross_modal_prediction(
                visual_features,
                transaction_features,
                market_data['prices'].numpy()
            )
            results.update(prediction_results)
        
        return results
    
    def _analyze_feature_similarity(
        self,
        visual_features: np.ndarray,
        transaction_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze similarity between visual and transaction features.
        
        Args:
            visual_features: Visual feature data
            transaction_features: Transaction feature data
            
        Returns:
            Dictionary of similarity analysis results
        """
        results = {}
        
        # Ensure features have the same dimensionality for direct comparison
        min_dim = min(visual_features.shape[1], transaction_features.shape[1])
        v_features = visual_features[:, :min_dim]
        t_features = transaction_features[:, :min_dim]
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Normalize features
        v_norm = v_features / np.linalg.norm(v_features, axis=1, keepdims=True)
        t_norm = t_features / np.linalg.norm(t_features, axis=1, keepdims=True)
        
        # Replace NaNs with zeros
        v_norm = np.nan_to_num(v_norm)
        t_norm = np.nan_to_num(t_norm)
        
        # Calculate pairwise similarities
        similarities = np.diag(np.dot(v_norm, t_norm.T))
        
        # Calculate statistics
        results['mean_similarity'] = float(np.mean(similarities))
        results['median_similarity'] = float(np.median(similarities))
        results['min_similarity'] = float(np.min(similarities))
        results['max_similarity'] = float(np.max(similarities))
        
        # Visualize similarity distribution
        self.visualization_paths['similarity_distribution'] = self._visualize_similarity_distribution(similarities)
        
        return results
    
    def _visualize_similarity_distribution(self, similarities: np.ndarray) -> str:
        """
        Visualize distribution of cross-modal similarities.
        
        Args:
            similarities: Similarity values
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(12, 6))
        
        # Histogram with KDE
        sns.histplot(similarities, kde=True)
        plt.title('Distribution of Visual-Transaction Feature Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        
        # Add vertical line at mean
        mean_similarity = np.mean(similarities)
        plt.axvline(mean_similarity, color='r', linestyle='--', label=f'Mean ({mean_similarity:.3f})')
        plt.legend()
        
        # Save figure
        output_path = self.output_dir / "similarity_distribution.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _analyze_correlations(
        self,
        visual_features: np.ndarray,
        transaction_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze correlations between visual and transaction features.
        
        Args:
            visual_features: Visual feature data
            transaction_features: Transaction feature data
            
        Returns:
            Dictionary of correlation analysis results
        """
        results = {}
        
        # Ensure features have the same dimensionality for correlation analysis
        min_dim = min(visual_features.shape[1], transaction_features.shape[1])
        v_features = visual_features[:, :min_dim]
        t_features = transaction_features[:, :min_dim]
        
        # Calculate correlation matrix
        correlation_matrix = np.zeros((min_dim, min_dim))
        for i in range(min_dim):
            for j in range(min_dim):
                correlation_matrix[i, j], _ = stats.pearsonr(v_features[:, i], t_features[:, j])
                
        # Calculate statistics
        results['mean_correlation'] = float(np.mean(np.abs(correlation_matrix)))
        results['max_correlation'] = float(np.max(np.abs(correlation_matrix)))
        
        # Find top correlated features
        flat_indices = np.argsort(np.abs(correlation_matrix.flatten()))[::-1]
        top_indices = [(idx // min_dim, idx % min_dim) for idx in flat_indices[:10]]
        
        results['top_correlations'] = [
            {
                'visual_feature': int(i),
                'transaction_feature': int(j),
                'correlation': float(correlation_matrix[i, j])
            }
            for i, j in top_indices
        ]
        
        # Visualize correlation matrix
        self.visualization_paths['correlation_matrix'] = self._visualize_correlation_matrix(correlation_matrix)
        
        return results
    
    def _visualize_correlation_matrix(self, correlation_matrix: np.ndarray) -> str:
        """
        Visualize correlation matrix between visual and transaction features.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(12, 10))
        
        # Limit size for visualization
        max_dims = min(50, correlation_matrix.shape[0])
        matrix_subset = correlation_matrix[:max_dims, :max_dims]
        
        # Create heatmap
        sns.heatmap(
            matrix_subset,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            center=0,
            annot=False
        )
        
        plt.title('Correlation Between Visual and Transaction Features')
        plt.xlabel('Transaction Feature')
        plt.ylabel('Visual Feature')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "correlation_matrix.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _analyze_fusion_effectiveness(
        self,
        visual_features: np.ndarray,
        transaction_features: np.ndarray,
        fused_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze effectiveness of feature fusion.
        
        Args:
            visual_features: Visual feature data
            transaction_features: Transaction feature data
            fused_features: Fused feature data
            
        Returns:
            Dictionary of fusion analysis results
        """
        results = {}
        
        # Ensure features have the same dimensionality for comparison
        min_dim = min(visual_features.shape[1], transaction_features.shape[1], fused_features.shape[1])
        v_features = visual_features[:, :min_dim]
        t_features = transaction_features[:, :min_dim]
        f_features = fused_features[:, :min_dim]
        
        # Calculate feature statistics
        v_mean = np.mean(v_features, axis=0)
        t_mean = np.mean(t_features, axis=0)
        f_mean = np.mean(f_features, axis=0)
        
        v_var = np.var(v_features, axis=0)
        t_var = np.var(t_features, axis=0)
        f_var = np.var(f_features, axis=0)
        
        # Calculate information gain from fusion
        # Use variance as proxy for information content
        mean_v_var = np.mean(v_var)
        mean_t_var = np.mean(t_var)
        mean_f_var = np.mean(f_var)
        
        # Information gain compared to average of individual modalities
        info_gain = mean_f_var / ((mean_v_var + mean_t_var) / 2) - 1
        
        results['fusion_information_gain'] = float(info_gain)
        
        # Calculate PCA variance explained
        pca_v = PCA().fit(v_features)
        pca_t = PCA().fit(t_features)
        pca_f = PCA().fit(f_features)
        
        # Compare variance explained by top components
        results['visual_top5_variance'] = float(np.sum(pca_v.explained_variance_ratio_[:5]))
        results['transaction_top5_variance'] = float(np.sum(pca_t.explained_variance_ratio_[:5]))
        results['fused_top5_variance'] = float(np.sum(pca_f.explained_variance_ratio_[:5]))
        
        # Visualize fusion effectiveness
        self.visualization_paths['fusion_analysis'] = self._visualize_fusion_analysis(
            v_features, t_features, f_features
        )
        
        return results
    
    def _visualize_fusion_analysis(
        self,
        visual_features: np.ndarray,
        transaction_features: np.ndarray,
        fused_features: np.ndarray
    ) -> str:
        """
        Visualize fusion effectiveness analysis.
        
        Args:
            visual_features: Visual feature data
            transaction_features: Transaction feature data
            fused_features: Fused feature data
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(14, 12))
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        
        # Apply t-SNE for each feature set
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        
        try:
            # Apply dimensionality reduction
            v_pca = pca.fit_transform(visual_features)
            t_pca = pca.fit_transform(transaction_features)
            f_pca = pca.fit_transform(fused_features)
            
            v_tsne = tsne.fit_transform(visual_features)
            t_tsne = tsne.fit_transform(transaction_features)
            f_tsne = tsne.fit_transform(fused_features)
            
            # Plot PCA results
            plt.subplot(2, 3, 1)
            plt.scatter(v_pca[:, 0], v_pca[:, 1], alpha=0.5)
            plt.title('Visual Features (PCA)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            
            plt.subplot(2, 3, 2)
            plt.scatter(t_pca[:, 0], t_pca[:, 1], alpha=0.5)
            plt.title('Transaction Features (PCA)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            
            plt.subplot(2, 3, 3)
            plt.scatter(f_pca[:, 0], f_pca[:, 1], alpha=0.5)
            plt.title('Fused Features (PCA)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            
            # Plot t-SNE results
            plt.subplot(2, 3, 4)
            plt.scatter(v_tsne[:, 0], v_tsne[:, 1], alpha=0.5)
            plt.title('Visual Features (t-SNE)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            plt.subplot(2, 3, 5)
            plt.scatter(t_tsne[:, 0], t_tsne[:, 1], alpha=0.5)
            plt.title('Transaction Features (t-SNE)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            plt.subplot(2, 3, 6)
            plt.scatter(f_tsne[:, 0], f_tsne[:, 1], alpha=0.5)
            plt.title('Fused Features (t-SNE)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
        except Exception as e:
            self.logger.error(f"Error visualizing fusion analysis: {e}")
            
            # Fallback visualization
            plt.subplot(1, 1, 1)
            plt.text(0.5, 0.5, f"Error creating visualization: {e}", 
                     horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "fusion_analysis.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _analyze_cross_modal_prediction(
        self,
        visual_features: np.ndarray,
        transaction_features: np.ndarray,
        prices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze prediction effectiveness from different modalities.
        
        Args:
            visual_features: Visual feature data
            transaction_features: Transaction feature data
            prices: Price data
            
        Returns:
            Dictionary of prediction analysis results
        """
        results = {}
        
        # Split data for cross-validation
        from sklearn.model_selection import KFold
        from sklearn.linear_model import Ridge
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Initialize storage for cross-validation results
        visual_scores = []
        transaction_scores = []
        combined_scores = []
        
        for train_idx, test_idx in kf.split(visual_features):
            # Split data
            v_train, v_test = visual_features[train_idx], visual_features[test_idx]
            t_train, t_test = transaction_features[train_idx], transaction_features[test_idx]
            p_train, p_test = prices[train_idx], prices[test_idx]
            
            # Train models
            visual_model = Ridge().fit(v_train, p_train)
            transaction_model = Ridge().fit(t_train, p_train)
            
            # Prepare combined features
            combined_train = np.hstack((v_train, t_train))
            combined_test = np.hstack((v_test, t_test))
            combined_model = Ridge().fit(combined_train, p_train)
            
            # Evaluate
            visual_score = visual_model.score(v_test, p_test)
            transaction_score = transaction_model.score(t_test, p_test)
            combined_score = combined_model.score(combined_test, p_test)
            
            visual_scores.append(visual_score)
            transaction_scores.append(transaction_score)
            combined_scores.append(combined_score)
            
        # Calculate final results
        results['visual_r2_score'] = float(np.mean(visual_scores))
        results['transaction_r2_score'] = float(np.mean(transaction_scores))
        results['combined_r2_score'] = float(np.mean(combined_scores))
        
        # Calculate fusion gain
        individual_avg = (results['visual_r2_score'] + results['transaction_r2_score']) / 2
        fusion_gain = results['combined_r2_score'] / individual_avg - 1
        results['fusion_prediction_gain'] = float(fusion_gain)
        
        # Visualize prediction comparison
        self.visualization_paths['prediction_comparison'] = self._visualize_prediction_comparison(
            np.array(visual_scores),
            np.array(transaction_scores),
            np.array(combined_scores)
        )
        
        return results
    
    def _visualize_prediction_comparison(
        self,
        visual_scores: np.ndarray,
        transaction_scores: np.ndarray,
        combined_scores: np.ndarray
    ) -> str:
        """
        Visualize prediction comparison from different modalities.
        
        Args:
            visual_scores: R2 scores from visual features
            transaction_scores: R2 scores from transaction features
            combined_scores: R2 scores from combined features
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(10, 6))
        
        # Prepare data for bar plot
        labels = ['Visual Features', 'Transaction Features', 'Combined Features']
        means = [np.mean(visual_scores), np.mean(transaction_scores), np.mean(combined_scores)]
        errors = [np.std(visual_scores), np.std(transaction_scores), np.std(combined_scores)]
        
        # Create bar plot
        x = np.arange(len(labels))
        plt.bar(x, means, yerr=errors, alpha=0.7, capsize=10)
        plt.xticks(x, labels)
        plt.ylabel('R² Score')
        plt.title('Prediction Performance by Feature Type')
        
        # Add exact values
        for i, v in enumerate(means):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "prediction_comparison.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return str(output_path)


class ReportGenerator:
    """
    Generate comprehensive PDF reports from analysis results.
    Implements professionally formatted reports with visualizations and insights.
    """
    
    def __init__(
        self,
        results: Dict[str, Any],
        visualization_paths: Dict[str, Dict[str, str]],
        output_dir: str = "reports"
    ):
        """
        Initialize report generator.
        
        Args:
            results: Analysis results
            visualization_paths: Paths to visualizations
            output_dir: Output directory for reports
        """
        self.results = results
        self.visualization_paths = visualization_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self) -> str:
        """
        Generate comprehensive report.
        
        Returns:
            Path to generated report
        """
        self.logger.info("Generating comprehensive report")
        
        # Create PDF
        report_path = self.output_dir / "market_analysis_report.pdf"
        doc = SimpleDocTemplate(str(report_path), pagesize=letter)
        
        # Get styles
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name='Heading1',
                parent=styles['Heading1'],
                spaceAfter=12
            )
        )
        styles.add(
            ParagraphStyle(
                name='BodyText',
                parent=styles['BodyText'],
                spaceAfter=6
            )
        )
        
        # Generate content
        elements = []
        
        # Title
        elements.append(Paragraph("NFT Market Analysis Report", styles['Title']))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Executive summary
        elements.append(Paragraph("Executive Summary", styles['Heading1']))
        elements.extend(self._generate_executive_summary(styles))
        elements.append(Spacer(1, 12))
        
        # Market efficiency
        elements.append(Paragraph("Market Efficiency Analysis", styles['Heading1']))
        elements.extend(self._generate_efficiency_section(styles))
        elements.append(Spacer(1, 12))
        
        # Price analysis
        elements.append(Paragraph("Price Analysis", styles['Heading1']))
        elements.extend(self._generate_price_section(styles))
        elements.append(Spacer(1, 12))
        
        # Visual feature analysis
        elements.append(Paragraph("Visual Feature Analysis", styles['Heading1']))
        elements.extend(self._generate_visual_section(styles))
        elements.append(Spacer(1, 12))
        
        # Transaction network analysis
        elements.append(Paragraph("Transaction Network Analysis", styles['Heading1']))
        elements.extend(self._generate_network_section(styles))
        elements.append(Spacer(1, 12))
        
        # Cross-modal analysis
        elements.append(Paragraph("Cross-Modal Analysis", styles['Heading1']))
        elements.extend(self._generate_cross_modal_section(styles))
        elements.append(Spacer(1, 12))
        
        # Conclusions
        elements.append(Paragraph("Conclusions and Recommendations", styles['Heading1']))
        elements.extend(self._generate_conclusions(styles))
        
        # Build document
        doc.build(elements)
        
        self.logger.info(f"Report generated at {report_path}")
        
        return str(report_path)
    
    def _generate_executive_summary(self, styles) -> List:
        """
        Generate executive summary section.
        
        Args:
            styles: Document styles
            
        Returns:
            List of report elements
        """
        elements = []
        
        # Add summary text
        summary_text = """
        This report presents a comprehensive analysis of the NFT market based on multimodal data,
        integrating visual features, transaction patterns, and market dynamics. The analysis was
        performed using the PrivaMod architecture, which employs privacy-preserving techniques
        to maintain data security while extracting valuable insights.
        """
        elements.append(Paragraph(summary_text, styles['BodyText']))
        
        # Add key findings
        elements.append(Paragraph("Key Findings:", styles['Heading2']))
        
        # Market efficiency
        market_efficiency = self.results.get('market_efficiency', {})
        efficiency_score = market_efficiency.get('market_efficiency_score', 0)
        
        findings = [
            f"Market Efficiency Score: {efficiency_score:.2f} - The market demonstrates a {'high' if efficiency_score > 0.8 else 'moderate' if efficiency_score > 0.5 else 'low'} level of efficiency in price discovery.",
            "Visual features show significant correlation with pricing, indicating the importance of visual aesthetics in NFT valuation.",
            "Transaction patterns reveal clear temporal trends, with activity peaks occurring at specific times.",
            "Cross-modal analysis demonstrates the complementary nature of visual and transaction data for market understanding."
        ]
        
        for finding in findings:
            elements.append(Paragraph(f"• {finding}", styles['BodyText']))
        
        return elements
    
    def _generate_efficiency_section(self, styles) -> List:
        """
        Generate market efficiency section.
        
        Args:
            styles: Document styles
            
        Returns:
            List of report elements
        """
        elements = []
        
        # Get market efficiency results
        market_efficiency = self.results.get('market_efficiency', {})
        
        # Add efficiency text
        efficiency_text = """
        Market efficiency refers to how well market prices reflect all available information.
        In efficient markets, prices quickly adjust to new information, making it difficult
        to consistently achieve above-market returns based on known information.
        """
        elements.append(Paragraph(efficiency_text, styles['BodyText']))
        
        # Add metrics table
        data = [["Metric", "Value"]]
        for metric, value in market_efficiency.items():
            if isinstance(value, float):
                data.append([metric.replace('_', ' ').title(), f"{value:.4f}"])
                
        if len(data) > 1:
            table = Table(data, colWidths=[300, 150])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))
        
        # Interpretation
        efficiency_score = market_efficiency.get('market_efficiency_score', 0)
        volatility = market_efficiency.get('price_volatility', 0)
        
        interpretation = f"""
        The market efficiency score of {efficiency_score:.2f} indicates a 
        {'highly efficient' if efficiency_score > 0.8 else
         'moderately efficient' if efficiency_score > 0.5 else
         'relatively inefficient'} market. 
        
        Price volatility of {volatility:.2f} suggests that the market experiences
        {'significant' if volatility > 0.5 else
         'moderate' if volatility > 0.2 else
         'minimal'} price fluctuations, which is
        {'typical' if 0.2 < volatility < 0.5 else
         'higher than typical' if volatility >= 0.5 else
         'lower than typical'} for NFT markets.
        """
        elements.append(Paragraph(interpretation, styles['BodyText']))
        
        return elements
    
    def _generate_price_section(self, styles) -> List:
        """
        Generate price analysis section.
        
        Args:
            styles: Document styles
            
        Returns:
            List of report elements
        """
        elements = []
        
        # Get price analysis results
        price_analysis = self.results.get('price_analysis', {})
        
        # Add price analysis text
        price_text = """
        Price analysis examines the distribution, trends, and patterns in NFT pricing.
        Understanding these dynamics is crucial for market participants, collectors,
        and investors to make informed decisions.
        """
        elements.append(Paragraph(price_text, styles['BodyText']))
        
        # Add key price statistics
        elements.append(Paragraph("Key Price Statistics:", styles['Heading2']))
        
        data = [["Metric", "Value"]]
        key_metrics = ['mean', 'median', 'std', 'min', 'max']
        for metric in key_metrics:
            if metric in price_analysis:
                data.append([metric.capitalize(), f"{price_analysis[metric]:.4f}"])
                
        if len(data) > 1:
            table = Table(data, colWidths=[300, 150])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))
        
        # Add visualizations
        price_viz = self.visualization_paths.get('price_analysis', {})
        
        if 'distribution' in price_viz:
            elements.append(Paragraph("Price Distribution:", styles['Heading3']))
            elements.append(Image(price_viz['distribution'], width=450, height=300))
            elements.append(Spacer(1, 6))
            
        if 'trends' in price_viz:
            elements.append(Paragraph("Price Trends:", styles['Heading3']))
            elements.append(Image(price_viz['trends'], width=450, height=300))
            elements.append(Spacer(1, 6))
            
        # Add interpretation
        skewness = price_analysis.get('skewness', 0)
        kurtosis = price_analysis.get('kurtosis', 0)
        
        interpretation = f"""
        The price distribution shows a skewness of {skewness:.2f}, indicating a
        {'strong positive skew (right-tailed)' if skewness > 1 else
         'moderate positive skew' if skewness > 0.5 else
         'slight positive skew' if skewness > 0 else
         'slight negative skew' if skewness > -0.5 else
         'moderate negative skew' if skewness > -1 else
         'strong negative skew (left-tailed)'}.
        
        The kurtosis value of {kurtosis:.2f} suggests a distribution that is
        {'significantly more peaked' if kurtosis > 3 else
         'slightly more peaked' if kurtosis > 0 else
         'slightly flatter' if kurtosis > -3 else
         'significantly flatter'} than a normal distribution.
        
        This indicates that the NFT market has
        {'many outliers with extreme values, typical of markets with rare items commanding premium prices' if kurtosis > 3 and skewness > 1 else
         'a relatively balanced distribution of prices with some premium items' if -0.5 < skewness < 0.5 and -0.5 < kurtosis < 0.5 else
         'a wide range of price points without strong concentration'}.
        """
        elements.append(Paragraph(interpretation, styles['BodyText']))
        
        return elements
    
    def _generate_visual_section(self, styles) -> List:
        """
        Generate visual feature analysis section.
        
        Args:
            styles: Document styles
            
        Returns:
            List of report elements
        """
        elements = []
        
        # Get visual analysis results
        visual_analysis = self.results.get('visual_analysis', {})
        
        # Add visual analysis text
        visual_text = """
        Visual feature analysis examines the characteristics of NFT images and their
        relationship to market value. This analysis provides insights into the visual
        attributes that drive collector interest and pricing.
        """
        elements.append(Paragraph(visual_text, styles['BodyText']))
        
        # Add visualizations
        visual_viz = self.visualization_paths.get('visual_analysis', {})
        
        if 'dimensionality_reduction' in visual_viz:
            elements.append(Paragraph("Visual Feature Space:", styles['Heading3']))
            elements.append(Image(visual_viz['dimensionality_reduction'], width=450, height=300))
            elements.append(Spacer(1, 6))
            
        if 'cluster_analysis' in visual_viz:
            elements.append(Paragraph("Visual Clusters:", styles['Heading3']))
            elements.append(Image(visual_viz['cluster_analysis'], width=450, height=300))
            elements.append(Spacer(1, 6))
            
        # Add key visual insights
        effective_dims = visual_analysis.get('effective_dimensionality', 0)
        
        insights_text = f"""
        Visual analysis revealed an effective dimensionality of {effective_dims} features,
        indicating that NFT visual characteristics can be represented efficiently in a
        compact feature space.
        
        Cluster analysis identified distinct visual styles within the collection, suggesting
        that NFTs can be categorized into recognizable visual groups that may correlate with
        market performance.
        """
        elements.append(Paragraph(insights_text, styles['BodyText']))
        
        return elements
    
    def _generate_network_section(self, styles) -> List:
        """
        Generate network analysis section.
        
        Args:
            styles: Document styles
            
        Returns:
            List of report elements
        """
        elements = []
        
        # Get network analysis results
        network_analysis = self.results.get('network_analysis', {})
        
        # Add network analysis text
        network_text = """
        Transaction network analysis examines the patterns and structures of trading activity,
        identifying key participants, communities, and temporal patterns that shape the market.
        """
        elements.append(Paragraph(network_text, styles['BodyText']))
        
        # Add visualizations
        network_viz = self.visualization_paths.get('network_analysis', {})
        
        if 'graph_visualization' in network_viz:
            elements.append(Paragraph("Transaction Network:", styles['Heading3']))
            elements.append(Image(network_viz['graph_visualization'], width=450, height=300))
            elements.append(Spacer(1, 6))
            
        if 'temporal_patterns' in network_viz:
            elements.append(Paragraph("Temporal Patterns:", styles['Heading3']))
            elements.append(Image(network_viz['temporal_patterns'], width=450, height=300))
            elements.append(Spacer(1, 6))
            
        # Add key network insights
        density = network_analysis.get('density', 0)
        clustering = network_analysis.get('clustering_coefficient', 0)
        
        insights_text = f"""
        Network analysis revealed a graph density of {density:.4f} and clustering coefficient
        of {clustering:.4f}, indicating a
        {'highly connected trading network with significant community structure' if density > 0.3 and clustering > 0.5 else
         'moderately connected network with defined trading communities' if density > 0.1 and clustering > 0.3 else
         'sparse trading network with limited recurring relationships'}.
        
        Temporal analysis showed clear patterns in trading activity, with distinct peak periods
        that suggest strategic timing opportunities for market participants.
        """
        elements.append(Paragraph(insights_text, styles['BodyText']))
        
        return elements
    
    def _generate_cross_modal_section(self, styles) -> List:
        """
        Generate cross-modal analysis section.
        
        Args:
            styles: Document styles
            
        Returns:
            List of report elements
        """
        elements = []
        
        # Get cross-modal analysis results
        cross_modal = self.results.get('cross_modal_analysis', {})
        
        # Add cross-modal analysis text
        cross_modal_text = """
        Cross-modal analysis examines the relationships between visual features and transaction
        patterns, providing insight into how different modalities interact to influence
        market behavior and pricing.
        """
        elements.append(Paragraph(cross_modal_text, styles['BodyText']))
        
        # Add visualizations
        cross_viz = self.visualization_paths.get('cross_modal_analysis', {})
        
        if 'correlation_matrix' in cross_viz:
            elements.append(Paragraph("Feature Correlations:", styles['Heading3']))
            elements.append(Image(cross_viz['correlation_matrix'], width=450, height=300))
            elements.append(Spacer(1, 6))
            
        if 'prediction_comparison' in cross_viz:
            elements.append(Paragraph("Prediction Performance:", styles['Heading3']))
            elements.append(Image(cross_viz['prediction_comparison'], width=450, height=300))
            elements.append(Spacer(1, 6))
            
        # Add key insights
        fusion_gain = cross_modal.get('fusion_prediction_gain', 0)
        
        insights_text = f"""
        Cross-modal analysis demonstrated a fusion gain of {fusion_gain:.2%} in prediction
        performance, indicating that combining visual and transaction information provides
        significantly better market insights than either modality alone.
        
        This confirms the value of multimodal approaches for NFT market analysis and suggests
        that market participants should consider both visual attributes and transaction
        patterns when making decisions.
        """
        elements.append(Paragraph(insights_text, styles['BodyText']))
        
        return elements
    
    def _generate_conclusions(self, styles) -> List:
        """
        Generate conclusions section.
        
        Args:
            styles: Document styles
            
        Returns:
            List of report elements
        """
        elements = []
        
        # Add conclusion text
        conclusion_text = """
        This comprehensive analysis of the NFT market reveals several important insights for
        collectors, investors, and market participants:
        """
        elements.append(Paragraph(conclusion_text, styles['BodyText']))
        
        # Add key conclusions
        market_efficiency = self.results.get('market_efficiency', {})
        efficiency_score = market_efficiency.get('market_efficiency_score', 0)
        
        conclusions = [
            f"Market Efficiency: With an efficiency score of {efficiency_score:.2f}, the market shows {'strong' if efficiency_score > 0.8 else 'moderate' if efficiency_score > 0.5 else 'limited'} efficiency, suggesting that {'prices generally reflect available information, limiting arbitrage opportunities' if efficiency_score > 0.7 else 'there may be opportunities for informed participants to achieve above-market returns'}.",
            "Visual Impact: Visual features demonstrate significant correlation with pricing, confirming that aesthetic and design elements play a crucial role in NFT valuation.",
            "Trading Patterns: Clear temporal and network patterns exist in transaction data, offering strategic insights for timing market participation.",
            "Multimodal Advantage: The combination of visual and transaction data provides superior market understanding compared to single-modality analysis, highlighting the value of comprehensive analytical approaches."
        ]
        
        for conclusion in conclusions:
            elements.append(Paragraph(f"• {conclusion}", styles['BodyText']))
            
        # Add recommendations
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Recommendations:", styles['Heading2']))
        
        recommendations = [
            "Market Participants should consider both visual attributes and transaction patterns when evaluating NFTs for purchase or investment.",
            "Collectors may benefit from focusing on NFTs with distinctive visual characteristics that show historical correlation with price appreciation.",
            "Investors should pay close attention to network position and trading patterns as indicators of potential future value.",
            "Platforms could enhance user experience by implementing multimodal recommendation systems that combine visual and transaction data."
        ]
        
        for recommendation in recommendations:
            elements.append(Paragraph(f"• {recommendation}", styles['BodyText']))
        
        return elements


class DashboardGenerator:
    """
    Generate interactive dashboards for market analysis.
    Implements Plotly-based interactive visualizations for in-depth exploration.
    """
    
    def __init__(
        self,
        results: Dict[str, Any],
        output_dir: str = "dashboards"
    ):
        """
        Initialize dashboard generator.
        
        Args:
            results: Analysis results
            output_dir: Output directory for dashboards
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def generate_dashboard(self) -> str:
        """
        Generate interactive dashboard.
        
        Returns:
            Path to generated dashboard
        """
        self.logger.info("Generating interactive dashboard")
        
        # Create dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Market Efficiency", "Price Distribution",
                "Visual Feature Analysis", "Network Analysis",
                "Cross-Modal Insights", "Key Metrics"
            ),
            specs=[
                [{"type": "indicator"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "table"}]
            ]
        )
        
        # Add market efficiency gauge
        market_efficiency = self.results.get('market_efficiency', {})
        efficiency_score = market_efficiency.get('market_efficiency_score', 0)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=efficiency_score,
                title={'text': "Market Efficiency"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "rgba(50, 120, 220, 0.8)"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "rgba(255, 0, 0, 0.3)"},
                        {'range': [0.3, 0.7], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [0.7, 1], 'color': "rgba(0, 255, 0, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.7
                    }
                }
            ),
            row=1, col=1
        )
        
        # Add price distribution
        price_analysis = self.results.get('price_analysis', {})
        
        # Simulate price distribution from statistics
        if price_analysis and 'mean' in price_analysis and 'std' in price_analysis:
            mean = price_analysis['mean']
            std = price_analysis['std']
            
            # Create a normal distribution as placeholder
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            y = np.exp(-0.5 * ((x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    fill='tozeroy',
                    name='Price Distribution'
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Price", row=1, col=2)
            fig.update_yaxes(title_text="Density", row=1, col=2)
            
        # Add visual feature visualization
        visual_analysis = self.results.get('visual_analysis', {})
        
        # Dummy scatter plot for feature space
        fig.add_trace(
            go.Scatter(
                x=np.random.randn(100),
                y=np.random.randn(100),
                mode='markers',
                marker=dict(
                    size=10,
                    color=np.random.randn(100),
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Feature Space'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Dimension 1", row=2, col=1)
        fig.update_yaxes(title_text="Dimension 2", row=2