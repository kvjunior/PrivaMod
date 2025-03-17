"""
PrivaMod Utility Functions
==================

This module provides comprehensive utility functions for the PrivaMod system:

1. Logging:
   - Configurable logging setup
   - Performance tracking
   - Error handling

2. Metrics:
   - Evaluation metrics for model performance
   - Privacy metrics
   - System performance metrics
  
3. Visualization:
   - Helper functions for generating plots
   - Color schemes and styles
   - Formatting utilities

4. System:
   - Hardware monitoring
   - Memory optimization
   - System benchmarking

These utilities support all aspects of the PrivaMod architecture,
providing common functionality across the system components.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import time
import json
import yaml
import psutil
import GPUtil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import threading
import concurrent.futures
from contextlib import contextmanager


# --------- Logging Utilities ---------

def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    debug: bool = False
) -> logging.Logger:
    """
    Set up a configurable logging system.
    
    Args:
        log_file: Optional path to log file
        console_level: Logging level for console output
        file_level: Logging level for file output
        debug: Whether to enable debug mode
        
    Returns:
        Configured logger instance
    """
    # Override levels if debug mode is enabled
    if debug:
        console_level = logging.DEBUG
        file_level = logging.DEBUG
    
    # Create logger
    logger = logging.getLogger('privaMod')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file is not None:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# --------- Performance Metrics ---------

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of regression metrics
    """
    # Ensure input is numpy array
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate mean absolute percentage error (with epsilon to avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    
    # Calculate explained variance
    explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'explained_variance': float(explained_variance)
    }


def calculate_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive binary classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Predicted probabilities (optional)
        
    Returns:
        Dictionary of classification metrics
    """
    # Ensure input is numpy array
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }
    
    # Calculate AUC if scores are provided
    if y_score is not None:
        y_score = np.array(y_score).flatten()
        try:
            auc = roc_auc_score(y_true, y_score)
            metrics['auc'] = float(auc)
        except:
            metrics['auc'] = 0.5  # Default value
    
    return metrics


def calculate_market_efficiency_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_range: Optional[float] = None
) -> float:
    """
    Calculate market efficiency score based on prediction accuracy.
    
    Args:
        y_true: Ground truth prices
        y_pred: Predicted prices
        price_range: Optional price range (max - min) for normalization
        
    Returns:
        Market efficiency score (0-1 scale)
    """
    # Ensure input is numpy array
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate mean absolute error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Normalize by price range if provided, otherwise use observed range
    if price_range is None:
        price_range = y_true.max() - y_true.min()
        if price_range == 0:
            price_range = np.mean(y_true)  # Fallback for constant prices
    
    # Calculate efficiency score (1 - normalized error)
    efficiency_score = 1 - (mae / price_range)
    
    # Clip to 0-1 range
    efficiency_score = max(0, min(1, efficiency_score))
    
    return float(efficiency_score)


# --------- System Monitoring ---------

def get_system_stats() -> Dict[str, Any]:
    """
    Get comprehensive system statistics.
    
    Returns:
        Dictionary of system statistics
    """
    stats = {}
    
    # CPU stats
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    stats['cpu'] = {
        'percent': cpu_percent,
        'cores': cpu_count,
        'frequency': cpu_freq.current if cpu_freq else None
    }
    
    # Memory stats
    memory = psutil.virtual_memory()
    stats['memory'] = {
        'total': memory.total,
        'available': memory.available,
        'used': memory.used,
        'percent': memory.percent
    }
    
    # Disk stats
    disk = psutil.disk_usage('/')
    stats['disk'] = {
        'total': disk.total,
        'used': disk.used,
        'free': disk.free,
        'percent': disk.percent
    }
    
    # GPU stats if available
    try:
        gpus = GPUtil.getGPUs()
        gpu_stats = []
        
        for i, gpu in enumerate(gpus):
            gpu_stats.append({
                'id': i,
                'name': gpu.name,
                'load': gpu.load,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'temperature': gpu.temperature
            })
            
        stats['gpu'] = gpu_stats
    except:
        stats['gpu'] = None
    
    return stats


@contextmanager
def measure_time(name: str = None) -> float:
    """
    Context manager to measure execution time.
    
    Args:
        name: Optional name for logging
        
    Returns:
        Elapsed time in seconds
    """
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    
    if name:
        print(f"{name} completed in {elapsed_time:.4f} seconds")
    
    return elapsed_time


def benchmark_system(
    config: Dict[str, Any],
    output_dir: str = "benchmarks"
) -> Dict[str, Any]:
    """
    Run comprehensive system benchmarks.
    
    Args:
        config: System configuration
        output_dir: Output directory for benchmark results
        
    Returns:
        Dictionary of benchmark results
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Starting system benchmarks...")
    benchmark_results = {}
    
    # Get system information
    benchmark_results['system_info'] = get_system_stats()
    
    # CPU benchmark
    print("Running CPU benchmark...")
    with measure_time("CPU benchmark") as cpu_time:
        # Matrix multiplication benchmark
        matrix_sizes = [1000, 2000, 4000]
        cpu_results = {}
        
        for size in matrix_sizes:
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            
            start = time.time()
            np.dot(a, b)
            end = time.time()
            
            cpu_results[f'{size}x{size}_matmul'] = end - start
            
        benchmark_results['cpu_benchmark'] = cpu_results
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        print("Running GPU benchmark...")
        with measure_time("GPU benchmark") as gpu_time:
            # GPU matrix multiplication benchmark
            matrix_sizes = [1000, 2000, 4000, 8000]
            gpu_results = {}
            
            for size in matrix_sizes:
                a = torch.rand(size, size, device='cuda')
                b = torch.rand(size, size, device='cuda')
                
                # Warmup
                torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # Benchmark
                start = time.time()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                end = time.time()
                
                gpu_results[f'{size}x{size}_matmul'] = end - start
                
            # GPU memory bandwidth benchmark
            memory_sizes = [1 * 2**20, 10 * 2**20, 100 * 2**20]  # 1MB, 10MB, 100MB
            for size in memory_sizes:
                elements = size // 4  # 4 bytes per float32
                
                a = torch.rand(elements, device='cuda')
                b = torch.rand(elements, device='cuda')
                
                # Warmup
                c = a + b
                torch.cuda.synchronize()
                
                # Benchmark
                start = time.time()
                for _ in range(10):
                    c = a + b
                torch.cuda.synchronize()
                end = time.time()
                
                # Calculate bandwidth in GB/s
                bytes_processed = elements * 4 * 3 * 10  # read a, read b, write c, 10 iterations
                seconds = end - start
                bandwidth = bytes_processed / (1024**3) / seconds
                
                gpu_results[f'{size//2**20}MB_bandwidth'] = bandwidth
                
            benchmark_results['gpu_benchmark'] = gpu_results
    
    # Data loading benchmark
    print("Running data loading benchmark...")
    with measure_time("Data loading benchmark") as data_time:
        data_sizes = [1000, 10000, 100000]
        batch_sizes = [32, 128, 512]
        num_workers_list = [0, 4, 8]
        
        data_results = {}
        
        for size in data_sizes:
            # Create random dataset
            dataset = [(np.random.rand(3, 224, 224), np.random.rand()) for _ in range(size)]
            
            for batch_size in batch_sizes:
                for num_workers in num_workers_list:
                    dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers
                    )
                    
                    start = time.time()
                    for _ in dataloader:
                        pass
                    end = time.time()
                    
                    data_results[f'size_{size}_batch_{batch_size}_workers_{num_workers}'] = end - start
                    
        benchmark_results['data_loading_benchmark'] = data_results
    
    # Privacy operations benchmark
    print("Running privacy operations benchmark...")
    with measure_time("Privacy operations benchmark") as privacy_time:
        from privacy import RenyiPrivacyAccountant
        
        # Create privacy accountant
        accountant = RenyiPrivacyAccountant()
        
        # Benchmark noise multiplier calculation
        data_sizes = [1000, 10000, 100000]
        batch_sizes = [32, 128, 512]
        epsilons = [0.1, 1.0, 10.0]
        
        privacy_results = {}
        
        for size in data_sizes:
            for batch_size in batch_sizes:
                for epsilon in epsilons:
                    start = time.time()
                    noise_multiplier = accountant.compute_noise_multiplier(
                        num_samples=size,
                        batch_size=batch_size,
                        epochs=10,
                        target_epsilon=epsilon
                    )
                    end = time.time()
                    
                    privacy_results[f'size_{size}_batch_{batch_size}_epsilon_{epsilon}'] = {
                        'time': end - start,
                        'noise_multiplier': noise_multiplier
                    }
                    
        benchmark_results['privacy_benchmark'] = privacy_results
    
    # Save results
    benchmark_path = output_path / "benchmark.yaml"
    with open(benchmark_path, 'w') as f:
        yaml.dump(benchmark_results, f, default_flow_style=False)
        
    print(f"Benchmarks completed and saved to {benchmark_path}")
    
    return benchmark_results


# --------- Data Utilities ---------

def load_json(file_path: str) -> Dict:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary of loaded data
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return {}
        

def save_json(data: Dict, file_path: str, indent: int = 2) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to JSON file
        indent: Indentation for formatting
        
    Returns:
        Success status
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False


def load_yaml(file_path: str) -> Dict:
    """
    Load YAML file with error handling.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary of loaded data
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML from {file_path}: {e}")
        return {}


def save_yaml(data: Dict, file_path: str) -> bool:
    """
    Save data to YAML file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to YAML file
        
    Returns:
        Success status
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving YAML to {file_path}: {e}")
        return False


def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert tensor to numpy with proper handling.
    
    Args:
        tensor: PyTorch tensor or NumPy array
        
    Returns:
        NumPy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Unsupported type: {type(tensor)}")


def seed_everything(seed: int = 42):
    """
    Set seeds for reproducibility.
    
    Args:
        seed: Seed value
    """
    # Set Python seed
    import random
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seed
    torch.manual_seed(seed)
    
    # Set CUDA seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # For deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------- Visualization Utilities ---------

def set_plotting_style():
    """Set consistent plotting style for visualizations."""
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


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_path: Optional path to save figure
        show: Whether to display the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o', markersize=4)
    plt.plot(val_losses, label='Validation Loss', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_privacy_utility_tradeoff(
    epsilons: List[float],
    utilities: List[float],
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot privacy-utility tradeoff curve.
    
    Args:
        epsilons: List of privacy budgets
        utilities: List of utility metrics
        output_path: Optional path to save figure
        show: Whether to display the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, utilities, marker='o', linewidth=2)
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Utility Metric')
    plt.title('Privacy-Utility Tradeoff')
    plt.grid(True)
    
    # Add annotations for key points
    for i, (eps, util) in enumerate(zip(epsilons, utilities)):
        plt.annotate(
            f'ε={eps:.2f}, U={util:.2f}',
            xy=(eps, util),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        output_path: Optional path to save figure
        show: Whether to display the figure
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: List[float],
    top_n: int = 20,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importance_scores: List of importance scores
        top_n: Number of top features to show
        output_path: Optional path to save figure
        show: Whether to display the figure
    """
    # Sort features by importance
    indices = np.argsort(importance_scores)[::-1]
    top_indices = indices[:top_n]
    
    # Get top feature names and scores
    top_names = [feature_names[i] for i in top_indices]
    top_scores = [importance_scores[i] for i in top_indices]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    plt.barh(range(len(top_names)), top_scores, align='center')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    
    # Add values on bars
    for i, score in enumerate(top_scores):
        plt.text(score + 0.01, i, f'{score:.3f}', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()