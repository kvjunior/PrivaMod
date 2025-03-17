"""
PrivaMod Training System
==================

This module implements an optimized training system for the PrivaMod architecture:

1. Training:
   - Distributed multi-GPU training with DDP
   - Mixed precision training for performance
   - Privacy-aware optimization

2. Evaluation:
   - Comprehensive model evaluation
   - Privacy-utility tradeoff analysis
   - Performance metrics and benchmarking

3. Optimization:
   - Adaptive learning rate scheduling
   - Gradient accumulation for large batches
   - Early stopping with validation monitoring

The training system is designed for high-performance execution on multi-GPU systems,
with specific optimizations for privacy-preserving neural architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import numpy as np
import os
import logging
import time
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import threading

# For monitoring
import psutil
import GPUtil

# Import privacy components
from privacy import PrivacyEngine, PrivacyAuditor

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    """
    Comprehensive trainer for PrivaMod architecture.
    Implements distributed training, evaluation, and optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None,
        output_dir: str = "results",
        privacy_engine: Optional[PrivacyEngine] = None
    ):
        """
        Initialize trainer with comprehensive configuration.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration
            output_dir: Output directory
            privacy_engine: Optional privacy engine
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.privacy_engine = privacy_engine
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Extract training parameters
        self.max_epochs = self.config.get('epochs', 100)
        self.lr = self.config.get('optimizer', {}).get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('optimizer', {}).get('weight_decay', 1e-5)
        self.batch_size = self.config.get('batch_size', 32)
        self.grad_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        # Loss weights
        self.loss_weights = self.config.get('loss', {})
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self._init_optimizer()
        self._init_scheduler()
        self._init_scaler()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.stop_training = False
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'train_metrics': [],
            'val_metrics': [],
            'privacy_metrics': []
        }
        
        # Performance monitoring
        self.monitoring_stats = {
            'gpu_utilization': [],
            'memory_usage': [],
            'batch_times': [],
            'epoch_times': []
        }
    
    def _init_optimizer(self):
        """Initialize optimizer with configuration."""
        # Get optimizer parameters
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam').lower()
        
        # Create optimizer
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(
                    optimizer_config.get('beta1', 0.9),
                    optimizer_config.get('beta2', 0.999)
                ),
                eps=optimizer_config.get('eps', 1e-8)
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=optimizer_config.get('momentum', 0.9)
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(
                    optimizer_config.get('beta1', 0.9),
                    optimizer_config.get('beta2', 0.999)
                ),
                eps=optimizer_config.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
        # Wrap with privacy-aware optimizer if privacy engine is provided
        if self.privacy_engine is not None:
            self.optimizer = self.privacy_engine.create_private_optimizer(
                self.optimizer,
                lr=self.lr
            )
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        # Get scheduler parameters
        scheduler_config = self.config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'cosine').lower()
        
        # Create scheduler
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', self.max_epochs),
                eta_min=scheduler_config.get('min_lr', 0)
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_name == 'warmup_cosine':
            # Create warmup scheduler
            from torch.optim.lr_scheduler import LambdaLR
            
            # Define warmup function
            warmup_epochs = scheduler_config.get('warmup_epochs', 5)
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    # Cosine decay after warmup
                    return 0.5 * (1 + np.cos(
                        np.pi * (epoch - warmup_epochs) / (self.max_epochs - warmup_epochs)
                    ))
                    
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.logger.warning(f"Unsupported scheduler: {scheduler_name}. Using None.")
            self.scheduler = None
    
    def _init_scaler(self):
        """Initialize gradient scaler for mixed precision training."""
        self.scaler = GradScaler() if self.config.get('use_amp', True) else None
    
    def train(self) -> str:
        """
        Train the model for specified number of epochs.
        
        Returns:
            Path to best model checkpoint
        """
        self.logger.info(f"Starting training for {self.max_epochs} epochs")
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['learning_rates'].append(current_lr)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Add privacy metrics if available
            if self.privacy_engine is not None:
                privacy_stats = self.privacy_engine.get_privacy_stats()
                self.history['privacy_metrics'].append(privacy_stats)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics, current_lr)
            
            # Check for improvement
            val_loss = val_metrics['val_loss']
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)
                
            # Regular checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self._save_checkpoint(epoch)
                
            # Update monitoring stats
            epoch_time = time.time() - epoch_start_time
            self.monitoring_stats['epoch_times'].append(epoch_time)
            
            # Check early stopping
            if self._check_early_stopping(val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
            # Check for stopping
            if self.stop_training:
                self.logger.info(f"Training stopped at epoch {epoch}")
                break
        
        # Final evaluation
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
            self.logger.info(f"Final test metrics: {test_metrics}")
            
        # Save training history
        self._save_history()
        
        # Log total training time
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        
        # Return path to best model
        return str(self.output_dir / "best_model.pt")
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        step_count = 0
        
        # Get dataset size for privacy accounting
        dataset_size = len(self.train_loader.dataset) if self.train_loader else None
        
        # Monitor batch times
        batch_times = []
        
        # Reset accumulated gradients
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Calculate loss
            with autocast(enabled=self.scaler is not None):
                # Forward pass
                outputs = self.model(**self._prepare_model_inputs(batch))
                
                # Calculate loss
                loss, loss_components = self._calculate_loss(outputs, batch)
                
                # Scale loss for gradient accumulation
                if self.grad_accumulation_steps > 1:
                    loss = loss / self.grad_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Update parameters when accumulation steps are reached
            if (batch_idx + 1) % self.grad_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Add privacy to gradients if needed
                if self.privacy_engine is not None:
                    current_epsilon = self.privacy_engine.update_privacy_budget(
                        batch_size=batch["image"].size(0) * self.grad_accumulation_steps,
                        data_size=dataset_size
                    )
                    
                    if current_epsilon is not None:
                        epoch_metrics['epsilon'] = current_epsilon
                
                # Update parameters
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                # Reset gradients
                self.optimizer.zero_grad()
                
            # Update metrics
            epoch_metrics['loss'] += loss.item() * self.grad_accumulation_steps
            for k, v in loss_components.items():
                epoch_metrics[k] += v.item()
                
            # Update step count
            step_count += 1
            self.global_step += 1
            
            # Record batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            self.monitoring_stats['batch_times'].append(batch_time)
            
            # Monitor GPU utilization
            if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
                gpu_stats = {}
                for i in range(torch.cuda.device_count()):
                    gpu_stats[f'gpu_{i}_util'] = torch.cuda.utilization(i)
                    gpu_stats[f'gpu_{i}_mem'] = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                    
                self.monitoring_stats['gpu_utilization'].append(gpu_stats)
                
            # Log batch metrics
            if (batch_idx + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.debug(
                    f"Epoch {self.epoch}, Batch {batch_idx+1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}, "
                    f"Batch Time: {batch_time:.3f}s"
                )
        
        # Calculate epoch averages
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= step_count
            
        # Add additional metrics
        epoch_metrics['avg_batch_time'] = np.mean(batch_times)
        epoch_metrics['gpu_memory_gb'] = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        
        return dict(epoch_metrics)
    
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        return self.evaluate(self.val_loader)
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on provided data loader.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        epoch_metrics = defaultdict(float)
        all_predictions = []
        all_targets = []
        step_count = 0
        
        for batch in data_loader:
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            with autocast(enabled=self.scaler is not None):
                outputs = self.model(**self._prepare_model_inputs(batch))
                
                # Calculate loss
                loss, loss_components = self._calculate_loss(outputs, batch)
            
            # Update metrics
            epoch_metrics['val_loss'] += loss.item()
            for k, v in loss_components.items():
                epoch_metrics[f"val_{k}"] += v.item()
                
            # Collect predictions for regression metrics
            if 'price_prediction' in outputs and 'price' in batch:
                all_predictions.append(outputs['price_prediction'].cpu())
                all_targets.append(batch['price'].cpu())
            
            # Update step count
            step_count += 1
            
        # Calculate epoch averages
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= step_count
            
        # Calculate regression metrics if available
        if all_predictions and all_targets:
            all_predictions = torch.cat(all_predictions).numpy()
            all_targets = torch.cat(all_targets).numpy()
            
            # Mean absolute error
            mae = np.mean(np.abs(all_predictions - all_targets))
            epoch_metrics['val_mae'] = mae
            
            # Mean squared error
            mse = np.mean((all_predictions - all_targets) ** 2)
            epoch_metrics['val_mse'] = mse
            
            # Root mean squared error
            rmse = np.sqrt(mse)
            epoch_metrics['val_rmse'] = rmse
            
            # Mean absolute percentage error (with epsilon to avoid division by zero)
            mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
            epoch_metrics['val_mape'] = mape
            
            # R^2 score
            if np.var(all_targets) > 0:
                r2 = 1 - (np.sum((all_targets - all_predictions) ** 2) / 
                           np.sum((all_targets - np.mean(all_targets)) ** 2))
                epoch_metrics['val_r2'] = r2
            
        return dict(epoch_metrics)
    
    def _calculate_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss with component weighting.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Tuple of (total loss, loss components)
        """
        loss_components = {}
        
        # Price prediction loss
        if 'price_prediction' in outputs and 'price' in batch:
            loss_components['price_loss'] = F.mse_loss(
                outputs['price_prediction'].squeeze(),
                batch['price']
            )
        
        # Attribute prediction loss
        if 'attribute_prediction' in outputs and 'attributes' in batch:
            loss_components['attribute_loss'] = F.binary_cross_entropy_with_logits(
                outputs['attribute_prediction'],
                batch['attributes']
            )
        
        # KL divergence loss (if available)
        if 'kl_loss' in outputs.get('losses', {}):
            loss_components['kl_loss'] = outputs['losses']['kl_loss']
        
        # Contrastive loss (if available)
        if 'contrastive_loss' in outputs.get('losses', {}):
            loss_components['contrastive_loss'] = outputs['losses']['contrastive_loss']
            
        # Mutual information loss (if available)
        if 'mi_loss' in outputs.get('losses', {}):
            loss_components['mi_loss'] = outputs['losses']['mi_loss']
        
        # Calculate weighted total loss
        total_loss = 0
        for loss_name, loss_value in loss_components.items():
            # Get weight from config (default to 1.0)
            weight = self.loss_weights.get(loss_name, 1.0)
            total_loss += weight * loss_value
        
        return total_loss, loss_components
    
    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch tensors to device.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch with tensors on device
        """
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, dict):
                result[k] = self._batch_to_device(v)
            else:
                result[k] = v
                
        return result
    
    def _prepare_model_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare inputs for model forward pass.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary of model inputs
        """
        # Extract required inputs from batch
        inputs = {}
        
        # Always include image and transaction data if available
        if 'image' in batch:
            inputs['images'] = batch['image']
            
        if 'sequence' in batch:
            inputs['transaction_data'] = batch['sequence']
            
        # Add graph if available
        if 'graph' in batch:
            inputs['transaction_graphs'] = batch['graph']
            
        return inputs
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        current_lr: float
    ):
        """
        Log metrics for epoch.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            current_lr: Current learning rate
        """
        # Log basic metrics
        log_message = (
            f"Epoch {epoch+1}/{self.max_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        
        # Add validation metrics
        val_info = ", ".join([
            f"{k.replace('val_', '')}: {v:.4f}"
            for k, v in val_metrics.items()
            if k != 'val_loss' and not k.startswith('val_val_')
        ])
        
        if val_info:
            log_message += f" | {val_info}"
            
        # Add privacy budget if available
        if 'epsilon' in train_metrics:
            log_message += f" | Privacy ε: {train_metrics['epsilon']:.4f}"
            
        self.logger.info(log_message)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved new best model with val_loss: {self.best_val_loss:.4f}")
            
        # Clean up old checkpoints
        if self.config.get('keep_n_checkpoints'):
            self._cleanup_checkpoints(self.config.get('keep_n_checkpoints'))
    
    def _cleanup_checkpoints(self, keep_n: int):
        """
        Remove old checkpoints, keeping only the most recent N.
        
        Args:
            keep_n: Number of checkpoints to keep
        """
        checkpoints = sorted([
            f for f in self.output_dir.glob("checkpoint_*.pt")
        ], key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-keep_n]:
            os.remove(checkpoint)
    
    def _save_history(self):
        """Save training history."""
        history_path = self.output_dir / "training_history.json"
        
        # Convert tensors and numpy arrays to lists
        serializable_history = {}
        for k, v in self.history.items():
            if isinstance(v, list):
                if v and isinstance(v[0], (dict, list)):
                    # Nested structure
                    serializable_history[k] = self._make_serializable(v)
                else:
                    # Simple list
                    serializable_history[k] = [float(x) if isinstance(x, (torch.Tensor, np.ndarray)) else x for x in v]
            else:
                serializable_history[k] = v
        
        # Save to file
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
            
        # Save monitoring stats
        monitoring_path = self.output_dir / "monitoring_stats.json"
        serializable_stats = self._make_serializable(self.monitoring_stats)
        
        with open(monitoring_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
    
    def _make_serializable(self, obj):
        """
        Convert object to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.item() if obj.size == 1 else obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        else:
            return obj
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            Whether to stop training
        """
        # Get early stopping config
        early_stopping = self.config.get('early_stopping', {})
        
        if not early_stopping.get('enabled', False):
            return False
            
        # Get parameters
        patience = early_stopping.get('patience', 10)
        min_delta = early_stopping.get('min_delta', 0)
        
        # Check if we have enough history
        if len(self.history['val_loss']) <= patience:
            return False
            
        # Check if validation loss has improved in the last 'patience' epochs
        min_loss_in_patience = min(self.history['val_loss'][-patience:])
        min_loss_before_patience = min(self.history['val_loss'][:-patience])
        
        # Stop if no improvement
        return min_loss_in_patience > min_loss_before_patience - min_delta
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load scaler state
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        # Load training state
        self.epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load history
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch-1}")


class PrivacyEvaluator:
    """
    Evaluate privacy-utility tradeoffs and privacy guarantees.
    Implements comprehensive privacy evaluation and analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        epsilon: float = 0.1,
        delta: float = 1e-5,
        output_dir: str = "privacy_evaluation"
    ):
        """
        Initialize privacy evaluator.
        
        Args:
            model: Model to evaluate
            train_dataset: Training dataset
            test_dataset: Test dataset
            epsilon: Target epsilon
            delta: Target delta
            output_dir: Output directory
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epsilon = epsilon
        self.delta = delta
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize privacy auditor
        self.auditor = PrivacyAuditor(
            model=model,
            dataset=train_dataset,
            epsilon=epsilon,
            delta=delta
        )
        
        # Initialize result storage
        self.results = {}
    
    def run_privacy_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive privacy evaluation.
        
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info("Starting privacy evaluation")
        
        # Run privacy attacks
        self._run_membership_inference()
        self._run_model_inversion()
        self._run_attribute_inference()
        
        # Assess privacy guarantees
        self.results['privacy_assessment'] = self.auditor.assess_privacy_guarantees()
        
        # Generate privacy-utility analysis
        self._analyze_privacy_utility()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _run_membership_inference(self):
        """Run membership inference attack evaluation."""
        self.logger.info("Running membership inference attack evaluation")
        
        # Run attack
        results = self.auditor.run_membership_inference_attack(self.test_dataset)
        
        # Store results
        self.results['membership_inference'] = results
        
        # Log results
        self.logger.info(f"Membership inference results: {results}")
    
    def _run_model_inversion(self):
        """Run model inversion attack evaluation."""
        self.logger.info("Running model inversion attack evaluation")
        
        # Run attack
        results = self.auditor.run_model_inversion_attack()
        
        # Store results
        self.results['model_inversion'] = results
        
        # Log results
        self.logger.info(f"Model inversion results: {results}")
    
    def _run_attribute_inference(self):
        """Run attribute inference attack evaluation."""
        self.logger.info("Running attribute inference attack evaluation")
        
        # Get available attributes
        attribute_name = "price"  # Default attribute
        
        if hasattr(self.train_dataset, 'attribute_mapping') and self.train_dataset.attribute_mapping:
            # Get first attribute as example
            attribute_name = list(self.train_dataset.attribute_mapping.keys())[0]
            
        # Run attack
        results = self.auditor.run_attribute_inference_attack(attribute_name)
        
        # Store results
        self.results['attribute_inference'] = results
        
        # Log results
        self.logger.info(f"Attribute inference results: {results}")
    
    def _analyze_privacy_utility(self):
        """Analyze privacy-utility tradeoff."""
        self.logger.info("Analyzing privacy-utility tradeoff")
        
        # Get assessment
        assessment = self.results.get('privacy_assessment', {})
        
        # Define utility metrics
        utility_metrics = {}
        
        # Membership inference utility cost
        if 'membership_inference' in self.results:
            # Lower advantage means better privacy but might impact utility
            utility_impact = 1.0 - self.results['membership_inference'].get('advantage', 0.0)
            utility_metrics['membership_inference_impact'] = utility_impact
            
        # Model inversion utility cost
        if 'model_inversion' in self.results:
            # Higher reconstruction error means better privacy
            privacy_score = 1.0 - self.results['model_inversion'].get('privacy_leakage', 0.0)
            utility_metrics['model_inversion_impact'] = privacy_score
            
        # Calculate overall privacy-utility score
        if utility_metrics:
            overall_score = sum(utility_metrics.values()) / len(utility_metrics)
            utility_metrics['overall_privacy_utility_score'] = overall_score
            
        # Store results
        self.results['privacy_utility'] = utility_metrics
        
        # Log results
        self.logger.info(f"Privacy-utility analysis: {utility_metrics}")
    
    def _save_results(self):
        """Save evaluation results."""
        results_path = self.output_dir / "privacy_evaluation.json"
        
        # Make results serializable
        serializable_results = json.loads(
            json.dumps(self.results, default=lambda x: float(x) if isinstance(x, (np.ndarray, np.number)) else str(x))
        )
        
        # Save to file
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        self.logger.info(f"Privacy evaluation results saved to {results_path}")
        
        # Generate visualization
        self._generate_privacy_visualization()
    
    def _generate_privacy_visualization(self):
        """Generate privacy evaluation visualization."""
        # Create visualization directory
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Privacy-utility tradeoff plot
        plt.figure(figsize=(10, 6))
        
        # Get metrics
        utility_metrics = self.results.get('privacy_utility', {})
        
        if utility_metrics:
            # Create bar chart
            plt.bar(
                range(len(utility_metrics)),
                list(utility_metrics.values()),
                tick_label=list(utility_metrics.keys())
            )
            
            plt.title('Privacy-Utility Tradeoff')
            plt.ylabel('Score (higher is better)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(vis_dir / "privacy_utility_tradeoff.png", dpi=300)
            plt.close()
            
        # Privacy guarantee plot
        plt.figure(figsize=(10, 6))
        
        # Get privacy assessment
        assessment = self.results.get('privacy_assessment', {})
        
        if 'empirical_guarantees' in assessment:
            guarantees = assessment['empirical_guarantees']
            
            # Plot empirical vs theoretical
            labels = []
            empirical = []
            theoretical = []
            
            for attack_type, metrics in guarantees.items():
                if 'empirical_advantage' in metrics and 'theoretical_bound' in metrics:
                    labels.append(attack_type)
                    empirical.append(metrics['empirical_advantage'])
                    theoretical.append(metrics['theoretical_bound'])
            
            if labels:
                x = range(len(labels))
                width = 0.35
                
                plt.bar(
                    [i - width/2 for i in x],
                    empirical,
                    width=width,
                    label='Empirical'
                )
                plt.bar(
                    [i + width/2 for i in x],
                    theoretical,
                    width=width,
                    label='Theoretical'
                )
                
                plt.title('Privacy Guarantees: Empirical vs Theoretical')
                plt.xticks(x, labels, rotation=45)
                plt.ylabel('Privacy Leakage')
                plt.legend()
                plt.tight_layout()
                
                # Save figure
                plt.savefig(vis_dir / "privacy_guarantees.png", dpi=300)
                plt.close()


def create_trainer(
    model: nn.Module,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    config: Dict[str, Any] = None,
    output_dir: str = "results",
    privacy_engine: Optional[PrivacyEngine] = None
) -> Trainer:
    """
    Create trainer instance.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Training configuration
        output_dir: Output directory
        privacy_engine: Optional privacy engine
        
    Returns:
        Configured Trainer instance
    """
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        output_dir=output_dir,
        privacy_engine=privacy_engine
    )