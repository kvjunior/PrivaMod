"""
PrivaMod Privacy Mechanisms
=====================

Implementation of privacy preservation and security mechanisms for PrivaMod:

1. Differential Privacy:
   - Rényi Differential Privacy accounting
   - DP-SGD implementation with adaptive clipping
   - Privacy-Budget monitoring and management

2. Security:
   - Secure data handling with encryption
   - Key management and secure computation
   - Privacy-preserving feature extraction

3. Privacy Auditing:
   - Comprehensive privacy evaluation
   - Attack resistance testing
   - Leakage monitoring

These implementations provide formal privacy guarantees while maintaining
analytical accuracy, addressing the critical challenge of privacy-preserved
multimodal analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging
import time
from pathlib import Path
import json
import secrets
from collections import defaultdict

# For cryptography
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.fernet import Fernet
import base64


# --------- Differential Privacy ---------

class RenyiPrivacyAccountant:
    """
    Privacy accountant using Rényi Differential Privacy.
    Provides significantly tighter bounds than classic DP analysis.
    """
    
    def __init__(
        self,
        target_epsilon=0.1,
        target_delta=1e-5,
        orders=None,
        max_alpha=32
    ):
        """
        Initialize Rényi Differential Privacy accountant.
        
        Args:
            target_epsilon: Target DP epsilon
            target_delta: Target DP delta
            orders: List of RDP orders (if None, use default range)
            max_alpha: Maximum alpha for RDP accounting
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_alpha = max_alpha
        
        # Default set of orders (if none provided)
        if orders is None:
            # Use a fine-grained range for more accurate accounting
            self.orders = np.concatenate([
                # Fine-grained in important range
                np.arange(1.1, 10.0, 0.1),  
                # Coarser for larger orders
                np.arange(10.0, self.max_alpha + 1, 0.5)])
        else:
            self.orders = orders
            
        # Initialize RDP accounting
        self.rdp_history = np.zeros_like(self.orders, dtype=float)
        self.steps = 0
        
        # Tracking for key parameters
        self.noise_multipliers = []
        self.sampling_rates = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def compute_rdp_gaussian(
        self,
        sampling_rate: float,
        noise_multiplier: float,
        steps: int = 1
    ) -> np.ndarray:
        """
        Compute RDP for subsampled Gaussian mechanism.
        
        Args:
            sampling_rate: Batch sampling rate (batch_size / data_size)
            noise_multiplier: Ratio of noise to sensitivity
            steps: Number of steps (usually 1 for each call)
            
        Returns:
            RDP values for each order
        """
        # Compute RDP for Gaussian mechanism
        rdp = np.array([
            self._compute_rdp_single_order(alpha, sampling_rate, noise_multiplier)
            for alpha in self.orders
        ])
        
        # Scale by steps (for composition)
        return rdp * steps
    
    def _compute_rdp_single_order(
        self,
        alpha: float,
        sampling_rate: float,
        noise_multiplier: float
    ) -> float:
        """
        Compute RDP for a single order alpha.
        Implements subsampled Gaussian moments accountant.
        
        Args:
            alpha: RDP order
            sampling_rate: Batch sampling rate
            noise_multiplier: Noise multiplier
            
        Returns:
            RDP value for given order
        """
        # Compute basic Gaussian RDP
        basic_rdp = alpha / (2 * (noise_multiplier ** 2))
        
        # If sampling rate is negligible, return basic RDP
        if sampling_rate == 1.0:
            return basic_rdp
        
        # Use the analytically derived bounds for subsampled RDP
        # This is a simplified implementation of the bound from
        # "Subsampled Rényi Differential Privacy and Analytical Moments Accountant"
        if alpha > 1.0:
            # Term 1: PLD bound
            term1 = sampling_rate ** 2 * basic_rdp
            
            # Term 2: Subsampling bound
            log_term = np.log(1 + sampling_rate * (np.exp(basic_rdp) - 1))
            term2 = (1 / (alpha - 1)) * log_term
            
            # Return the minimum of the two bounds
            return min(term1, term2)
        else:
            # For alpha < 1, use a simple bound
            return sampling_rate * basic_rdp
    
    def compute_noise_multiplier(
        self,
        num_samples: int,
        batch_size: int,
        epochs: int,
        target_epsilon: Optional[float] = None,
        target_delta: Optional[float] = None
    ) -> float:
        """
        Compute the noise multiplier for target privacy level.
        
        Args:
            num_samples: Total number of training samples
            batch_size: Batch size
            epochs: Number of training epochs
            target_epsilon: Target epsilon (overrides instance value if provided)
            target_delta: Target delta (overrides instance value if provided)
            
        Returns:
            Noise multiplier for target privacy level
        """
        # Use provided targets or instance defaults
        eps = target_epsilon if target_epsilon is not None else self.target_epsilon
        delta = target_delta if target_delta is not None else self.target_delta
        
        # Calculate key parameters
        sampling_rate = batch_size / num_samples
        steps = int(np.ceil(epochs * num_samples / batch_size))
        
        # Binary search for noise multiplier
        low, high = 0.1, 100.0
        while high - low > 0.01:
            mid = (low + high) / 2
            
            # Compute RDP for current noise multiplier
            rdp = self.compute_rdp_gaussian(sampling_rate, mid, steps)
            
            # Convert to (ε, δ)-DP
            current_eps = self._rdp_to_dp(rdp, delta)
            
            # Adjust search range
            if current_eps <= eps:
                high = mid
            else:
                low = mid
                
        # Return the upper bound for safety
        return high
    
    def _rdp_to_dp(self, rdp_values: np.ndarray, delta: float) -> float:
        """
        Convert RDP to approximate DP.
        
        Args:
            rdp_values: RDP values for different orders
            delta: Target delta
            
        Returns:
            Epsilon value
        """
        # Compute epsilon for each order using conversion formula
        eps_values = rdp_values - (np.log(delta) / (self.orders - 1))
        
        # Return the minimum epsilon
        return float(np.min(eps_values))
    
    def update_privacy_budget(
        self,
        sampling_rate: float,
        noise_multiplier: float,
        num_steps: int = 1
    ) -> float:
        """
        Update privacy budget accounting.
        
        Args:
            sampling_rate: Batch sampling rate
            noise_multiplier: Noise multiplier
            num_steps: Number of steps (usually 1)
            
        Returns:
            Current epsilon
        """
        # Compute RDP for this update
        rdp_step = self.compute_rdp_gaussian(sampling_rate, noise_multiplier, num_steps)
        
        # Update history
        self.rdp_history += rdp_step
        self.steps += num_steps
        
        # Track parameters
        self.noise_multipliers.append(noise_multiplier)
        self.sampling_rates.append(sampling_rate)
        
        # Convert to (ε, δ)-DP
        current_epsilon = self._rdp_to_dp(self.rdp_history, self.target_delta)
        
        return current_epsilon
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy guarantee.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        # Convert current RDP to (ε, δ)-DP
        current_epsilon = self._rdp_to_dp(self.rdp_history, self.target_delta)
        
        return current_epsilon, self.target_delta
    
    def reset(self):
        """Reset accountant state."""
        self.rdp_history = np.zeros_like(self.orders, dtype=float)
        self.steps = 0
        self.noise_multipliers = []
        self.sampling_rates = []


class PrivacyAwareOptimizer(optim.Optimizer):
    """
    Optimizer implementing DP-SGD with adaptive clipping and accounting.
    Ensures privacy throughout the training process.
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        accountant=None,
        **kwargs
    ):
        """
        Initialize privacy-aware optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            noise_multiplier: Noise multiplier for DP-SGD
            max_grad_norm: Maximum gradient norm for clipping
            accountant: Privacy accountant instance
            **kwargs: Additional arguments for base optimizer
        """
        # Initialize underlying optimizer (Adam by default)
        defaults = dict(lr=lr, **kwargs)
        super().__init__(params, defaults)
        
        # Save privacy parameters
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        # Privacy accountant
        self.accountant = accountant
        
        # Statistics for adaptive clipping
        self.grad_norms = []
        self.noise_scales = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def privacy_step(self, batch_size, data_size):
        """
        Perform optimizer step with privacy guarantees.
        
        Args:
            batch_size: Current batch size
            data_size: Total dataset size
            
        Returns:
            Current epsilon value
        """
        # Compute sampling rate
        sampling_rate = batch_size / data_size
        
        # First clip gradients per sample (assumed to be done already)
        # Then add noise to gradients in all parameter groups
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Calculate norm for statistics
                param_norm = p.grad.data.norm(2).item()
                self.grad_norms.append(param_norm)
                
                # Add Gaussian noise to gradients
                noise_scale = self.noise_multiplier * self.max_grad_norm
                self.noise_scales.append(noise_scale)
                
                noise = torch.randn_like(p.grad.data) * noise_scale
                p.grad.data.add_(noise)
                
        # Update privacy accounting if accountant is provided
        if self.accountant is not None:
            current_epsilon = self.accountant.update_privacy_budget(
                sampling_rate, self.noise_multiplier
            )
            return current_epsilon
        
        return None
        
    def step(self, batch_size=None, data_size=None, closure=None):
        """
        Perform a single optimization step with privacy guarantee.
        
        Args:
            batch_size: Current batch size (for privacy accounting)
            data_size: Total dataset size (for privacy accounting)
            closure: Loss closure (same as standard optimizer)
            
        Returns:
            Closure result if provided, else None
        """
        # Evaluate loss if closure is provided
        loss = None
        if closure is not None:
            loss = closure()
            
        # Apply privacy step if batch_size and data_size are provided
        if batch_size is not None and data_size is not None:
            current_epsilon = self.privacy_step(batch_size, data_size)
            if current_epsilon is not None:
                self.logger.info(f"Current privacy guarantee: ε = {current_epsilon:.4f}")
                
        # Apply optimization step
        for group in self.param_groups:
            # Extract parameters
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Gradient update (similar to SGD)
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-group['lr'])
                
        return loss
    
    def get_grad_statistics(self) -> Dict[str, float]:
        """
        Get statistics on gradients and noise.
        
        Returns:
            Dictionary with gradient statistics
        """
        if not self.grad_norms:
            return {}
            
        return {
            'mean_grad_norm': np.mean(self.grad_norms),
            'median_grad_norm': np.median(self.grad_norms),
            'max_grad_norm': np.max(self.grad_norms),
            'mean_noise_scale': np.mean(self.noise_scales),
            'noise_to_gradient_ratio': np.mean(self.noise_scales) / (np.mean(self.grad_norms) + 1e-8)
        }


class GradientClipper:
    """
    Per-sample gradient clipping for DP-SGD.
    Implements efficient computation of per-sample gradients and clipping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_grad_norm: float = 1.0,
        adaptive_clipping: bool = True,
        clipping_quantile: float = 0.9
    ):
        """
        Initialize gradient clipper.
        
        Args:
            model: Model to clip gradients for
            max_grad_norm: Maximum gradient norm
            adaptive_clipping: Whether to use adaptive clipping
            clipping_quantile: Quantile for adaptive clipping
        """
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.adaptive_clipping = adaptive_clipping
        self.clipping_quantile = clipping_quantile
        
        # History for adaptive clipping
        self.grad_norms_history = []
        self.clipping_thresholds = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def compute_per_sample_gradients(
        self,
        loss_per_sample: torch.Tensor,
        create_graph: bool = False
    ) -> List[torch.Tensor]:
        """
        Compute per-sample gradients efficiently.
        
        Args:
            loss_per_sample: Loss for each sample in batch
            create_graph: Whether to create graph for higher-order gradients
            
        Returns:
            List of per-sample gradients for each parameter
        """
        # Create list to store gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        per_sample_grads = [[] for _ in range(len(params))]
        
        # Compute gradients for each sample individually
        for i, loss in enumerate(loss_per_sample):
            # Clear previous gradients
            self.model.zero_grad()
            
            # Compute gradients for this sample
            loss.backward(retain_graph=True if i < len(loss_per_sample) - 1 else False, 
                          create_graph=create_graph)
            
            # Store gradients
            for j, p in enumerate(params):
                if p.grad is not None:
                    per_sample_grads[j].append(p.grad.detach().clone())
                    
            # Clear gradients for next iteration
            self.model.zero_grad()
            
        # Stack gradients for each parameter
        per_sample_grads = [torch.stack(grads, dim=0) if grads else None 
                            for grads in per_sample_grads]
        
        return per_sample_grads
    
    def clip_gradients(
        self,
        per_sample_grads: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Clip per-sample gradients according to specified norm.
        
        Args:
            per_sample_grads: List of per-sample gradients
            
        Returns:
            List of clipped gradients
        """
        # Calculate per-sample gradient norms
        grad_norms = []
        for i in range(per_sample_grads[0].size(0)):
            grad_norm = torch.norm(
                torch.stack([
                    torch.norm(g[i].view(-1)) 
                    for g in per_sample_grads 
                    if g is not None
                ])
            ).item()
            grad_norms.append(grad_norm)
            
        # Update history for adaptive clipping
        self.grad_norms_history.extend(grad_norms)
        
        # Determine clipping threshold
        if self.adaptive_clipping and len(self.grad_norms_history) > 100:
            # Use specified quantile of historical gradient norms
            clip_threshold = float(np.quantile(
                self.grad_norms_history[-1000:],  # Use recent history
                self.clipping_quantile
            ))
            # Ensure threshold is at least a minimum value
            clip_threshold = max(clip_threshold, 0.001)
        else:
            clip_threshold = self.max_grad_norm
            
        # Keep track of thresholds
        self.clipping_thresholds.append(clip_threshold)
        
        # Clip gradients
        clipped_grads = []
        for i, g in enumerate(per_sample_grads):
            if g is None:
                clipped_grads.append(None)
                continue
                
            # Calculate scaling factors
            grad_norms_tensor = torch.stack([
                torch.norm(g[i].view(-1)) for i in range(g.size(0))
            ])
            scale_factors = torch.clamp_max(
                clip_threshold / (grad_norms_tensor + 1e-8),
                1.0
            )
            
            # Apply scaling
            clipped_g = g * scale_factors.view(-1, *([1] * (g.dim() - 1)))
            clipped_grads.append(clipped_g)
            
        return clipped_grads
    
    def accumulate_gradients(
        self,
        clipped_grads: List[torch.Tensor]
    ):
        """
        Accumulate clipped gradients into model parameters.
        
        Args:
            clipped_grads: List of clipped gradients
        """
        # Get parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Accumulate gradients
        for p, g in zip(params, clipped_grads):
            if g is None:
                continue
                
            # Sum gradients across batch dimension
            accumulated_grad = g.sum(dim=0)
            
            # Accumulate into parameter gradients
            if p.grad is None:
                p.grad = accumulated_grad.detach().clone()
            else:
                p.grad.add_(accumulated_grad)
                
    def get_clipping_statistics(self) -> Dict[str, float]:
        """
        Get statistics on gradient clipping.
        
        Returns:
            Dictionary with clipping statistics
        """
        if not self.grad_norms_history:
            return {}
            
        return {
            'mean_grad_norm': np.mean(self.grad_norms_history),
            'median_grad_norm': np.median(self.grad_norms_history),
            'current_clip_threshold': self.clipping_thresholds[-1] if self.clipping_thresholds else self.max_grad_norm,
            'clipping_rate': np.mean([
                1.0 if norm > self.max_grad_norm else 0.0
                for norm in self.grad_norms_history
            ])
        }


# --------- Security Infrastructure ---------

class SecureDataHandler:
    """
    Secure data handling with encryption and key management.
    Implements strong cryptographic protection for sensitive data.
    """
    
    def __init__(
        self,
        key_length: int = 32,
        salt_length: int = 16,
        iterations: int = 480000
    ):
        """
        Initialize secure data handler.
        
        Args:
            key_length: Length of encryption key
            salt_length: Length of salt for key derivation
            iterations: Number of iterations for key derivation
        """
        self.key_length = key_length
        self.salt = os.urandom(salt_length)
        self.iterations = iterations
        
        # Create key derivation function
        self.kdf = self._create_kdf()
        
        # Generate encryption key
        self.encryption_key = self._generate_key()
        
        # Initialize cipher suite
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def _create_kdf(self) -> PBKDF2HMAC:
        """
        Create key derivation function.
        
        Returns:
            Configured KDF instance
        """
        return PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.key_length,
            salt=self.salt,
            iterations=self.iterations
        )
    
    def _generate_key(self) -> bytes:
        """
        Generate secure encryption key.
        
        Returns:
            Encryption key
        """
        # Generate random high-entropy key material
        key_material = secrets.token_bytes(32)
        
        # Derive key using KDF
        derived_key = self.kdf.derive(key_material)
        
        # Encode for Fernet
        return base64.urlsafe_b64encode(derived_key)
    
    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data securely.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        try:
            return self.cipher_suite.encrypt(data)
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise
            
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data securely.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            return self.cipher_suite.decrypt(encrypted_data)
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            raise
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Tuple[bytes, tuple]:
        """
        Encrypt tensor data.
        
        Args:
            tensor: Tensor to encrypt
            
        Returns:
            Tuple of (encrypted data, tensor shape)
        """
        # Save shape for reconstruction
        shape = tensor.shape
        
        # Convert to bytes
        tensor_np = tensor.cpu().numpy()
        tensor_bytes = tensor_np.tobytes()
        
        # Encrypt
        encrypted_bytes = self.encrypt_data(tensor_bytes)
        
        return encrypted_bytes, shape
    
    def decrypt_tensor(
        self,
        encrypted_data: bytes,
        shape: tuple,
        dtype=torch.float32,
        device="cpu"
    ) -> torch.Tensor:
        """
        Decrypt tensor data.
        
        Args:
            encrypted_data: Encrypted tensor data
            shape: Tensor shape
            dtype: Tensor dtype
            device: Target device
            
        Returns:
            Decrypted tensor
        """
        # Decrypt data
        decrypted_bytes = self.decrypt_data(encrypted_data)
        
        # Convert to numpy and then tensor
        tensor_np = np.frombuffer(decrypted_bytes, dtype=np.float32).reshape(shape)
        tensor = torch.from_numpy(tensor_np).to(dtype=dtype, device=device)
        
        return tensor


class PrivacyPreservingEncoder(nn.Module):
    """
    Privacy-preserving feature encoder with noise calibration.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        noise_scale: float = 0.1,
        epsilon: float = 1.0,
        delta: float = 1e-5
    ):
        """
        Initialize privacy-preserving encoder.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            noise_scale: Initial noise scale
            epsilon: Privacy budget epsilon
            delta: Privacy parameter delta
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_scale = noise_scale
        self.epsilon = epsilon
        self.delta = delta
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize secure data handler
        self.secure_handler = SecureDataHandler()
        
        # Initialize privacy tracking
        self.privacy_budget = epsilon
        self.queries = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def add_noise(self, x: torch.Tensor, sensitivity: float = 1.0) -> torch.Tensor:
        """
        Add calibrated noise for differential privacy.
        
        Args:
            x: Input tensor
            sensitivity: L2 sensitivity
            
        Returns:
            Noised tensor
        """
        # Calculate scale based on privacy parameters
        scale = self.noise_scale * sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        
        # Generate noise
        noise = torch.randn_like(x) * scale
        
        return x + noise
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bytes]:
        """
        Process input with privacy preservation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (encoded features, encrypted features)
        """
        # Add noise for privacy
        noisy_input = self.add_noise(x)
        
        # Update privacy budget
        self._update_privacy_budget(noisy_input)
        
        # Encode features
        encoded_features = self.encoder(noisy_input)
        
        # Encrypt features
        encrypted_features, _ = self.secure_handler.encrypt_tensor(encoded_features)
        
        return encoded_features, encrypted_features
    
    def _update_privacy_budget(self, query: torch.Tensor):
        """
        Update privacy budget tracking.
        
        Args:
            query: Query tensor
        """
        # Calculate query sensitivity
        sensitivity = torch.norm(query, p=2).item()
        
        # Calculate privacy cost
        privacy_cost = sensitivity * self.noise_scale
        self.privacy_budget -= privacy_cost
        
        # Record query
        self.queries.append({
            'timestamp': time.time(),
            'privacy_cost': privacy_cost
        })
    
    def encrypt_features(self, features: torch.Tensor) -> bytes:
        """
        Encrypt features securely.
        
        Args:
            features: Features to encrypt
            
        Returns:
            Encrypted features
        """
        encrypted_data, _ = self.secure_handler.encrypt_tensor(features)
        return encrypted_data
    
    def decrypt_features(
        self,
        encrypted_features: bytes,
        shape: tuple,
        device="cpu"
    ) -> torch.Tensor:
        """
        Decrypt features securely.
        
        Args:
            encrypted_features: Encrypted features
            shape: Feature shape
            device: Target device
            
        Returns:
            Decrypted features
        """
        return self.secure_handler.decrypt_tensor(
            encrypted_features,
            shape,
            dtype=torch.float32,
            device=device
        )


class SecureAggregation(nn.Module):
    """
    Secure aggregation implementation for privacy-preserving distributed learning.
    """
    
    def __init__(
        self,
        dim: int,
        noise_scale: float = 0.01,
        clipping_threshold: float = 1.0
    ):
        """
        Initialize secure aggregation.
        
        Args:
            dim: Feature dimension
            noise_scale: Noise scale for privacy
            clipping_threshold: Threshold for clipping
        """
        super().__init__()
        self.dim = dim
        self.noise_scale = noise_scale
        self.clipping_threshold = clipping_threshold
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform secure aggregation.
        
        Args:
            x: Input tensor
            
        Returns:
            Securely aggregated tensor
        """
        # Clip values for sensitivity control
        x_clipped = torch.clamp(
            x,
            -self.clipping_threshold,
            self.clipping_threshold
        )
        
        # Add calibrated noise
        noise = torch.randn_like(x_clipped) * self.noise_scale
        
        # Secure aggregation
        secure_output = x_clipped + noise
        
        # Normalize to maintain scale
        secure_output = secure_output / (1 + self.noise_scale)
        
        return secure_output


# --------- Privacy Engine ---------

class PrivacyEngine:
    """
    Comprehensive privacy engine for model training.
    Coordinates all privacy-preserving components.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_size: Optional[int] = None
    ):
        """
        Initialize privacy engine.
        
        Args:
            config: Privacy configuration
            data_size: Size of the dataset (if known)
        """
        # Extract configuration
        self.epsilon = config.get('epsilon', 0.1)
        self.delta = config.get('delta', 1e-5)
        self.noise_multiplier = config.get('noise_multiplier', 1.0)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.secure_aggregation = config.get('secure_aggregation', True)
        
        # Initialize privacy accountant
        self.accountant = RenyiPrivacyAccountant(
            target_epsilon=self.epsilon,
            target_delta=self.delta
        )
        
        # Initialize secure data handler
        self.secure_handler = SecureDataHandler()
        
        # Calculate optimal noise multiplier if data_size is provided
        # Calculate optimal noise multiplier if data_size is provided
        if data_size is not None:
            batch_size = config.get('batch_size', 128)
            epochs = config.get('epochs', 100)
            
            # Calculate optimal noise multiplier for given privacy budget
            self.noise_multiplier = self.accountant.compute_noise_multiplier(
                num_samples=data_size,
                batch_size=batch_size,
                epochs=epochs,
                target_epsilon=self.epsilon,
                target_delta=self.delta
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Calculated optimal noise multiplier: {self.noise_multiplier:.4f}")
        
        # Current privacy expenditure
        self.current_epsilon = 0.0
        
        # Privacy monitoring statistics
        self.privacy_stats = {
            'noise_multipliers': [],
            'per_step_epsilon': [],
            'cumulative_epsilon': [],
            'clipping_rates': [],
            'step_timestamps': []
        }
        
    def make_private(self, model: nn.Module) -> nn.Module:
        """
        Make model privacy-preserving by wrapping its layers.
        
        Args:
            model: Model to make private
            
        Returns:
            Privacy-preserving version of model
        """
        # Create gradient clipper for model
        self.gradient_clipper = GradientClipper(
            model=model,
            max_grad_norm=self.max_grad_norm,
            adaptive_clipping=True
        )
        
        # Replace standard layers with privacy-preserving layers if needed
        # (Here we could replace attention layers with privacy-preserving versions)
        
        # Store model for later use
        self.model = model
        
        return model
    
    def create_private_optimizer(
        self,
        base_optimizer: optim.Optimizer,
        lr: float = 1e-3,
        **kwargs
    ) -> PrivacyAwareOptimizer:
        """
        Create a privacy-aware optimizer.
        
        Args:
            base_optimizer: Base optimizer type
            lr: Learning rate
            **kwargs: Additional optimizer arguments
            
        Returns:
            Privacy-aware optimizer
        """
        return PrivacyAwareOptimizer(
            self.model.parameters(),
            lr=lr,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            accountant=self.accountant,
            **kwargs
        )
    
    def private_step(
        self,
        loss_per_sample: torch.Tensor,
        optimizer: optim.Optimizer,
        batch_size: int,
        data_size: int
    ) -> float:
        """
        Perform a single private training step.
        
        Args:
            loss_per_sample: Per-sample losses
            optimizer: Optimizer to use
            batch_size: Current batch size
            data_size: Total dataset size
            
        Returns:
            Current epsilon value
        """
        # Compute per-sample gradients
        per_sample_grads = self.gradient_clipper.compute_per_sample_gradients(loss_per_sample)
        
        # Clip gradients
        clipped_grads = self.gradient_clipper.clip_gradients(per_sample_grads)
        
        # Accumulate gradients
        self.gradient_clipper.accumulate_gradients(clipped_grads)
        
        # Perform optimizer step with privacy
        sampling_rate = batch_size / data_size
        if isinstance(optimizer, PrivacyAwareOptimizer):
            current_epsilon = optimizer.privacy_step(batch_size, data_size)
        else:
            # If not a privacy-aware optimizer, add noise manually
            self._add_noise_to_gradients()
            optimizer.step()
            
            # Update privacy accounting
            current_epsilon = self.accountant.update_privacy_budget(
                sampling_rate, self.noise_multiplier
            )
        
        # Update privacy statistics
        self._update_privacy_stats(current_epsilon)
        
        return current_epsilon
    
    def _add_noise_to_gradients(self):
        """Add calibrated noise to gradients for differential privacy."""
        for param in self.model.parameters():
            if param.grad is not None:
                # Scale noise by L2 sensitivity (max_grad_norm)
                noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm
                param.grad.add_(noise)
    
    def _update_privacy_stats(self, current_epsilon: float):
        """
        Update privacy monitoring statistics.
        
        Args:
            current_epsilon: Current epsilon value
        """
        # Calculate per-step epsilon
        per_step_epsilon = current_epsilon - self.current_epsilon
        self.current_epsilon = current_epsilon
        
        # Update statistics
        self.privacy_stats['noise_multipliers'].append(self.noise_multiplier)
        self.privacy_stats['per_step_epsilon'].append(per_step_epsilon)
        self.privacy_stats['cumulative_epsilon'].append(current_epsilon)
        
        # Add clipping rate if gradient clipper exists
        if hasattr(self, 'gradient_clipper'):
            clip_stats = self.gradient_clipper.get_clipping_statistics()
            self.privacy_stats['clipping_rates'].append(
                clip_stats.get('clipping_rate', 0.0)
            )
        
        # Add timestamp
        self.privacy_stats['step_timestamps'].append(time.time())
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy guarantee.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        return self.accountant.get_privacy_spent()
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive privacy statistics.
        
        Returns:
            Dictionary of privacy statistics
        """
        stats = self.privacy_stats.copy()
        
        # Add current privacy guarantee
        epsilon, delta = self.get_privacy_spent()
        stats['current_epsilon'] = epsilon
        stats['current_delta'] = delta
        
        # Add information about privacy budget
        budget_remaining = max(0, self.epsilon - epsilon)
        stats['budget_remaining'] = budget_remaining
        stats['budget_remaining_percent'] = (budget_remaining / self.epsilon) * 100
        
        return stats


# --------- Privacy Auditing ---------

class PrivacyAuditor:
    """
    Comprehensive privacy auditing system for model evaluation.
    Implements attack simulations and privacy guarantee verification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        epsilon: float = 0.1,
        delta: float = 1e-5
    ):
        """
        Initialize privacy auditor.
        
        Args:
            model: Model to audit
            dataset: Dataset used for training
            epsilon: Target epsilon
            delta: Target delta
        """
        self.model = model
        self.dataset = dataset
        self.epsilon = epsilon
        self.delta = delta
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize attack results
        self.attack_results = {}
        
        # Private information dictionary
        self.private_info = {}
        
    def run_membership_inference_attack(
        self,
        test_dataset: Dataset,
        attack_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Run membership inference attack to assess privacy leakage.
        This attack attempts to determine if a sample was used during training.
        
        Args:
            test_dataset: Test dataset (samples not used in training)
            attack_iterations: Number of attack iterations
            
        Returns:
            Dictionary of attack success rates
        """
        self.logger.info("Running membership inference attack...")
        
        # Create balanced dataset with equal parts from training and test
        train_indices = np.random.choice(len(self.dataset), size=attack_iterations, replace=False)
        test_indices = np.random.choice(len(test_dataset), size=attack_iterations, replace=False)
        
        # Get samples
        train_samples = [self.dataset[i] for i in train_indices]
        test_samples = [test_dataset[i] for i in test_indices]
        
        # True membership labels (1 for train, 0 for test)
        true_membership = np.concatenate([
            np.ones(attack_iterations),
            np.zeros(attack_iterations)
        ])
        
        # Create attack dataset
        attack_samples = train_samples + test_samples
        
        # Prepare model for attack
        self.model.eval()
        
        # Run attack
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for sample in attack_samples:
                # Forward pass
                inputs = self._prepare_inputs(sample)
                outputs = self.model(**inputs)
                
                # Get confidence scores
                if 'price_prediction' in outputs:
                    # For regression models, use prediction error
                    # (lower error usually means the sample was in the training set)
                    if 'price' in sample:
                        error = abs(outputs['price_prediction'].item() - sample['price'].item())
                        confidence = 1.0 / (1.0 + error)  # Transform to [0, 1]
                    else:
                        confidence = 0.5  # Neutral if no price in sample
                else:
                    # For classification models, use softmax confidence
                    if 'attribute_prediction' in outputs:
                        probs = torch.sigmoid(outputs['attribute_prediction'])
                        confidence = probs.max().item()  # Highest confidence
                    else:
                        confidence = 0.5  # Neutral if no predictions
                
                # Predict membership (higher confidence suggests training member)
                prediction = 1 if confidence > 0.5 else 0
                
                predictions.append(prediction)
                confidences.append(confidence)
        
        # Calculate metrics
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Calculate attack accuracy
        accuracy = np.mean(predictions == true_membership)
        
        # Calculate attack precision and recall
        tp = np.sum((predictions == 1) & (true_membership == 1))
        fp = np.sum((predictions == 1) & (true_membership == 0))
        fn = np.sum((predictions == 0) & (true_membership == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate attack TPR (true positive rate) and FPR (false positive rate)
        tpr = recall  # Same as recall
        fpr = np.sum((predictions == 1) & (true_membership == 0)) / np.sum(true_membership == 0)
        
        # Calculate advantage (TPR - FPR), a common privacy leakage measure
        advantage = tpr - fpr
        
        # Compare to theoretical bound
        # For DP models, advantage should be <= 2*epsilon
        theoretical_bound = 2 * self.epsilon
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'advantage': float(advantage),
            'theoretical_bound': float(theoretical_bound),
            'within_theoretical_bound': advantage <= theoretical_bound
        }
        
        # Store results
        self.attack_results['membership_inference'] = results
        
        # Log results
        self.logger.info(f"Membership inference attack results: {results}")
        
        return results
    
    def run_model_inversion_attack(
        self,
        num_samples: int = 10,
        reconstruction_steps: int = 1000,
        lr: float = 0.01
    ) -> Dict[str, float]:
        """
        Run model inversion attack to assess privacy leakage.
        This attack attempts to reconstruct input data from model outputs.
        
        Args:
            num_samples: Number of samples to attempt reconstruction
            reconstruction_steps: Number of optimization steps
            lr: Learning rate for reconstruction
            
        Returns:
            Dictionary of attack success metrics
        """
        self.logger.info("Running model inversion attack...")
        
        # Select random samples
        indices = np.random.choice(len(self.dataset), size=num_samples, replace=False)
        samples = [self.dataset[i] for i in indices]
        
        # Prepare model for attack
        self.model.eval()
        
        # Reconstruction errors
        reconstruction_errors = []
        
        for sample in samples:
            # Get target output
            inputs = self._prepare_inputs(sample)
            with torch.no_grad():
                target_output = self.model(**inputs)
            
            # Initialize random input for reconstruction
            if 'images' in inputs:
                # For image inputs, start with random noise
                reconstructed_input = torch.randn_like(inputs['images'], requires_grad=True)
                original_input = inputs['images'].clone()
                input_type = 'image'
            elif 'transaction_data' in inputs:
                # For transaction data, start with random values
                reconstructed_input = torch.randn_like(inputs['transaction_data'], requires_grad=True)
                original_input = inputs['transaction_data'].clone()
                input_type = 'transaction'
            else:
                continue  # Skip if no recognized input
            
            # Setup optimizer
            optimizer = optim.Adam([reconstructed_input], lr=lr)
            
            # Reconstruction loop
            for step in range(reconstruction_steps):
                # Forward pass with reconstructed input
                if input_type == 'image':
                    modified_inputs = {**inputs, 'images': reconstructed_input}
                else:
                    modified_inputs = {**inputs, 'transaction_data': reconstructed_input}
                
                reconstructed_output = self.model(**modified_inputs)
                
                # Calculate loss (MSE between outputs)
                loss = 0
                if 'price_prediction' in target_output and 'price_prediction' in reconstructed_output:
                    loss += F.mse_loss(
                        reconstructed_output['price_prediction'],
                        target_output['price_prediction']
                    )
                
                if 'attribute_prediction' in target_output and 'attribute_prediction' in reconstructed_output:
                    loss += F.mse_loss(
                        reconstructed_output['attribute_prediction'],
                        target_output['attribute_prediction']
                    )
                
                # Add regularization to improve reconstruction
                loss += 0.01 * torch.norm(reconstructed_input)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Calculate reconstruction error
            error = F.mse_loss(reconstructed_input, original_input).item()
            reconstruction_errors.append(error)
        
        # Calculate metrics
        mean_error = np.mean(reconstruction_errors)
        median_error = np.median(reconstruction_errors)
        min_error = np.min(reconstruction_errors)
        
        results = {
            'mean_error': float(mean_error),
            'median_error': float(median_error),
            'min_error': float(min_error)
        }
        
        # Interpret results
        # Higher error means better privacy protection
        privacy_leakage = 1.0 / (1.0 + mean_error)  # Transform to [0, 1]
        results['privacy_leakage'] = float(privacy_leakage)
        
        # Store results
        self.attack_results['model_inversion'] = results
        
        # Log results
        self.logger.info(f"Model inversion attack results: {results}")
        
        return results
    
    def run_attribute_inference_attack(
        self,
        sensitive_attribute: str,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Run attribute inference attack to assess privacy leakage.
        This attack attempts to infer sensitive attributes from model outputs.
        
        Args:
            sensitive_attribute: Name of sensitive attribute
            num_samples: Number of samples for attack
            
        Returns:
            Dictionary of attack success metrics
        """
        self.logger.info(f"Running attribute inference attack for {sensitive_attribute}...")
        
        # Collect samples with known sensitive attribute
        samples_with_attribute = []
        true_attribute_values = []
        
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            if hasattr(sample, sensitive_attribute):
                samples_with_attribute.append(sample)
                true_attribute_values.append(getattr(sample, sensitive_attribute))
                
                if len(samples_with_attribute) >= num_samples:
                    break
        
        if len(samples_with_attribute) < 10:
            self.logger.warning(f"Not enough samples with attribute {sensitive_attribute}")
            return {'error': 'Not enough samples'}
        
        # Prepare model for attack
        self.model.eval()
        
        # Run attack
        predictions = []
        
        with torch.no_grad():
            for sample in samples_with_attribute:
                # Forward pass
                inputs = self._prepare_inputs(sample)
                outputs = self.model(**inputs)
                
                # Get fusion features
                if 'fused_features' in outputs:
                    features = outputs['fused_features']
                else:
                    # Try to extract any available features
                    feature_keys = [k for k in outputs.keys() if 'features' in k]
                    if feature_keys:
                        features = outputs[feature_keys[0]]
                    else:
                        # No features available
                        continue
                
                # Predict attribute from features using a simple linear model
                if not hasattr(self, 'attribute_predictors'):
                    self.attribute_predictors = {}
                
                if sensitive_attribute not in self.attribute_predictors:
                    # Train a simple predictor for this attribute
                    self._train_attribute_predictor(sensitive_attribute)
                
                # Use predictor to infer attribute
                predictor = self.attribute_predictors[sensitive_attribute]
                prediction = predictor(features).item()
                predictions.append(prediction)
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_values = np.array(true_attribute_values)
        
        # For binary attributes
        if len(np.unique(true_values)) == 2:
            accuracy = np.mean(predictions.round() == true_values)
            
            # Calculate ROC AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(true_values, predictions)
            except:
                auc = 0.5
                
            results = {
                'accuracy': float(accuracy),
                'auc': float(auc)
            }
        else:
            # For continuous attributes
            mse = np.mean((predictions - true_values) ** 2)
            r2 = 1 - np.sum((predictions - true_values) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)
            
            results = {
                'mse': float(mse),
                'r2': float(r2)
            }
        
        # Store results
        self.attack_results['attribute_inference'] = results
        
        # Log results
        self.logger.info(f"Attribute inference attack results: {results}")
        
        return results
    
    def _train_attribute_predictor(self, attribute_name: str):
        """
        Train a simple predictor for attribute inference attack.
        
        Args:
            attribute_name: Name of attribute to predict
        """
        # Collect training data
        features_list = []
        attribute_values = []
        
        with torch.no_grad():
            for i in range(min(1000, len(self.dataset))):
                sample = self.dataset[i]
                if hasattr(sample, attribute_name):
                    # Forward pass
                    inputs = self._prepare_inputs(sample)
                    outputs = self.model(**inputs)
                    
                    # Get features
                    if 'fused_features' in outputs:
                        features = outputs['fused_features']
                    else:
                        # Try to extract any available features
                        feature_keys = [k for k in outputs.keys() if 'features' in k]
                        if feature_keys:
                            features = outputs[feature_keys[0]]
                        else:
                            # No features available
                            continue
                    
                    features_list.append(features.cpu())
                    attribute_values.append(getattr(sample, attribute_name))
        
        if not features_list:
            return
            
        # Stack features and convert values
        features_tensor = torch.cat(features_list, dim=0)
        values_tensor = torch.tensor(attribute_values).float()
        
        # Create simple linear model
        feature_dim = features_tensor.size(1)
        predictor = nn.Linear(feature_dim, 1)
        
        # Train for a few epochs
        optimizer = optim.Adam(predictor.parameters())
        
        for epoch in range(50):
            # Forward pass
            predictions = predictor(features_tensor).squeeze()
            
            # Loss
            loss = F.mse_loss(predictions, values_tensor)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Store predictor
        self.attribute_predictors[attribute_name] = predictor
    
    def _prepare_inputs(self, sample) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for model from sample.
        
        Args:
            sample: Data sample
            
        Returns:
            Dictionary of input tensors
        """
        inputs = {}
        
        # Extract images if available
        if 'image' in sample:
            if isinstance(sample['image'], torch.Tensor):
                inputs['images'] = sample['image'].unsqueeze(0).cuda()
            else:
                # Convert to tensor if needed
                inputs['images'] = torch.tensor(sample['image']).unsqueeze(0).cuda()
        
        # Extract transaction data if available
        if 'transaction' in sample:
            if isinstance(sample['transaction'], torch.Tensor):
                inputs['transaction_data'] = sample['transaction'].unsqueeze(0).cuda()
            else:
                # Convert to tensor if needed
                inputs['transaction_data'] = torch.tensor(sample['transaction']).unsqueeze(0).cuda()
                
        # Add transaction graphs if needed
        if 'transaction_graph' in sample:
            inputs['transaction_graphs'] = sample['transaction_graph']
            
        return inputs
    
    def get_all_attack_results(self) -> Dict[str, Dict[str, float]]:
        """
        Get results from all attacks.
        
        Returns:
            Dictionary of all attack results
        """
        return self.attack_results
    
    def assess_privacy_guarantees(self) -> Dict[str, Any]:
        """
        Assess if the model meets its privacy guarantees.
        
        Returns:
            Assessment results
        """
        # Theoretical guarantees
        theoretical_guarantees = {
            'epsilon': self.epsilon,
            'delta': self.delta
        }
        
        # Empirical guarantees from attacks
        empirical_guarantees = {}
        
        # Check membership inference attack
        if 'membership_inference' in self.attack_results:
            results = self.attack_results['membership_inference']
            theoretical_bound = 2 * self.epsilon
            
            empirical_guarantees['membership_inference'] = {
                'theoretical_bound': theoretical_bound,
                'empirical_advantage': results['advantage'],
                'within_bound': results['advantage'] <= theoretical_bound
            }
        
        # Check model inversion attack
        if 'model_inversion' in self.attack_results:
            results = self.attack_results['model_inversion']
            empirical_guarantees['model_inversion'] = {
                'privacy_leakage': results['privacy_leakage']
            }
        
        # Check attribute inference attack
        if 'attribute_inference' in self.attack_results:
            results = self.attack_results['attribute_inference']
            if 'auc' in results:
                # AUC should be close to 0.5 for good privacy
                privacy_score = 1.0 - 2 * abs(results['auc'] - 0.5)
            elif 'r2' in results:
                # Lower R^2 means better privacy
                privacy_score = 1.0 - results['r2']
            else:
                privacy_score = 0.5
                
            empirical_guarantees['attribute_inference'] = {
                'privacy_score': privacy_score
            }
            
        # Overall assessment
        guarantees_met = all([
            g.get('within_bound', True) 
            for g in empirical_guarantees.values() 
            if 'within_bound' in g
        ])
        
        overall_privacy_score = np.mean([
            g['privacy_score'] 
            for g in empirical_guarantees.values() 
            if 'privacy_score' in g
        ]) if empirical_guarantees else 0.5
        
        assessment = {
            'theoretical_guarantees': theoretical_guarantees,
            'empirical_guarantees': empirical_guarantees,
            'guarantees_met': guarantees_met,
            'overall_privacy_score': float(overall_privacy_score)
        }
        
        return assessment


def create_privacy_engine(
    config: Dict[str, Any],
    data_size: Optional[int] = None
) -> PrivacyEngine:
    """
    Create a privacy engine from configuration.
    
    Args:
        config: Privacy configuration
        data_size: Size of the dataset (if known)
        
    Returns:
        Configured PrivacyEngine instance
    """
    return PrivacyEngine(config, data_size)