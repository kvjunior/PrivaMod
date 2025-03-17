"""
PrivaMod Neural Models
==================

Implementation of all neural network architectures for the PrivaMod system:

1. Visual Encoders:
   - ContrastiveVisualEncoder: Self-supervised visual representation learning
   - VisionTransformerEncoder: Advanced vision transformer for NFT images
   - CompositeAttributeAnalyzer: Compositional reasoning for NFT attributes

2. Transaction Encoders:
   - TemporalGraphNN: Graph-based transaction sequence analysis
   - TransactionLongformer: Long-range transaction history modeling

3. Fusion Modules:
   - BayesianMultimodalFusion: Principled uncertainty-aware fusion
   - InfoTheoreticFusion: Information-theoretic approach to optimal fusion

4. Complete PrivaMod Architecture:
   - PrivaModSystem: End-to-end implementation of the complete system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

# Make sure to install the following libraries in requirements.txt:
# transformers, timm, dgl, torch_geometric

import timm
from transformers import LongformerModel, LongformerConfig
import dgl
import dgl.nn as dglnn
from torch_geometric.nn import GCNConv, GATConv

# --------- Visual Encoders ---------

class ContrastiveVisualEncoder(nn.Module):
    """
    Self-supervised contrastive visual encoder with projection head.
    Implements momentum-based contrastive learning similar to MoCo v2.
    """
    
    def __init__(
        self, 
        backbone='resnet50', 
        projection_dim=128,
        pretrained=True,
        temperature=0.07,
        queue_size=65536,
        momentum=0.999
    ):
        """
        Initialize contrastive visual encoder.
        
        Args:
            backbone: Backbone architecture (from timm)
            projection_dim: Dimension of projection space
            pretrained: Whether to use pretrained weights
            temperature: Temperature for contrastive loss
            queue_size: Size of memory queue
            momentum: Momentum for key encoder update
        """
        super().__init__()
        # Load backbone with pretrained weights
        self.encoder_q = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.encoder_k = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension from backbone
        feature_dim = self.encoder_q.num_features
        
        # Create projection heads
        self.projector_q = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        
        self.projector_k = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        
        # Copy parameters from query encoder to key encoder
        for param_q, param_k in zip(
            list(self.encoder_q.parameters()) + list(self.projector_q.parameters()),
            list(self.encoder_k.parameters()) + list(self.projector_k.parameters())
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        # Create queue
        self.register_buffer("queue", torch.randn(queue_size, projection_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Save hyperparameters
        self.temperature = temperature
        self.momentum = momentum
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update key encoder using momentum."""
        for param_q, param_k in zip(
            list(self.encoder_q.parameters()) + list(self.projector_q.parameters()),
            list(self.encoder_k.parameters()) + list(self.projector_k.parameters())
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue with new keys."""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue.shape[0]:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Handle wrap-around
            remainder = self.queue.shape[0] - ptr
            self.queue[ptr:] = keys[:remainder]
            self.queue[:batch_size - remainder] = keys[remainder:]
            
        ptr = (ptr + batch_size) % self.queue.shape[0]
        self.queue_ptr[0] = ptr
        
    def forward(self, x, compute_key=True):
        """
        Forward pass through contrastive encoder.
        
        Args:
            x: Input image batch
            compute_key: Whether to compute key representation (for inference, set to False)
            
        Returns:
            query_features: Raw features from query encoder
            query_proj: Normalized projection of query features
            logits: Contrastive prediction logits (if compute_key=True)
            labels: Contrastive prediction targets (if compute_key=True)
        """
        # Extract query features
        query_features = self.encoder_q(x)
        query_proj = self.projector_q(query_features)
        query_proj = F.normalize(query_proj, dim=1)
        
        if not compute_key:
            return query_features, query_proj, None, None
            
        # Compute key features with no gradient
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            # Shuffle batch for key encoder
            idx_shuffle = torch.randperm(x.shape[0], device=x.device)
            key_img = x[idx_shuffle]
            
            key_features = self.encoder_k(key_img)
            key_proj = self.projector_k(key_features)
            key_proj = F.normalize(key_proj, dim=1)
            
            # Undo shuffle
            idx_unshuffle = torch.argsort(idx_shuffle)
            key_proj = key_proj[idx_unshuffle]
            
        # Compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [query_proj, key_proj]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,kc->nk', [query_proj, self.queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels: positives are the 0-th
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(key_proj)
        
        return query_features, query_proj, logits, labels


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer with MLP-Mixer for visual feature extraction.
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        dropout=0.1,
        attn_dropout=0.1,
        mixer_tokens=True,
        mixer_channels=True
    ):
        """
        Initialize Vision Transformer with MLP-Mixer.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            embed_dim: Embedding dimension
            depth: Depth of transformer
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            mixer_tokens: Whether to use token mixing MLP
            mixer_channels: Whether to use channel mixing MLP
        """
        super().__init__()
        
        # Use timm's ViT implementation with pretrained weights
        self.vit = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=dropout,
            attn_drop_rate=attn_dropout
        )
        
        # Token mixer (MLP-Mixer style)
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        ) if mixer_tokens else nn.Identity()
        
        # Channel mixer (MLP-Mixer style)
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        ) if mixer_channels else nn.Identity()
        
        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Forward pass through ViT + MLP-Mixer.
        
        Args:
            x: Input image batch [B, C, H, W]
            
        Returns:
            x: Output feature tensor [B, embed_dim]
        """
        # Extract patch tokens
        x = self.vit.forward_features(x)  # [B, num_patches, embed_dim]
        
        # Apply token mixing across patches (if enabled)
        if not isinstance(self.token_mixer, nn.Identity):
            # Transpose for token mixing
            x_t = x.transpose(1, 2)  # [B, embed_dim, num_patches]
            x = x + self.token_mixer(x_t).transpose(1, 2)
        
        # Apply channel mixing (if enabled)
        if not isinstance(self.channel_mixer, nn.Identity):
            x = x + self.channel_mixer(x)
        
        # Global pooling and normalization
        x = x.mean(dim=1)  # [B, embed_dim]
        x = self.norm(x)
        
        return x


class CompositeAttributeAnalyzer(nn.Module):
    """
    Compositional visual reasoning for NFT attributes.
    """
    
    def __init__(
        self,
        num_attributes=100,
        backbone='resnet50',
        embedding_dim=256,
        num_transformer_layers=3,
        num_attention_heads=8,
        pretrained=True
    ):
        """
        Initialize CompositeAttributeAnalyzer.
        
        Args:
            num_attributes: Number of possible attributes
            backbone: Backbone architecture (from timm)
            embedding_dim: Dimension of attribute embeddings
            num_transformer_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Attribute embeddings
        self.attribute_embeddings = nn.Embedding(num_attributes, embedding_dim)
        
        # Attribute detector CNN
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        # Detection head
        self.attribute_detector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 2, num_attributes)
        )
        
        # Transformer encoder for compositional reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.composition_transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, images):
        """
        Process images through compositional attribute analyzer.
        
        Args:
            images: Input image batch [B, C, H, W]
            
        Returns:
            compositional_features: Features capturing attribute compositions
            attribute_probs: Probabilities for each attribute
        """
        # Extract image features
        image_features = self.backbone(images)
        
        # Detect attributes
        attribute_logits = self.attribute_detector(image_features)
        attribute_probs = torch.sigmoid(attribute_logits)
        
        # Get embeddings for detected attributes
        # weighted_embeddings: [B, num_attributes, embedding_dim]
        weighted_embeddings = attribute_probs.unsqueeze(-1) * self.attribute_embeddings.weight.unsqueeze(0)
        
        # Sum embeddings along attribute dimension
        # This creates a single embedding per image that is a weighted sum of attribute embeddings
        # This approach uses "soft" attribute detection, rather than hard decisions
        summed_embeddings = weighted_embeddings.sum(dim=1)  # [B, embedding_dim]
        
        # Process through transformer for compositional reasoning
        # Reshape to [B, 1, embedding_dim] for transformer
        transformer_input = summed_embeddings.unsqueeze(1)
        
        # Convert to [1, B, embedding_dim] for transformer (seq_len, batch, features)
        transformer_input = transformer_input.transpose(0, 1)
        
        # Apply transformer to model interactions between attributes
        compositional_features = self.composition_transformer(transformer_input)
        
        # Reshape back to [B, embedding_dim]
        compositional_features = compositional_features.transpose(0, 1).squeeze(1)
        
        # Final projection
        compositional_features = self.output_projection(compositional_features)
        
        return compositional_features, attribute_probs


# --------- Transaction Encoders ---------

class TemporalGraphNN(nn.Module):
    """
    Temporal Graph Neural Network for transaction data.
    Combines graph structure with temporal information.
    """
    
    def __init__(
        self,
        feature_dim=128,
        hidden_dim=256,
        output_dim=512,
        num_gnn_layers=2,
        gnn_type='gcn',
        temporal_encoder_type='gru',
        num_temporal_layers=1,
        dropout=0.1
    ):
        """
        Initialize Temporal Graph Neural Network.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_gnn_layers: Number of GNN layers
            gnn_type: Type of GNN ('gcn' or 'gat')
            temporal_encoder_type: Type of temporal encoder ('gru' or 'transformer')
            num_temporal_layers: Number of temporal encoder layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        if gnn_type == 'gcn':
            # Graph Convolutional Network
            for i in range(num_gnn_layers):
                in_dim = hidden_dim if i > 0 else hidden_dim
                self.gnn_layers.append(GCNConv(in_dim, hidden_dim))
        elif gnn_type == 'gat':
            # Graph Attention Network
            for i in range(num_gnn_layers):
                in_dim = hidden_dim if i > 0 else hidden_dim
                self.gnn_layers.append(GATConv(in_dim, hidden_dim // 8, heads=8))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
            
        # Temporal encoder
        if temporal_encoder_type == 'gru':
            self.temporal_encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout if num_temporal_layers > 1 else 0
            )
        elif temporal_encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_temporal_layers
            )
        else:
            raise ValueError(f"Unknown temporal encoder type: {temporal_encoder_type}")
            
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Temporal encoder type
        self.temporal_encoder_type = temporal_encoder_type
        
    def forward(self, transaction_graphs, transaction_sequences):
        """
        Process transaction data through temporal GNN.
        
        Args:
            transaction_graphs: Batch of transaction graphs
            transaction_sequences: Batch of transaction sequences [B, T, F]
            
        Returns:
            combined_features: Combined graph and temporal features
        """
        batch_size = transaction_sequences.size(0)
        
        # Process graph structure
        graph_features = transaction_graphs.ndata['features']
        graph_features = self.input_projection(graph_features)
        
        for layer in self.gnn_layers:
            graph_features = F.relu(layer(transaction_graphs, graph_features))
            
        # Extract node features for each instance in the batch
        batch_graph_features = []
        for i in range(batch_size):
            # Get nodes corresponding to this instance
            mask = transaction_graphs.ndata['batch'] == i
            instance_features = graph_features[mask]
            
            # Pool node features (mean pooling)
            instance_features = instance_features.mean(dim=0)
            batch_graph_features.append(instance_features)
            
        # Stack features
        batch_graph_features = torch.stack(batch_graph_features, dim=0)
        batch_graph_features = self.norm1(batch_graph_features)
        
        # Process temporal sequences
        # Project to hidden dimension
        temporal_features = self.input_projection(transaction_sequences)
        
        if self.temporal_encoder_type == 'gru':
            # GRU encoder
            _, temporal_features = self.temporal_encoder(temporal_features)
            # Get last layer output
            temporal_features = temporal_features[-1]
        else:
            # Transformer encoder
            # Reshape to [T, B, F]
            temporal_features = temporal_features.transpose(0, 1)
            temporal_features = self.temporal_encoder(temporal_features)
            # Use mean pooling over time dimension
            temporal_features = temporal_features.mean(dim=0)
            
        temporal_features = self.norm2(temporal_features)
        
        # Combine graph and temporal features
        combined_features = torch.cat([batch_graph_features, temporal_features], dim=1)
        combined_features = self.output_projection(combined_features)
        
        return combined_features


class TransactionLongformer(nn.Module):
    """
    Longformer for efficient processing of long transaction sequences.
    """
    
    def __init__(
        self,
        input_dim=128,
        hidden_dim=768,
        output_dim=512,
        max_seq_len=4096,
        num_hidden_layers=4,
        num_attention_heads=8,
        attention_window=512,
        dropout=0.1
    ):
        """
        Initialize Longformer for transaction sequences.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            max_seq_len: Maximum sequence length
            num_hidden_layers: Number of hidden layers
            num_attention_heads: Number of attention heads
            attention_window: Size of attention window
            dropout: Dropout rate
        """
        super().__init__()
        
        # Embedding layer to project inputs to hidden dimension
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional embeddings
        self.pos_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        # Create custom Longformer configuration
        config = LongformerConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_dim * 4,
            attention_window=[attention_window] * num_hidden_layers,
            max_position_embeddings=max_seq_len,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        
        # Initialize Longformer with custom config
        self.longformer = LongformerModel(config)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embeddings, std=0.02)
        
    def forward(self, transaction_sequence, attention_mask=None):
        """
        Process transaction sequence through Longformer.
        
        Args:
            transaction_sequence: Batch of transaction sequences [B, T, F]
            attention_mask: Optional attention mask [B, T]
            
        Returns:
            sequence_features: Transaction sequence features
        """
        batch_size, seq_len, _ = transaction_sequence.shape
        
        # Create embedding
        embedded_seq = self.embedding(transaction_sequence)
        
        # Add positional embeddings
        embedded_seq = embedded_seq + self.pos_embeddings[:, :seq_len, :]
        
        # Create attention mask if not provided
        if attention_mask is None:
            # Mask based on zero padding
            attention_mask = (transaction_sequence.sum(dim=-1) != 0).float()
        
        # Convert to Longformer global attention format
        # Set CLS token to global attention
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1  # global attention for CLS token
        
        # Process with Longformer
        outputs = self.longformer(
            inputs_embeds=embedded_seq,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        
        # Get sequence representation from CLS token
        sequence_features = outputs.last_hidden_state[:, 0]
        
        # Final projection
        sequence_features = self.output_projection(sequence_features)
        
        return sequence_features


# --------- Fusion Modules ---------

class BayesianMultimodalFusion(nn.Module):
    """
    Bayesian multimodal fusion with uncertainty modeling.
    """
    
    def __init__(
        self,
        visual_dim=512,
        transaction_dim=512,
        fusion_dim=512,
        dropout=0.1
    ):
        """
        Initialize Bayesian multimodal fusion.
        
        Args:
            visual_dim: Visual feature dimension
            transaction_dim: Transaction feature dimension
            fusion_dim: Fusion dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Visual feature processing
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Transaction feature processing
        self.transaction_encoder = nn.Sequential(
            nn.Linear(transaction_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Uncertainty estimation
        self.visual_uncertainty = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.transaction_uncertainty = nn.Sequential(
            nn.Linear(transaction_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, visual_features, transaction_features):
        """
        Fuse visual and transaction features with uncertainty modeling.
        
        Args:
            visual_features: Visual features [B, visual_dim]
            transaction_features: Transaction features [B, transaction_dim]
            
        Returns:
            fused_features: Fused features [B, fusion_dim]
            visual_mean: Visual mean features [B, fusion_dim]
            transaction_mean: Transaction mean features [B, fusion_dim]
            visual_logvar: Log variance for visual features [B, fusion_dim]
            transaction_logvar: Log variance for transaction features [B, fusion_dim]
        """
        # Get mean representations
        visual_mean = self.visual_encoder(visual_features)
        transaction_mean = self.transaction_encoder(transaction_features)
        
        # Estimate log variance (uncertainty)
        visual_logvar = self.visual_uncertainty(visual_features)
        transaction_logvar = self.transaction_uncertainty(transaction_features)
        
        # Precision-weighted fusion (more weight to more certain modality)
        visual_precision = torch.exp(-visual_logvar)
        transaction_precision = torch.exp(-transaction_logvar)
        
        # Weighted average based on precision
        total_precision = visual_precision + transaction_precision
        fused_features = (visual_mean * visual_precision + 
                          transaction_mean * transaction_precision) / (total_precision + 1e-8)
        
        # Final projection
        fused_features = self.output_projection(fused_features)
        
        return fused_features, visual_mean, transaction_mean, visual_logvar, transaction_logvar


class InfoTheoreticFusion(nn.Module):
    """
    Information-theoretic multimodal fusion.
    """
    
    def __init__(
        self,
        visual_dim=512,
        transaction_dim=512,
        fusion_dim=512,
        temperature=0.07,
        dropout=0.1
    ):
        """
        Initialize information-theoretic fusion.
        
        Args:
            visual_dim: Visual feature dimension
            transaction_dim: Transaction feature dimension
            fusion_dim: Fusion dimension
            temperature: Temperature for InfoNCE loss
            dropout: Dropout rate
        """
        super().__init__()
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Transaction encoder
        self.transaction_encoder = nn.Sequential(
            nn.Linear(transaction_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Mutual information estimator network
        self.mi_estimator = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1)
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Temperature for InfoNCE
        self.temperature = temperature
        
    def forward(self, visual_features, transaction_features):
        """
        Fuse visual and transaction features using information-theoretic approach.
        
        Args:
            visual_features: Visual features [B, visual_dim]
            transaction_features: Transaction features [B, transaction_dim]
            
        Returns:
            fused_features: Fused features [B, fusion_dim]
            mi_loss: Mutual information loss
        """
        batch_size = visual_features.size(0)
        
        # Encode features
        visual_z = self.visual_encoder(visual_features)
        transaction_z = self.transaction_encoder(transaction_features)
        
        # Estimate mutual information (InfoNCE objective)
        # Create joint samples (positive pairs)
        joint = torch.cat([visual_z, transaction_z], dim=1)
        
        # Create shuffled pairs (negative pairs)
        transaction_z_shuffled = transaction_z[torch.randperm(batch_size)]
        marginal = torch.cat([visual_z, transaction_z_shuffled], dim=1)
        
        # Estimate mutual information
        joint_scores = self.mi_estimator(joint)
        marginal_scores = self.mi_estimator(marginal)
        
        # InfoNCE loss
        mi_loss = F.binary_cross_entropy_with_logits(
            joint_scores, 
            torch.ones_like(joint_scores)
        ) + F.binary_cross_entropy_with_logits(
            marginal_scores, 
            torch.zeros_like(marginal_scores)
        )
        
        # Dynamic fusion weights based on mutual information
        alpha = torch.sigmoid(joint_scores)
        
        # Weighted combination plus direct fusion network
        combined_features = torch.cat([visual_z, transaction_z], dim=1)
        fused_features = self.fusion_network(combined_features)
        
        return fused_features, mi_loss


# --------- Complete PrivaMod System ---------

class PrivaModSystem(nn.Module):
    """
    Complete end-to-end PrivaMod system.
    """
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """
        Initialize PrivaMod system with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Extract configuration
        visual_config = config.get('visual', {})
        transaction_config = config.get('transaction', {})
        fusion_config = config.get('fusion', {})
        prediction_config = config.get('prediction', {})
        
        # Visual encoder selection
        visual_encoder_type = visual_config.get('type', 'vit')
        if visual_encoder_type == 'contrastive':
            self.visual_encoder = ContrastiveVisualEncoder(
                backbone=visual_config.get('backbone', 'resnet50'),
                projection_dim=visual_config.get('projection_dim', 128),
                pretrained=visual_config.get('pretrained', True),
                temperature=visual_config.get('temperature', 0.07),
                queue_size=visual_config.get('queue_size', 65536),
                momentum=visual_config.get('momentum', 0.999)
            )
        elif visual_encoder_type == 'vit':
            self.visual_encoder = VisionTransformerEncoder(
                img_size=visual_config.get('img_size', 224),
                patch_size=visual_config.get('patch_size', 16),
                in_chans=visual_config.get('in_chans', 3),
                embed_dim=visual_config.get('embed_dim', 768),
                depth=visual_config.get('depth', 12),
                num_heads=visual_config.get('num_heads', 12),
                mlp_ratio=visual_config.get('mlp_ratio', 4.),
                dropout=visual_config.get('dropout', 0.1),
                attn_dropout=visual_config.get('attn_dropout', 0.1),
                mixer_tokens=visual_config.get('mixer_tokens', True),
                mixer_channels=visual_config.get('mixer_channels', True)
            )
        elif visual_encoder_type == 'attribute':
            self.visual_encoder = CompositeAttributeAnalyzer(
                num_attributes=visual_config.get('num_attributes', 100),
                backbone=visual_config.get('backbone', 'resnet50'),
                embedding_dim=visual_config.get('embedding_dim', 256),
                num_transformer_layers=visual_config.get('num_transformer_layers', 3),
                num_attention_heads=visual_config.get('num_attention_heads', 8),
                pretrained=visual_config.get('pretrained', True)
            )
        else:
            raise ValueError(f"Unknown visual encoder type: {visual_encoder_type}")
            
        # Extract visual dimension
        if visual_encoder_type == 'contrastive':
            visual_dim = visual_config.get('projection_dim', 128)
        elif visual_encoder_type == 'vit':
            visual_dim = visual_config.get('embed_dim', 768)
        elif visual_encoder_type == 'attribute':
            visual_dim = visual_config.get('embedding_dim', 256)
        
        # Transaction encoder selection
        transaction_encoder_type = transaction_config.get('type', 'longformer')
        if transaction_encoder_type == 'graph':
            self.transaction_encoder = TemporalGraphNN(
                feature_dim=transaction_config.get('feature_dim', 128),
                hidden_dim=transaction_config.get('hidden_dim', 256),
                output_dim=transaction_config.get('output_dim', 512),
                num_gnn_layers=transaction_config.get('num_gnn_layers', 2),
                gnn_type=transaction_config.get('gnn_type', 'gcn'),
                temporal_encoder_type=transaction_config.get('temporal_encoder_type', 'gru'),
                num_temporal_layers=transaction_config.get('num_temporal_layers', 1),
                dropout=transaction_config.get('dropout', 0.1)
            )
        elif transaction_encoder_type == 'longformer':
            self.transaction_encoder = TransactionLongformer(
                input_dim=transaction_config.get('input_dim', 128),
                hidden_dim=transaction_config.get('hidden_dim', 768),
                output_dim=transaction_config.get('output_dim', 512),
                max_seq_len=transaction_config.get('max_seq_len', 4096),
                num_hidden_layers=transaction_config.get('num_hidden_layers', 4),
                num_attention_heads=transaction_config.get('num_attention_heads', 8),
                attention_window=transaction_config.get('attention_window', 512),
                dropout=transaction_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown transaction encoder type: {transaction_encoder_type}")
            
        # Extract transaction dimension
        transaction_dim = transaction_config.get('output_dim', 512)
        
        # Fusion module selection
        fusion_type = fusion_config.get('type', 'bayesian')
        fusion_dim = fusion_config.get('fusion_dim', 512)
        
        if fusion_type == 'bayesian':
            self.fusion_module = BayesianMultimodalFusion(
                visual_dim=visual_dim,
                transaction_dim=transaction_dim,
                fusion_dim=fusion_dim,
                dropout=fusion_config.get('dropout', 0.1)
            )
        elif fusion_type == 'infotheoretic':
            self.fusion_module = InfoTheoreticFusion(
                visual_dim=visual_dim,
                transaction_dim=transaction_dim,
                fusion_dim=fusion_dim,
                temperature=fusion_config.get('temperature', 0.07),
                dropout=fusion_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        # Prediction heads
        self.price_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(prediction_config.get('dropout', 0.1)),
            nn.Linear(fusion_dim // 2, 1)
        )
        
        self.attribute_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(prediction_config.get('dropout', 0.1)),
            nn.Linear(fusion_dim // 2, prediction_config.get('num_attributes', 100))
        )
        
        # Save fusion type for forward pass
        self.fusion_type = fusion_type
        self.visual_encoder_type = visual_encoder_type
        self.transaction_encoder_type = transaction_encoder_type
        
    def forward(
        self,
        images,
        transaction_data,
        transaction_graphs=None,
        compute_loss=True
    ):
        """
        Forward pass through PrivaMod system.
        
        Args:
            images: Input images [B, C, H, W]
            transaction_data: Transaction data [B, T, F] or [B, F]
            transaction_graphs: Transaction graphs (only for graph encoder)
            compute_loss: Whether to compute auxiliary losses
            
        Returns:
            Dictionary containing:
                price_prediction: Predicted prices [B, 1]
                attribute_prediction: Predicted attributes [B, num_attributes]
                fused_features: Fused features [B, fusion_dim]
                losses: Dictionary of auxiliary losses (if compute_loss=True)
        """
        # Process visual data
        if self.visual_encoder_type == 'contrastive':
            visual_features, visual_proj, contrastive_logits, contrastive_labels = self.visual_encoder(
                images, compute_key=compute_loss
            )
        elif self.visual_encoder_type == 'attribute':
            visual_features, attribute_probs = self.visual_encoder(images)
        else:
            visual_features = self.visual_encoder(images)
            
        # Process transaction data
        if self.transaction_encoder_type == 'graph':
            if transaction_graphs is None:
                raise ValueError("Transaction graphs required for graph encoder")
            transaction_features = self.transaction_encoder(transaction_graphs, transaction_data)
        else:
            transaction_features = self.transaction_encoder(transaction_data)
            
        # Fuse features
        if self.fusion_type == 'bayesian':
            fused_features, visual_mean, transaction_mean, visual_logvar, transaction_logvar = self.fusion_module(
                visual_features, transaction_features
            )
        elif self.fusion_type == 'infotheoretic':
            fused_features, mi_loss = self.fusion_module(
                visual_features, transaction_features
            )
            
        # Compute predictions
        price_prediction = self.price_head(fused_features)
        attribute_prediction = self.attribute_head(fused_features)
        
        # Prepare output dictionary
        outputs = {
            'price_prediction': price_prediction,
            'attribute_prediction': attribute_prediction,
            'fused_features': fused_features,
            'visual_features': visual_features,
            'transaction_features': transaction_features
        }
        
        # Compute auxiliary losses
        if compute_loss:
            losses = {}
            
            # Contrastive loss for visual encoder
            if self.visual_encoder_type == 'contrastive' and contrastive_logits is not None:
                contrastive_loss = F.cross_entropy(contrastive_logits, contrastive_labels)
                losses['contrastive_loss'] = contrastive_loss
                
            # Uncertainty loss for Bayesian fusion
            if self.fusion_type == 'bayesian':
                # KL divergence between modality distributions
                kl_loss = 0.5 * torch.sum(
                    visual_logvar - transaction_logvar + 
                    (torch.exp(transaction_logvar) + (transaction_mean - visual_mean).pow(2)) /
                    torch.exp(visual_logvar) - 1
                ) / visual_features.size(0)
                losses['kl_loss'] = kl_loss
                
            # Mutual information loss for InfoTheoretic fusion
            if self.fusion_type == 'infotheoretic':
                losses['mi_loss'] = mi_loss
                
            outputs['losses'] = losses
            
        return outputs


# Model registry for easy model creation
class ModelRegistry:
    """Registry of available model configurations."""
    
    @staticmethod
    def get_model_types():
        """Get available model types."""
        return {
            'visual_encoder': ['contrastive', 'vit', 'attribute'],
            'transaction_encoder': ['graph', 'longformer'],
            'fusion': ['bayesian', 'infotheoretic']
        }
        
    @staticmethod
    def get_default_config(model_type):
        """Get default configuration for a model type."""
        configs = {
            'visual_contrastive': {
                'type': 'contrastive',
                'backbone': 'resnet50',
                'projection_dim': 128,
                'pretrained': True,
                'temperature': 0.07,
                'queue_size': 65536,
                'momentum': 0.999
            },
            'visual_vit': {
                'type': 'vit',
                'img_size': 224,
                'patch_size': 16,
                'in_chans': 3,
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4.0,
                'dropout': 0.1,
                'attn_dropout': 0.1,
                'mixer_tokens': True,
                'mixer_channels': True
            },
            'visual_attribute': {
                'type': 'attribute',
                'num_attributes': 100,
                'backbone': 'resnet50',
                'embedding_dim': 256,
                'num_transformer_layers': 3,
                'num_attention_heads': 8,
                'pretrained': True
            },
            'transaction_graph': {
                'type': 'graph',
                'feature_dim': 128,
                'hidden_dim': 256,
                'output_dim': 512,
                'num_gnn_layers': 2,
                'gnn_type': 'gcn',
                'temporal_encoder_type': 'gru',
                'num_temporal_layers': 1,
                'dropout': 0.1
            },
            'transaction_longformer': {
                'type': 'longformer',
                'input_dim': 128,
                'hidden_dim': 768,
                'output_dim': 512,
                'max_seq_len': 4096,
                'num_hidden_layers': 4,
                'num_attention_heads': 8,
                'attention_window': 512,
                'dropout': 0.1
            },
            'fusion_bayesian': {
                'type': 'bayesian',
                'fusion_dim': 512,
                'dropout': 0.1
            },
            'fusion_infotheoretic': {
                'type': 'infotheoretic',
                'fusion_dim': 512,
                'temperature': 0.07,
                'dropout': 0.1
            }
        }
        
        return configs.get(model_type, {})

def create_model(config):
    """
    Create a model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Instantiated model
    """
    model_type = config.get('architecture', 'privaMod')
    
    if model_type == 'privaMod':
        return PrivaModSystem(config)
    elif model_type == 'visual_encoder':
        visual_type = config.get('type', 'vit')
        if visual_type == 'contrastive':
            return ContrastiveVisualEncoder(**config)
        elif visual_type == 'vit':
            return VisionTransformerEncoder(**config)
        elif visual_type == 'attribute':
            return CompositeAttributeAnalyzer(**config)
    elif model_type == 'transaction_encoder':
        transaction_type = config.get('type', 'longformer')
        if transaction_type == 'graph':
            return TemporalGraphNN(**config)
        elif transaction_type == 'longformer':
            return TransactionLongformer(**config)
    elif model_type == 'fusion':
        fusion_type = config.get('type', 'bayesian')
        if fusion_type == 'bayesian':
            return BayesianMultimodalFusion(**config)
        elif fusion_type == 'infotheoretic':
            return InfoTheoreticFusion(**config)
    
    raise ValueError(f"Unknown model type: {model_type}")