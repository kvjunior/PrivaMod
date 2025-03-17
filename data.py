"""
PrivaMod Data Processing System
========================

Implements optimized data processing pipeline for multimodal NFT data:

1. Data Loading:
   - Memory-mapped operations for large datasets
   - Parallel data loading with optimized caching
   - Stream processing for transaction sequences

2. Preprocessing:
   - Advanced image preprocessing with visual augmentation
   - Transaction sequence normalization and feature extraction
   - Graph construction for transaction networks

3. Dataset Implementation:
   - Optimized multimodal dataset with efficient memory usage
   - Custom samplers for balanced training
   - Privacy-aware data handling

This module is optimized for high-performance processing on multi-GPU systems,
with specific optimizations for the PrivaMod multimodal architecture.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import pandas as pd
import os
import json
import h5py
import mmap
import cv2
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import threading
from queue import Queue
import dgl
import random
import pickle

# For image preprocessing
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

# For efficient memory-mapped processing
from tqdm import tqdm


class NFTDatasetPreprocessor:
    """
    Comprehensive data preprocessor for NFT data.
    Implements efficient parallel processing and caching.
    """
    
    def __init__(
        self,
        image_dir: str,
        transaction_file: str,
        output_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        max_seq_length: int = 128,
        num_workers: int = 16,
        cache_size: int = 10000,
        compute_graph: bool = True
    ):
        """
        Initialize NFT dataset preprocessor.
        
        Args:
            image_dir: Directory containing NFT images
            transaction_file: Path to transaction data file
            output_dir: Output directory for processed data
            image_size: Image size for resizing
            max_seq_length: Maximum transaction sequence length
            num_workers: Number of parallel workers
            cache_size: Size of in-memory cache
            compute_graph: Whether to compute transaction graphs
        """
        self.image_dir = Path(image_dir)
        self.transaction_file = Path(transaction_file)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.compute_graph = compute_graph
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize processing queues
        self.image_queue = Queue(maxsize=cache_size)
        self.transaction_queue = Queue(maxsize=cache_size)
        self.result_queue = Queue(maxsize=cache_size)
        
        # Initialize counters
        self.processed_images = 0
        self.processed_transactions = 0
        
        # Statistics tracking
        self.stats = {
            'image_processing_time': [],
            'transaction_processing_time': [],
            'memory_usage': []
        }
        
    def process_all(self):
        """Process all NFT data and save to disk."""
        self.logger.info(f"Starting data preprocessing: images from {self.image_dir}, transactions from {self.transaction_file}")
        
        # Start parallel processing threads
        image_thread = threading.Thread(target=self._process_images)
        transaction_thread = threading.Thread(target=self._process_transactions)
        
        # Start threads
        image_thread.start()
        transaction_thread.start()
        
        # Wait for completion
        image_thread.join()
        transaction_thread.join()
        
        # Process relationships and create final dataset
        self._create_final_dataset()
        
        self.logger.info(f"Preprocessing complete: {self.processed_images} images, {self.processed_transactions} transactions")
        
    def _process_images(self):
        """Process all NFT images with parallel workers."""
        self.logger.info(f"Starting image processing with {self.num_workers} workers")
        
        # Get all image files
        image_files = list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg'))
        total_images = len(image_files)
        
        if not image_files:
            self.logger.warning(f"No images found in {self.image_dir}")
            return
            
        # Create output directory for processed images
        image_output_dir = self.output_dir / 'images'
        image_output_dir.mkdir(exist_ok=True)
        
        # Initialize HDF5 dataset for processed images
        h5_path = self.output_dir / 'processed_images.h5'
        with h5py.File(h5_path, 'w') as f:
            # Create dataset with compression
            f.create_dataset(
                'images',
                shape=(total_images, 3, *self.image_size),
                dtype=np.float32,
                compression='gzip',
                compression_opts=9
            )
            
            # Create metadata dataset
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('image_ids', shape=(total_images,), dtype=dt)
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for i, image_file in enumerate(image_files):
                futures.append(
                    executor.submit(self._process_single_image, image_file, i)
                )
                
            # Process results as they complete
            for i, future in enumerate(tqdm(futures, total=len(futures), desc="Processing images")):
                try:
                    image_id, processed_image = future.result()
                    
                    # Save to HDF5 file
                    with h5py.File(h5_path, 'a') as f:
                        f['images'][i] = processed_image
                        f['image_ids'][i] = image_id
                        
                    self.processed_images += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing image: {e}")
        
        self.logger.info(f"Image processing complete: {self.processed_images} images processed")
    
    def _process_single_image(self, image_path: Path, index: int) -> Tuple[str, np.ndarray]:
        """
        Process a single image.
        
        Args:
            image_path: Path to image file
            index: Image index
            
        Returns:
            Tuple of (image_id, processed_image)
        """
        start_time = time.time()
        
        try:
            # Extract image ID from filename
            image_id = image_path.stem
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize and normalize
            image = image.resize(self.image_size, Image.LANCZOS)
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Convert to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Track processing time
            processing_time = time.time() - start_time
            self.stats['image_processing_time'].append(processing_time)
            
            return image_id, image_array
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            # Return empty image in case of error
            empty_image = np.zeros((3, *self.image_size), dtype=np.float32)
            return str(image_path.stem), empty_image
    
    def _process_transactions(self):
        """Process transaction data with memory-efficient streaming."""
        self.logger.info(f"Starting transaction processing")
        
        # Create output directory for processed transactions
        transaction_output_dir = self.output_dir / 'transactions'
        transaction_output_dir.mkdir(exist_ok=True)
        
        try:
            # Memory-efficient reading
            transactions = []
            with open(self.transaction_file, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                for line in tqdm(iter(mm.readline, b''), desc="Reading transactions"):
                    try:
                        # Decode and parse JSON
                        transaction = json.loads(line.decode().strip())
                        transactions.append(transaction)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Error parsing transaction: {e}")
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(transactions)
            
            # Optimize data types for memory efficiency
            self._optimize_dataframe(df)
            
            # Process and normalize transaction features
            df = self._process_transaction_features(df)
            
            # Build transaction sequences
            sequence_data = self._build_transaction_sequences(df)
            
            # Build transaction graphs if requested
            if self.compute_graph:
                graph_data = self._build_transaction_graphs(df)
                
                # Save graph data
                with open(self.output_dir / 'transaction_graphs.pkl', 'wb') as f:
                    pickle.dump(graph_data, f)
            
            # Save processed transactions
            self._save_processed_transactions(df, sequence_data)
            
            self.processed_transactions = len(df)
            
        except Exception as e:
            self.logger.error(f"Error processing transactions: {e}")
            raise
            
        self.logger.info(f"Transaction processing complete: {self.processed_transactions} transactions processed")
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        start_mem = df.memory_usage().sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        # Optimize categorical and object columns
        for col in df.select_dtypes(include=['object']).columns:
            # Check if column can be converted to categorical
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage().sum() / 1024**2
        reduction = (start_mem - end_mem) / start_mem * 100
        
        self.logger.info(f"DataFrame memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.2f}% reduction)")
        
        return df
    
    def _process_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and normalize transaction features.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Ensure required columns exist
        required_columns = ['id', 'timestamp', 'price', 'sender', 'receiver']
        for col in required_columns:
            if col not in df.columns:
                self.logger.warning(f"Required column '{col}' not found. Creating empty column.")
                if col == 'timestamp':
                    df[col] = pd.Timestamp.now()
                elif col in ['sender', 'receiver']:
                    df[col] = "unknown"
                else:
                    df[col] = 0
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                self.logger.warning("Error converting timestamps. Using current time.")
                df['timestamp'] = pd.Timestamp.now()
                
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Normalize price values
        if 'price' in df.columns:
            # Handle outliers (clip at 99.5 percentile)
            price_threshold = df['price'].quantile(0.995)
            df['price_clipped'] = df['price'].clip(upper=price_threshold)
            
            # Log transform for skewed distribution
            df['price_log'] = np.log1p(df['price_clipped'])
            
            # Normalize to [0, 1] range
            price_min = df['price_log'].min()
            price_max = df['price_log'].max()
            if price_max > price_min:
                df['price_normalized'] = (df['price_log'] - price_min) / (price_max - price_min)
            else:
                df['price_normalized'] = 0
        
        # Convert categorical data to numerical
        # Handle token type if present
        if 'type' in df.columns:
            # Create type mapping
            type_mapping = {t: i for i, t in enumerate(df['type'].unique())}
            df['type_id'] = df['type'].map(type_mapping)
            
            # Save type mapping
            with open(self.output_dir / 'type_mapping.json', 'w') as f:
                json.dump(type_mapping, f)
        
        # Handle attributes/traits if present
        if 'attributes' in df.columns:
            # Extract all unique attributes
            all_attributes = set()
            for attrs in df['attributes'].dropna():
                if isinstance(attrs, list):
                    all_attributes.update(attrs)
                elif isinstance(attrs, str):
                    # Try to parse as JSON
                    try:
                        parsed_attrs = json.loads(attrs)
                        if isinstance(parsed_attrs, list):
                            all_attributes.update(parsed_attrs)
                    except:
                        pass
            
            # Create attribute mapping
            attribute_mapping = {attr: i for i, attr in enumerate(sorted(all_attributes))}
            
            # Create attribute vectors
            attr_vectors = np.zeros((len(df), len(attribute_mapping)), dtype=np.float32)
            
            for i, attrs in enumerate(df['attributes'].fillna("[]")):
                if isinstance(attrs, list):
                    for attr in attrs:
                        if attr in attribute_mapping:
                            attr_vectors[i, attribute_mapping[attr]] = 1.0
                elif isinstance(attrs, str):
                    # Try to parse as JSON
                    try:
                        parsed_attrs = json.loads(attrs)
                        if isinstance(parsed_attrs, list):
                            for attr in parsed_attrs:
                                if attr in attribute_mapping:
                                    attr_vectors[i, attribute_mapping[attr]] = 1.0
                    except:
                        pass
            
            # Save as h5py array
            with h5py.File(self.output_dir / 'attribute_vectors.h5', 'w') as f:
                f.create_dataset('vectors', data=attr_vectors, compression='gzip')
                
                # Save attribute mapping
                dt = h5py.special_dtype(vlen=str)
                attrs_dataset = f.create_dataset('attributes', shape=(len(attribute_mapping),), dtype=dt)
                for attr, idx in attribute_mapping.items():
                    attrs_dataset[idx] = attr
            
            # Save attribute mapping as JSON
            with open(self.output_dir / 'attribute_mapping.json', 'w') as f:
                json.dump(attribute_mapping, f)
        
        # Compute additional features
        # Add time-based features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['year'] = df['timestamp'].dt.year
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Normalize time features
            for col in ['hour', 'day', 'month', 'day_of_week']:
                if col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        df[f"{col}_normalized"] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df[f"{col}_normalized"] = 0
        
        return df
    
    def _build_transaction_sequences(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Build transaction sequences for each token.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Dictionary of token sequences
        """
        # Group by token ID
        token_groups = df.groupby('id')
        
        # Feature columns to include in sequences
        feature_cols = [
            'price_normalized', 'type_id',
            'hour_normalized', 'day_normalized', 'month_normalized', 'day_of_week_normalized'
        ]
        
        # Only include columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Add one-hot encoded columns if they exist
        onehot_cols = [col for col in df.columns if col.startswith('type_') and col != 'type_id']
        feature_cols.extend(onehot_cols)
        
        # Calculate feature dimension
        feature_dim = len(feature_cols)
        
        # Create sequences dictionary
        sequences = {}
        sequence_lengths = []
        
        for token_id, group in tqdm(token_groups, desc="Building sequences"):
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Get sequence length
            seq_len = min(len(group), self.max_seq_length)
            sequence_lengths.append(seq_len)
            
            # Initialize sequence array
            sequence = np.zeros((self.max_seq_length, feature_dim), dtype=np.float32)
            
            # Fill sequence with available data
            for i, (_, row) in enumerate(group.iloc[-seq_len:].iterrows()):
                for j, col in enumerate(feature_cols):
                    if col in row and not pd.isna(row[col]):
                        sequence[i, j] = row[col]
            
            # Store sequence
            sequences[str(token_id)] = sequence
        
        # Log sequence statistics
        avg_seq_len = np.mean(sequence_lengths)
        self.logger.info(f"Built {len(sequences)} sequences with average length {avg_seq_len:.2f} (max: {self.max_seq_length})")
        
        return sequences
    
    def _build_transaction_graphs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Build transaction graphs for network analysis.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Dictionary of graph data
        """
        self.logger.info("Building transaction graphs")
        
        # Ensure required columns exist
        if 'sender' not in df.columns or 'receiver' not in df.columns:
            self.logger.warning("Sender or receiver columns missing. Cannot build graphs.")
            return {}
        
        # Create node mapping
        all_addresses = set(df['sender'].unique()).union(set(df['receiver'].unique()))
        address_to_idx = {addr: i for i, addr in enumerate(all_addresses)}
        
        # Create graph
        src_nodes = [address_to_idx[addr] for addr in df['sender']]
        dst_nodes = [address_to_idx[addr] for addr in df['receiver']]
        
        # Create DGL graph
        g = dgl.graph((src_nodes, dst_nodes))
        
        # Add node features
        # Count transactions per address
        addr_counts = df['sender'].value_counts().to_dict()
        addr_counts.update(df['receiver'].value_counts().to_dict())
        
        # Node activity features
        node_activity = np.array([
            addr_counts.get(addr, 0) for addr in all_addresses
        ], dtype=np.float32)
        
        # Calculate total volume per address
        addr_volumes = {}
        for addr in all_addresses:
            # Outgoing volume
            out_volume = df[df['sender'] == addr]['price'].sum()
            # Incoming volume
            in_volume = df[df['receiver'] == addr]['price'].sum()
            # Total volume
            addr_volumes[addr] = out_volume + in_volume
            
        # Node volume features
        node_volumes = np.array([
            addr_volumes.get(addr, 0) for addr in all_addresses
        ], dtype=np.float32)
        
        # Normalize node features
        if node_activity.max() > 0:
            node_activity = node_activity / node_activity.max()
        if node_volumes.max() > 0:
            node_volumes = node_volumes / node_volumes.max()
            
        # Add to graph
        g.ndata['activity'] = torch.FloatTensor(node_activity)
        g.ndata['volume'] = torch.FloatTensor(node_volumes)
        
        # Add edge features
        edge_timestamps = torch.FloatTensor(
            pd.to_datetime(df['timestamp']).astype(np.int64).values
        )
        edge_prices = torch.FloatTensor(df['price'].values)
        
        # Normalize edge features
        if edge_timestamps.max() > edge_timestamps.min():
            edge_timestamps = (edge_timestamps - edge_timestamps.min()) / (edge_timestamps.max() - edge_timestamps.min())
        else:
            edge_timestamps = torch.zeros_like(edge_timestamps)
            
        if edge_prices.max() > 0:
            edge_prices = edge_prices / edge_prices.max()
        
        # Add to graph
        g.edata['timestamp'] = edge_timestamps
        g.edata['price'] = edge_prices
        
        # Save mappings
        graph_data = {
            'graph': g,
            'address_to_idx': address_to_idx,
            'idx_to_address': {i: addr for addr, i in address_to_idx.items()},
            'num_nodes': len(all_addresses),
            'num_edges': len(df)
        }
        
        self.logger.info(f"Built transaction graph with {len(all_addresses)} nodes and {len(df)} edges")
        
        return graph_data
    
    def _save_processed_transactions(
        self,
        df: pd.DataFrame,
        sequence_data: Dict[str, np.ndarray]
    ):
        """
        Save processed transaction data.
        
        Args:
            df: Processed transaction DataFrame
            sequence_data: Token sequence data
        """
        # Save processed DataFrame
        df.to_pickle(self.output_dir / 'processed_transactions.pkl')
        
        # Save sequences as HDF5
        with h5py.File(self.output_dir / 'transaction_sequences.h5', 'w') as f:
            # Create token ID dataset
            dt = h5py.special_dtype(vlen=str)
            token_ids = list(sequence_data.keys())
            id_dataset = f.create_dataset('token_ids', shape=(len(token_ids),), dtype=dt)
            
            # Create sequence dataset
            if token_ids:
                # Get shape from first sequence
                first_seq = sequence_data[token_ids[0]]
                seq_shape = first_seq.shape
                
                # Create dataset
                seq_dataset = f.create_dataset(
                    'sequences',
                    shape=(len(token_ids), *seq_shape),
                    dtype=np.float32,
                    compression='gzip'
                )
                
                # Fill datasets
                for i, token_id in enumerate(token_ids):
                    id_dataset[i] = token_id
                    seq_dataset[i] = sequence_data[token_id]
        
        # Save feature information
        feature_cols = [
            'price_normalized', 'type_id',
            'hour_normalized', 'day_normalized', 'month_normalized', 'day_of_week_normalized'
        ]
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Add one-hot encoded columns if they exist
        onehot_cols = [col for col in df.columns if col.startswith('type_') and col != 'type_id']
        feature_cols.extend(onehot_cols)
        
        with open(self.output_dir / 'feature_columns.json', 'w') as f:
            json.dump(feature_cols, f)
    
    def _create_final_dataset(self):
        """Create final dataset with metadata."""
        # Create metadata
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'image_count': self.processed_images,
            'transaction_count': self.processed_transactions,
            'image_size': self.image_size,
            'max_seq_length': self.max_seq_length,
            'compute_graph': self.compute_graph,
            'processing_stats': {
                'avg_image_processing_time': np.mean(self.stats['image_processing_time']) if self.stats['image_processing_time'] else 0,
                'avg_transaction_processing_time': np.mean(self.stats['transaction_processing_time']) if self.stats['transaction_processing_time'] else 0
            }
        }
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Final dataset created at {self.output_dir}")


class NFTImageTransforms:
    """
    Advanced image transformations for NFT data augmentation.
    Implements specialized transforms for NFT visual characteristics.
    """
    
    @staticmethod
    def get_train_transforms(
        image_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> transforms.Compose:
        """
        Get training transforms with augmentation.
        
        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Composed transforms
        """
        return transforms.Compose([
            transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    @staticmethod
    def get_val_transforms(
        image_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> transforms.Compose:
        """
        Get validation transforms without augmentation.
        
        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Composed transforms
        """
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    @staticmethod
    def get_test_transforms(
        image_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> transforms.Compose:
        """
        Get test transforms.
        
        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Composed transforms
        """
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


class NFTDataset(Dataset):
    """
    Optimized multimodal NFT dataset implementation.
    Implements efficient data loading with caching and parallel processing.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_transforms: Optional[transforms.Compose] = None,
        max_seq_length: int = 128,
        load_graphs: bool = False,
        memcache_size: int = 1000,
        price_threshold: Optional[float] = None
    ):
        """
        Initialize NFT dataset.
        
        Args:
            data_dir: Directory containing processed data
            image_transforms: Image transformations to apply
            max_seq_length: Maximum transaction sequence length
            load_graphs: Whether to load transaction graphs
            memcache_size: Size of memory cache
            price_threshold: Optional threshold for filtering by price
        """
        self.data_dir = Path(data_dir)
        self.image_transforms = image_transforms
        self.max_seq_length = max_seq_length
        self.load_graphs = load_graphs
        self.memcache_size = memcache_size
        self.price_threshold = price_threshold
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load dataset metadata
        self._load_metadata()
        
        # Initialize caches
        self.image_cache = {}
        self.sequence_cache = {}
        self.attribute_cache = {}
        self.graph_cache = None
        
        # Load data
        self._load_data()
    
    def _load_metadata(self):
        """Load dataset metadata."""
        metadata_path = self.data_dir / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.logger.warning(f"Metadata file not found at {metadata_path}")
            self.metadata = {}
            
        # Load feature columns information
        feature_cols_path = self.data_dir / 'feature_columns.json'
        
        if feature_cols_path.exists():
            with open(feature_cols_path, 'r') as f:
                self.feature_columns = json.load(f)
        else:
            self.logger.warning(f"Feature columns file not found at {feature_cols_path}")
            self.feature_columns = []
            
        # Load mappings
        self.type_mapping = {}
        type_mapping_path = self.data_dir / 'type_mapping.json'
        
        if type_mapping_path.exists():
            with open(type_mapping_path, 'r') as f:
                self.type_mapping = json.load(f)
                
        self.attribute_mapping = {}
        attr_mapping_path = self.data_dir / 'attribute_mapping.json'
        
        if attr_mapping_path.exists():
            with open(attr_mapping_path, 'r') as f:
                self.attribute_mapping = json.load(f)
    
    def _load_data(self):
        """Load dataset data."""
        # Load processed transactions
        trans_path = self.data_dir / 'processed_transactions.pkl'
        if trans_path.exists():
            self.transactions = pd.read_pickle(trans_path)
            
            # Apply price threshold if specified
            if self.price_threshold is not None and 'price' in self.transactions.columns:
                self.transactions = self.transactions[self.transactions['price'] <= self.price_threshold]
                
            self.logger.info(f"Loaded {len(self.transactions)} transactions")
        else:
            self.logger.warning(f"Transaction file not found at {trans_path}")
            self.transactions = pd.DataFrame()
            
        # Load token IDs and map to indices
        self.token_ids = []
        self.token_to_idx = {}
        
        with h5py.File(self.data_dir / 'transaction_sequences.h5', 'r') as f:
            if 'token_ids' in f:
                # Get token IDs
                token_ids = [id.decode() for id in f['token_ids'][:]]
                self.token_ids = token_ids
                
                # Create mapping
                self.token_to_idx = {id: i for i, id in enumerate(token_ids)}
                
                self.logger.info(f"Loaded {len(self.token_ids)} token IDs")
            else:
                self.logger.warning("No token IDs found in sequence file")
                
        # Load image IDs
        with h5py.File(self.data_dir / 'processed_images.h5', 'r') as f:
            if 'image_ids' in f:
                # Get image IDs
                image_ids = [id.decode() for id in f['image_ids'][:]]
                self.image_ids = image_ids
                
                # Create mapping
                self.image_to_idx = {id: i for i, id in enumerate(image_ids)}
                
                self.logger.info(f"Loaded {len(self.image_ids)} image IDs")
            else:
                self.logger.warning("No image IDs found in image file")
                self.image_ids = []
                self.image_to_idx = {}
                
        # Create valid token list (tokens with both image and transaction data)
        self.valid_tokens = [
            token_id for token_id in self.token_ids
            if token_id in self.image_to_idx
        ]
        
        self.logger.info(f"Found {len(self.valid_tokens)} tokens with both image and transaction data")
        
        # Load attribute vectors if available
        attr_path = self.data_dir / 'attribute_vectors.h5'
        if attr_path.exists():
            with h5py.File(attr_path, 'r') as f:
                if 'vectors' in f:
                    # Store reference to avoid loading everything in memory
                    self.attribute_vectors_file = attr_path
                    self.attribute_vectors_shape = f['vectors'].shape
                    
                    self.logger.info(f"Attribute vectors available with shape {self.attribute_vectors_shape}")
                else:
                    self.logger.warning("No attribute vectors found")
                    self.attribute_vectors_file = None
                    self.attribute_vectors_shape = None
        else:
            self.logger.warning(f"Attribute vectors file not found at {attr_path}")
            self.attribute_vectors_file = None
            self.attribute_vectors_shape = None
            
        # Load transaction graphs if requested
        if self.load_graphs:
            graph_path = self.data_dir / 'transaction_graphs.pkl'
            if graph_path.exists():
                with open(graph_path, 'rb') as f:
                    self.graph_data = pickle.load(f)
                    
                self.logger.info(f"Loaded transaction graph with {self.graph_data['num_nodes']} nodes and {self.graph_data['num_edges']} edges")
            else:
                self.logger.warning(f"Transaction graph file not found at {graph_path}")
                self.graph_data = None
        else:
            self.graph_data = None
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.valid_tokens)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary of tensors
        """
        # Get token ID
        token_id = self.valid_tokens[idx]
        
        # Get image
        image = self._get_image(token_id)
        
        # Get transaction sequence
        sequence = self._get_sequence(token_id)
        
        # Get attributes
        attributes = self._get_attributes(token_id)
        
        # Get price (if available)
        price = self._get_price(token_id)
        
        # Get token type (if available)
        token_type = self._get_token_type(token_id)
        
        # Create sample dictionary
        sample = {
            'token_id': token_id,
            'image': image,
            'sequence': sequence,
            'attributes': attributes,
            'price': price,
            'token_type': token_type
        }
        
        # Add transaction graph if requested
        if self.load_graphs and self.graph_data is not None:
            sample['graph'] = self.graph_data['graph']
        
        return sample
    
    def _get_image(self, token_id: str) -> torch.Tensor:
        """
        Get image tensor for token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Image tensor
        """
        # Check cache first
        if token_id in self.image_cache:
            image = self.image_cache[token_id]
        else:
            # Get image index
            image_idx = self.image_to_idx.get(token_id)
            
            if image_idx is None:
                # Return empty image if not found
                empty_image = torch.zeros(3, 224, 224)
                return empty_image
                
            # Load image from HDF5
            with h5py.File(self.data_dir / 'processed_images.h5', 'r') as f:
                image_array = f['images'][image_idx]
                
            # Convert to tensor
            image = torch.from_numpy(image_array)
            
            # Update cache (with LRU-like behavior)
            if len(self.image_cache) >= self.memcache_size:
                # Remove random item
                random_key = random.choice(list(self.image_cache.keys()))
                del self.image_cache[random_key]
                
            self.image_cache[token_id] = image
        
        # Apply transforms if available
        if self.image_transforms is not None:
            # Convert to PIL for transforms
            image_pil = transforms.ToPILImage()(image)
            image = self.image_transforms(image_pil)
            
        return image
    
    def _get_sequence(self, token_id: str) -> torch.Tensor:
        """
        Get transaction sequence for token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Sequence tensor
        """
        # Check cache first
        if token_id in self.sequence_cache:
            return self.sequence_cache[token_id]
            
        # Get token index
        token_idx = self.token_to_idx.get(token_id)
        
        if token_idx is None:
            # Return empty sequence if not found
            empty_seq = torch.zeros(self.max_seq_length, len(self.feature_columns))
            return empty_seq
            
        # Load sequence from HDF5
        with h5py.File(self.data_dir / 'transaction_sequences.h5', 'r') as f:
            sequence_array = f['sequences'][token_idx]
            
        # Convert to tensor
        sequence = torch.from_numpy(sequence_array).float()
        
        # Update cache (with LRU-like behavior)
        if len(self.sequence_cache) >= self.memcache_size:
            # Remove random item
            random_key = random.choice(list(self.sequence_cache.keys()))
            del self.sequence_cache[random_key]
            
        self.sequence_cache[token_id] = sequence
        
        return sequence
    
    def _get_attributes(self, token_id: str) -> torch.Tensor:
        """
        Get attribute vector for token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Attribute tensor
        """
        # Check cache first
        if token_id in self.attribute_cache:
            return self.attribute_cache[token_id]
            
        # Return empty tensor if no attribute vectors available
        if self.attribute_vectors_file is None:
            return torch.zeros(1)
            
        # Get token index in transaction file
        token_records = self.transactions[self.transactions['id'] == token_id]
        
        if len(token_records) == 0:
            # Return empty vector if no records found
            return torch.zeros(self.attribute_vectors_shape[1])
            
        # Use the first record's index
        record_idx = token_records.index[0]
        
        # Load attribute vector
        with h5py.File(self.attribute_vectors_file, 'r') as f:
            if record_idx < len(f['vectors']):
                attr_vector = f['vectors'][record_idx]
            else:
                # Return empty vector if index is out of bounds
                attr_vector = np.zeros(self.attribute_vectors_shape[1])
        
        # Convert to tensor
        attributes = torch.from_numpy(attr_vector).float()
        
        # Update cache (with LRU-like behavior)
        if len(self.attribute_cache) >= self.memcache_size:
            # Remove random item
            random_key = random.choice(list(self.attribute_cache.keys()))
            del self.attribute_cache[random_key]
            
        self.attribute_cache[token_id] = attributes
        
        return attributes
    
    def _get_price(self, token_id: str) -> torch.Tensor:
        """
        Get price for token (last transaction price).
        
        Args:
            token_id: Token ID
            
        Returns:
            Price tensor
        """
        # Get token transactions
        token_records = self.transactions[self.transactions['id'] == token_id]
        
        if len(token_records) == 0 or 'price' not in token_records.columns:
            # Return zero if no records found or no price column
            return torch.tensor(0.0).float()
            
        # Get latest transaction price
        latest_price = token_records.iloc[-1]['price']
        
        # Use normalized price if available
        if 'price_normalized' in token_records.columns:
            latest_price = token_records.iloc[-1]['price_normalized']
            
        return torch.tensor(latest_price).float()
    
    def _get_token_type(self, token_id: str) -> torch.Tensor:
        """
        Get token type as one-hot encoded tensor.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token type tensor
        """
        # Get token records
        token_records = self.transactions[self.transactions['id'] == token_id]
        
        if len(token_records) == 0 or 'type' not in token_records.columns:
            # Return empty vector if no records found or no type column
            return torch.zeros(len(self.type_mapping) if self.type_mapping else 1)
            
        # Get token type
        token_type = token_records.iloc[0]['type']
        
        # Convert to one-hot
        if not self.type_mapping:
            return torch.zeros(1)
            
        # Create one-hot tensor
        type_id = self.type_mapping.get(token_type, 0)
        one_hot = torch.zeros(len(self.type_mapping))
        one_hot[type_id] = 1.0
        
        return one_hot


class NFTDataModule:
    """
    NFT data module for managing train/validation/test splits.
    Implements efficient data loading and preprocessing.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        val_split: float = 0.2,
        test_split: float = 0.1,
        image_size: Tuple[int, int] = (224, 224),
        max_seq_length: int = 128,
        load_graphs: bool = False,
        price_threshold: Optional[float] = None,
        use_ddp: bool = False
    ):
        """
        Initialize NFT data module.
        
        Args:
            data_dir: Directory containing processed data
            batch_size: Batch size
            num_workers: Number of parallel workers
            val_split: Validation split ratio
            test_split: Test split ratio
            image_size: Image size
            max_seq_length: Maximum transaction sequence length
            load_graphs: Whether to load transaction graphs
            price_threshold: Optional threshold for filtering by price
            use_ddp: Whether to use DistributedDataParallel
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        self.load_graphs = load_graphs
        self.price_threshold = price_threshold
        self.use_ddp = use_ddp
        
        # Image transforms
        self.train_transforms = NFTImageTransforms.get_train_transforms(image_size)
        self.val_transforms = NFTImageTransforms.get_val_transforms(image_size)
        self.test_transforms = NFTImageTransforms.get_test_transforms(image_size)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def setup(self):
        """Set up datasets and splits."""
        # Create full dataset
        full_dataset = NFTDataset(
            data_dir=self.data_dir,
            image_transforms=self.train_transforms,
            max_seq_length=self.max_seq_length,
            load_graphs=self.load_graphs,
            price_threshold=self.price_threshold
        )
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * self.val_split)
        test_size = int(dataset_size * self.test_split)
        train_size = dataset_size - val_size - test_size
        
        # Create splits
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Set different transforms for validation and test
        self.val_dataset.dataset = NFTDataset(
            data_dir=self.data_dir,
            image_transforms=self.val_transforms,
            max_seq_length=self.max_seq_length,
            load_graphs=self.load_graphs,
            price_threshold=self.price_threshold
        )
        
        self.test_dataset.dataset = NFTDataset(
            data_dir=self.data_dir,
            image_transforms=self.test_transforms,
            max_seq_length=self.max_seq_length,
            load_graphs=self.load_graphs,
            price_threshold=self.price_threshold
        )
        
        self.logger.info(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
        
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.use_ddp,  # Don't shuffle if using DDP
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(self.train_dataset) if self.use_ddp else None
        )
        
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=DistributedSampler(self.val_dataset, shuffle=False) if self.use_ddp else None
        )
        
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=DistributedSampler(self.test_dataset, shuffle=False) if self.use_ddp else None
        )


# Helper functions for data loading
def create_dataset_preprocessor(
    image_dir: str,
    transaction_file: str,
    output_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    max_seq_length: int = 128,
    num_workers: int = 16,
    compute_graph: bool = True
) -> NFTDatasetPreprocessor:
    """
    Create dataset preprocessor for raw data.
    
    Args:
        image_dir: Directory containing NFT images
        transaction_file: Path to transaction data file
        output_dir: Output directory for processed data
        image_size: Image size for resizing
        max_seq_length: Maximum transaction sequence length
        num_workers: Number of parallel workers
        compute_graph: Whether to compute transaction graphs
        
    Returns:
        Configured NFTDatasetPreprocessor instance
    """
    return NFTDatasetPreprocessor(
        image_dir=image_dir,
        transaction_file=transaction_file,
        output_dir=output_dir,
        image_size=image_size,
        max_seq_length=max_seq_length,
        num_workers=num_workers,
        compute_graph=compute_graph
    )

def create_data_module(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    val_split: float = 0.2,
    test_split: float = 0.1,
    image_size: Tuple[int, int] = (224, 224),
    max_seq_length: int = 128,
    load_graphs: bool = False,
    price_threshold: Optional[float] = None,
    use_ddp: bool = False
) -> NFTDataModule:
    """
    Create NFT data module for processed data.
    
    Args:
        data_dir: Directory containing processed data
        batch_size: Batch size
        num_workers: Number of parallel workers
        val_split: Validation split ratio
        test_split: Test split ratio
        image_size: Image size
        max_seq_length: Maximum transaction sequence length
        load_graphs: Whether to load transaction graphs
        price_threshold: Optional threshold for filtering by price
        use_ddp: Whether to use DistributedDataParallel
        
    Returns:
        Configured NFTDataModule instance
    """
    data_module = NFTDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split,
        image_size=image_size,
        max_seq_length=max_seq_length,
        load_graphs=load_graphs,
        price_threshold=price_threshold,
        use_ddp=use_ddp
    )
    data_module.setup()
    
    return data_module