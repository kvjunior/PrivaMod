# PrivaMod: Privacy-Preserving Multimodal Integration for NFT Visual and Transaction Analysis

## Overview

PrivaMod is a novel privacy-preserving Bayesian framework for multimodal analysis of NFT markets. The system integrates visual features and transaction patterns while maintaining strong differential privacy guarantees, addressing the critical challenge of analyzing sensitive market data without compromising participant privacy.


## Key Features

- **Uncertainty-Aware Bayesian Fusion**: Dynamically weights information from each modality based on confidence levels, improving prediction accuracy by 18.9% compared to single-modality approaches.
- **Strong Privacy Guarantees**: Maintains differential privacy with ε = 0.08, δ = 1e-5 throughout the analytical pipeline.
- **Parameter-Efficient Architecture**: Achieves 99.98% parameter efficiency through strategic parameter sharing, enhancing performance with limited training data.
- **Comprehensive Market Analytics**: Provides visual feature analysis, transaction network insights, cross-modal correlations, and market efficiency metrics.
- **Extensive Validation**: Tested on 167,492 CryptoPunk transactions, demonstrating superior performance in both analytical accuracy and privacy preservation.

## System Architecture

PrivaMod consists of four primary components:

1. **Privacy-Aware Data Preprocessing**: Implements image transformations and transaction data anonymization.
2. **Modality-Specific Encoders**:
   - **Visual**: Vision Transformer (ViT) with contrastive learning
   - **Transaction**: Longformer with sliding window attention and graph structural modeling
3. **Bayesian Multimodal Fusion**: Integrates visual and transaction features with uncertainty quantification
4. **Market Analysis Modules**

![System Architecture](architecture_diagram.png)

## Installation

### Requirements

- Python 3.8+
- CUDA 11.6+ (for GPU acceleration)
- 16GB+ RAM

### Setup

# Clone repository
git clone
cd privamod

# Create virtual environment
python -m venv privamod-env
source privamod-env/bin/activate  # On Windows: privamod-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Usage

### Data Preparation

from data import NFTDatasetPreprocessor, create_data_module

# Preprocess raw data
preprocessor = NFTDatasetPreprocessor(
    image_dir="path/to/nft_images",
    transaction_file="path/to/transactions.json",
    output_dir="processed_data",
    image_size=(224, 224),
    max_seq_length=128,
    compute_graph=True
)
preprocessor.process_all()

# Create data module
data_module = create_data_module(
    data_dir="processed_data",
    batch_size=32,
    num_workers=8,
    val_split=0.2,
    test_split=0.1,
    image_size=(224, 224),
    max_seq_length=128,
    load_graphs=True
)

### Model Training

from models import create_model
from privacy import create_privacy_engine
from train import create_trainer
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create model
model = create_model(config["model"])

# Initialize privacy engine
privacy_engine = create_privacy_engine(
    config["privacy"],
    data_size=len(data_module.train_dataset)
)

# Create trainer
trainer = create_trainer(
    model=model,
    train_loader=data_module.train_dataloader(),
    val_loader=data_module.val_dataloader(),
    test_loader=data_module.test_dataloader(),
    config=config["training"],
    output_dir="results",
    privacy_engine=privacy_engine
)

# Train model
best_model_path = trainer.train()


### Market Analysis


from analysis import MarketAnalyzer

# Create analyzer
analyzer = MarketAnalyzer(
    model=model,
    data_loader=data_module.test_dataloader(),
    config=config["analysis"],
    output_dir="analysis_results"
)

# Run analysis
results = analyzer.analyze()


## Configuration

PrivaMod uses YAML-based configuration for flexibility. Key configuration sections include:

### Model Configuration

model:
  architecture: "privaMod"
  visual:
    type: "vit"
    img_size: 224
    patch_size: 16
    embed_dim: 768
    depth: 12
    num_heads: 12
  transaction:
    type: "longformer"
    input_dim: 128
    hidden_dim: 768
    output_dim: 512
    max_seq_length: 4096
  fusion:
    type: "bayesian"
    fusion_dim: 512

### Privacy Configuration

privacy:
  epsilon: 0.1
  delta: 1.0e-5
  noise_multiplier: 1.0
  max_grad_norm: 1.0
  secure_aggregation: true
  adaptive_clipping: true

### Analysis Configuration

analysis:
  market_analysis:
    enabled: true
    metrics: ["price_prediction", "market_efficiency", "price_volatility"]
  visual_analysis:
    enabled: true
    cluster_analysis: true
  network_analysis:
    enabled: true
    centrality_metrics: true
  cross_modal_analysis:
    enabled: true
    correlation_analysis: true

## Privacy Guarantees

PrivaMod implements Rényi Differential Privacy with:

- Target ε = 0.08
- Target δ = 1e-5

Privacy mechanisms include:

1. **DP-SGD**: Implements per-sample gradient clipping and calibrated noise addition
2. **Adaptive Clipping**: Automatically adjusts clipping thresholds based on observed gradient norms
3. **Privacy Budget Accounting**: Precisely tracks privacy expenditure across training iterations
4. **Secure Aggregation**: Protects intermediate computations with cryptographic techniques

## Performance Metrics

### Market Efficiency

PrivaMod achieves a market efficiency score of 0.874, representing a 13.4% improvement over baseline approaches.

### Prediction Accuracy

- R² Score: 0.912
- MAE: 0.0298 ETH
- MAPE: 7.23%

### Privacy Protection

- Membership inference attack success rate: 53.4% (near random guessing)
- Model inversion attack reconstruction error: High (indicates strong privacy)
- Attribute inference attack success rate: Near baseline

## Reproducing Results

To reproduce the results from our paper:

1. Download the CryptoPunks dataset from [our data repository]([https://example.com/cryptopunks](https://www.kaggle.com/datasets/tunguz/cryptopunks))
2. Run the data preprocessing pipeline using `scripts/preprocess_data.py`
3. Train the model using the configuration in `configs/paper_config.yaml`
4. Evaluate results using `scripts/evaluate.py`

All random seeds are fixed for reproducibility.

## Dataset

The evaluation was performed on the CryptoPunks NFT collection, comprising:
- 10,000 unique NFT images
- 167,492 transactions from 2017 to 2021
- Price distribution with high skewness (4.7) and kurtosis (23.8)
- Mean price of 29.39 ETH and median price of 15.40 ETH
