"""
PrivaMod: Advanced Privacy-Preserving Multimodal Integration for Digital Assets
===========================================================================

Main entry point for training, evaluation, and analysis with the PrivaMod system.
Provides a command-line interface for all system functionality with robust
error handling and comprehensive logging.
"""

import argparse
import logging
import yaml
import torch
import os
from datetime import datetime
from pathlib import Path

from models import create_model, ModelRegistry
from data import create_dataset, DataRegistry
from train import Trainer
from privacy import PrivacyEngine
from analysis import MarketAnalyzer, generate_report
from utils import setup_logging, seed_everything

def parse_args():
    """Parse command-line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(description="PrivaMod: Privacy-preserving Multimodal Integration for Digital Assets")
    
    # Core arguments
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'analyze', 'benchmark'], default='train',
                        help='Operation mode')
    
    # Optional overrides
    parser.add_argument('--model', type=str, help='Model architecture to use (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--privacy_epsilon', type=float, help='Privacy budget epsilon (overrides config)')
    parser.add_argument('--experiment_name', type=str, help='Experiment name for logging')
    parser.add_argument('--output_dir', type=str, help='Output directory (overrides config)')
    
    # System configuration
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()

def load_config(config_path):
    """Load and validate configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    required_sections = ['model', 'data', 'training', 'privacy', 'system']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
            
    return config

def update_config_with_args(config, args):
    """Update configuration with command-line arguments."""
    # Override model if specified
    if args.model:
        config['model']['architecture'] = args.model
    
    # Override training parameters
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
        
    # Override privacy parameters
    if args.privacy_epsilon:
        config['privacy']['epsilon'] = args.privacy_epsilon
        
    # Override output directory
    if args.output_dir:
        config['system']['output_dir'] = args.output_dir
        
    # Update GPU count
    config['system']['num_gpus'] = args.gpus
    
    # Update random seed
    config['system']['seed'] = args.seed
    
    return config

def setup_experiment(config, args):
    """Set up experiment directory and logging."""
    # Create experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"PrivaMod_{config['model']['architecture']}_{timestamp}"
    
    # Create output directory
    output_dir = Path(config['system']['output_dir']) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "experiment.log"
    logger = setup_logging(log_file, debug=args.debug)
    
    # Save configuration
    config_file = output_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_dir, logger

def train(config, output_dir, logger):
    """Train the model with specified configuration."""
    logger.info("Starting training process")
    
    # Set random seed for reproducibility
    seed_everything(config['system']['seed'])
    
    # Create dataset
    train_loader, val_loader, test_loader = create_dataset(
        config['data'],
        batch_size=config['training']['batch_size'],
        num_workers=config['system']['num_workers']
    )
    
    # Create model
    model = create_model(config['model'])
    
    # Initialize privacy engine
    privacy_engine = PrivacyEngine(
        config['privacy'],
        data_size=len(train_loader.dataset)
    )
    
    # Apply privacy mechanisms to model
    model = privacy_engine.make_private(model)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config['training'],
        output_dir=output_dir,
        privacy_engine=privacy_engine
    )
    
    # Train model
    best_model_path = trainer.train()
    
    logger.info(f"Training completed. Best model saved at: {best_model_path}")
    return best_model_path

def evaluate(config, model_path, output_dir, logger):
    """Evaluate the model with specified configuration."""
    logger.info(f"Starting evaluation using model: {model_path}")
    
    # Create dataset
    _, _, test_loader = create_dataset(
        config['data'],
        batch_size=config['training']['batch_size'],
        num_workers=config['system']['num_workers']
    )
    
    # Load model
    model = create_model(config['model'])
    model.load_state_dict(torch.load(model_path))
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        test_loader=test_loader,
        config=config['training'],
        output_dir=output_dir
    )
    
    # Evaluate model
    metrics = trainer.evaluate()
    
    # Save metrics
    metrics_file = output_dir / "evaluation_metrics.yaml"
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    logger.info(f"Evaluation completed. Results saved to: {metrics_file}")
    return metrics

def analyze(config, model_path, output_dir, logger):
    """Analyze the market with the trained model."""
    logger.info(f"Starting market analysis using model: {model_path}")
    
    # Create dataset
    _, _, test_loader = create_dataset(
        config['data'],
        batch_size=config['training']['batch_size'],
        num_workers=config['system']['num_workers']
    )
    
    # Load model
    model = create_model(config['model'])
    model.load_state_dict(torch.load(model_path))
    
    # Initialize market analyzer
    analyzer = MarketAnalyzer(
        model=model,
        data_loader=test_loader,
        config=config['analysis'],
        output_dir=output_dir
    )
    
    # Run analysis
    results = analyzer.analyze()
    
    # Generate report
    report_path = generate_report(
        results=results,
        config=config,
        output_dir=output_dir
    )
    
    logger.info(f"Analysis completed. Report generated at: {report_path}")
    return report_path

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command-line arguments
    config = update_config_with_args(config, args)
    
    # Setup experiment
    output_dir, logger = setup_experiment(config, args)
    
    try:
        # Execute based on mode
        if args.mode == 'train':
            model_path = train(config, output_dir, logger)
            metrics = evaluate(config, model_path, output_dir, logger)
            analyze(config, model_path, output_dir, logger)
            
        elif args.mode == 'evaluate':
            model_path = str(Path(output_dir) / "best_model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at: {model_path}")
            metrics = evaluate(config, model_path, output_dir, logger)
            
        elif args.mode == 'analyze':
            model_path = str(Path(output_dir) / "best_model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at: {model_path}")
            report_path = analyze(config, model_path, output_dir, logger)
            
        elif args.mode == 'benchmark':
            from utils import benchmark_system
            benchmark_results = benchmark_system(config, output_dir)
            logger.info(f"Benchmark completed. Results saved to: {output_dir / 'benchmark.yaml'}")
            
        logger.info(f"PrivaMod execution completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()