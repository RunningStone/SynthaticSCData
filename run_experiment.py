"""
Main Experiment Runner

Executes the complete experimental workflow for Phase 1: Simplified Verification
"""

import yaml
import torch
import numpy as np
from pathlib import Path
import argparse

from Data import (
    DistributionParameterizer,
    PotentialFunction,
    TrajectorySampler,
    DatasetConstructor
)
from Model import OptimalTransportModel, SchrodingerBridgeModel, VAEModel
from Trainer import ModelTrainer, ModelEvaluator
from Analyser import (
    DataQualityMonitor,
    TrajectoryVisualizer,
    ModelComparisonVisualizer,
    GeneralizationVisualizer,
    StatisticalReportGenerator
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def phase1_simplified_verification(config: dict, output_dir: str):
    """
    Phase 1: Simplified Verification
    
    - Single topology (2-mode final state)
    - Geometric extrapolation
    - Train OT, SB, VAE models
    - Compare generalization performance
    """
    print("="*80)
    print("PHASE 1: SIMPLIFIED VERIFICATION")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    seed = config['experiment']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # ========== Step 1: Generate Datasets ==========
    print("\n[Step 1/5] Generating datasets...")
    
    dataset_constructor = DatasetConstructor(config, seed=seed)
    
    # Generate training dataset
    train_dataset = dataset_constructor.construct_dataset(
        dataset_type='train',
        save_path=output_path / 'data' / 'train_dataset.h5'
    )
    
    # Generate test dataset
    test_dataset = dataset_constructor.construct_dataset(
        dataset_type='test',
        save_path=output_path / 'data' / 'test_dataset.h5'
    )
    
    print(f"Train dataset: {train_dataset['trajectories'].shape}")
    print(f"Test dataset: {test_dataset['trajectories'].shape}")
    
    # ========== Step 2: Quality Control ==========
    print("\n[Step 2/5] Data quality control...")
    
    monitor = DataQualityMonitor(config)
    train_quality = monitor.check_dataset(train_dataset)
    test_quality = monitor.check_dataset(test_dataset)
    
    monitor.save_report(
        train_quality,
        output_path / 'analysis' / 'train_quality_report.json'
    )
    monitor.save_report(
        test_quality,
        output_path / 'analysis' / 'test_quality_report.json'
    )
    
    # ========== Step 3: Train Models ==========
    print("\n[Step 3/5] Training models...")
    
    device = config['experiment']['device']
    dimension = config['data']['dimension']
    
    # Initialize models
    models = {
        'ot': OptimalTransportModel(
            dimension=dimension,
            **config['models']['ot']
        ),
        'sb': SchrodingerBridgeModel(
            dimension=dimension,
            diffusion_coeff=config['data']['diffusion_coeff'],
            **config['models']['sb']
        ),
        'vae': VAEModel(
            dimension=dimension,
            **config['models']['vae']
        )
    }
    
    # Train each model
    trained_models = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name.upper()} model...")
        
        trainer = ModelTrainer(model, model_name, config, device)
        history = trainer.train(
            train_dataset,
            val_dataset=None,  # Could split train_dataset for validation
            save_dir=output_path / 'checkpoints'
        )
        
        trained_models[model_name] = model
    
    # ========== Step 4: Evaluate Models ==========
    print("\n[Step 4/5] Evaluating models...")
    
    evaluator = ModelEvaluator(trained_models['ot'], 'ot', device)
    comparison_results = evaluator.compare_models(
        trained_models,
        test_dataset,
        n_samples=config['evaluation']['n_samples']
    )
    
    # Save comparison results
    evaluator.save_results(
        comparison_results,
        output_path / 'results' / 'model_comparison.json'
    )
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for model_name in ['ot', 'sb', 'vae']:
        if model_name in comparison_results:
            path_error = comparison_results[model_name]['path']['mean_error']
            print(f"{model_name.upper()}: Path Error = {path_error:.6f}")
    
    if 'path_information_gain' in comparison_results:
        gain = comparison_results['path_information_gain']
        print(f"\nPath Information Gain (ΔL = L_OT - L_SB): {gain:.6f}")
    
    # ========== Step 5: Visualization ==========
    print("\n[Step 5/5] Generating visualizations...")
    
    # Trajectory visualization
    traj_viz = TrajectoryVisualizer(config)
    traj_viz.plot_trajectories(
        train_dataset,
        save_path=output_path / 'figures' / 'train_trajectories.png'
    )
    
    # Model comparison visualization
    comp_viz = ModelComparisonVisualizer(config)
    comp_viz.plot_comparison(
        comparison_results,
        test_dataset,
        save_path=output_path / 'figures' / 'model_comparison.png'
    )
    
    # Generalization analysis
    gen_viz = GeneralizationVisualizer(config)
    gen_viz.plot_generalization(
        comparison_results,
        save_path=output_path / 'figures' / 'generalization_analysis.png'
    )
    
    # Statistical report
    report_gen = StatisticalReportGenerator(config)
    report_gen.generate_report(
        comparison_results,
        save_path=output_path / 'report.md'
    )
    
    print("\n" + "="*80)
    print(f"Experiment complete! Results saved to: {output_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Run Schrödinger Bridge Experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/naive_compare',
        help='Output directory'
    )
    parser.add_argument(
        '--phase',
        type=str,
        default='1',
        choices=['1', '2', '3', '4'],
        help='Experiment phase to run'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run experiment
    if args.phase == '1':
        phase1_simplified_verification(config, args.output)
    else:
        print(f"Phase {args.phase} not yet implemented")


if __name__ == '__main__':
    main()
