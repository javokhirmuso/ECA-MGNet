"""
Run all experiments: Train proposed model and baselines on all datasets.
Collects results for paper writing.

Usage:
    python src/run_all_experiments.py --datasets_dir datasets --results_dir results
    python src/run_all_experiments.py --datasets_dir /path/to/datasets --results_dir ./results --epochs 50
"""
import os
import sys
import json
import argparse
import torch
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import run_experiment, run_two_phase_experiment
from models import get_model, count_parameters

# Models to evaluate
MODELS = ['ecamgnet', 'mobilenetv2', 'efficientnet_b0', 'shufflenetv2', 'resnet18']

# Expected dataset directory names
DATASET_NAMES = ['flowers102', 'dtd', 'food101', 'eurosat']


def discover_datasets(datasets_dir):
    """Find target dataset directories within the given root."""
    datasets = []
    for name in DATASET_NAMES:
        d = Path(datasets_dir) / name
        if d.is_dir():
            subdirs = [s for s in d.iterdir() if s.is_dir()]
            if len(subdirs) >= 2:
                datasets.append(d)
    return datasets


def run_all(datasets_dir, results_dir, config):
    """Run all experiments and collect results."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(datasets_dir)
    if not datasets:
        print(f"ERROR: No datasets found in '{datasets_dir}'. Please download datasets first.")
        print(f"Expected subdirectories: {DATASET_NAMES}")
        return

    print(f"Found {len(datasets)} datasets:")
    for d in datasets:
        print(f"  - {d.name}")

    all_results = {}

    for dataset_path in datasets:
        dataset_name = dataset_path.name
        print(f"\n{'#' * 70}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#' * 70}")

        dataset_results = {}

        for model_name in MODELS:
            save_dir = results_dir / dataset_name / model_name
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- Training {model_name} on {dataset_name} ---")
            try:
                if model_name == 'ecamgnet':
                    # ECA-MGNet uses two-phase transfer learning
                    test_metrics, history, info = run_two_phase_experiment(
                        str(dataset_path), str(save_dir),
                        img_size=config['img_size'],
                        batch_size=config['batch_size'],
                        width_mult=config['width_mult'],
                    )
                else:
                    # Baselines: single-phase, fine-tuned from ImageNet-pretrained
                    test_metrics, history, info = run_experiment(
                        str(dataset_path), model_name, str(save_dir),
                        img_size=config['img_size'],
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        lr=config['lr'],
                        patience=config['patience'],
                        width_mult=config['width_mult'],
                        pretrained=True,
                    )

                # Get parameter count
                model = get_model(model_name, info['num_classes'],
                                  pretrained=False, width_mult=config['width_mult'])
                params = count_parameters(model)

                dataset_results[model_name] = {
                    'accuracy': test_metrics['accuracy'],
                    'precision': test_metrics['precision_macro'],
                    'recall': test_metrics['recall_macro'],
                    'f1': test_metrics['f1_macro'],
                    'auc': test_metrics['auc'],
                    'parameters': params,
                    'confusion_matrix': test_metrics['confusion_matrix'],
                    'per_class': test_metrics.get('per_class', {}),
                    'history': history,
                    'dataset_info': info,
                }
                print(f"  Accuracy: {test_metrics['accuracy']:.4f}, "
                      f"F1: {test_metrics['f1_macro']:.4f}, "
                      f"Params: {params:,}")

            except Exception as e:
                print(f"  ERROR: {e}")
                dataset_results[model_name] = {'error': str(e)}

        all_results[dataset_name] = dataset_results

    # Save comprehensive results
    results_file = results_dir / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_file}")

    # Print summary table
    print_summary_table(all_results)

    return all_results


def print_summary_table(all_results):
    """Print a formatted summary table."""
    print(f"\n{'=' * 100}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 100}")

    for dataset_name, results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'Params':>12}")
        print("-" * 82)

        for model_name in MODELS:
            if model_name in results and 'error' not in results[model_name]:
                r = results[model_name]
                print(f"{model_name:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
                      f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f} "
                      f"{r['parameters']:>12,}")
            else:
                print(f"{model_name:<20} {'ERROR':>10}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run all training experiments for ECA-MGNet and baseline models.'
    )
    parser.add_argument('--datasets_dir', type=str, default='datasets',
                        help='Root directory containing dataset subdirectories '
                             '(flowers102, dtd, food101, eurosat). Default: datasets/')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save all results. Default: results/')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size. Default: 224')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size. Default: 32')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum training epochs. Default: 50')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate. Default: 0.001')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience. Default: 15')
    parser.add_argument('--width_mult', type=float, default=3.0,
                        help='Width multiplier for ECA-MGNet. Default: 3.0')
    args = parser.parse_args()

    config = {
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'patience': args.patience,
        'width_mult': args.width_mult,
    }

    run_all(args.datasets_dir, args.results_dir, config)
