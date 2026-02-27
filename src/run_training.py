"""
Run all training experiments sequentially.
Trains proposed model + baselines on all available datasets.

This script supports resuming: already-completed experiments are skipped
and results are saved incrementally after each run.

Usage:
    python src/run_training.py --datasets_dir datasets --results_dir results
    python src/run_training.py --datasets_dir /data/datasets --results_dir ./results --epochs 50
"""
import os
import sys
import json
import argparse
import traceback
import torch
import time
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))
from train import run_experiment, run_two_phase_experiment
from models import get_model, count_parameters

MODELS = ['ecamgnet', 'mobilenetv2', 'efficientnet_b0', 'shufflenetv2', 'resnet18']


def discover_datasets(datasets_dir):
    """Find valid dataset directories (containing at least 2 class subdirs)."""
    datasets = []
    datasets_dir = Path(datasets_dir)
    if not datasets_dir.is_dir():
        return datasets
    for d in sorted(datasets_dir.iterdir()):
        if d.is_dir() and not d.name.endswith('_raw'):
            subdirs = [s for s in d.iterdir() if s.is_dir()]
            if len(subdirs) >= 2:
                datasets.append(d)
    return datasets


def run_all(datasets_dir, results_dir, config):
    """Run all experiments, resuming from any previously saved results."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(datasets_dir)
    if not datasets:
        print(f"ERROR: No datasets found in '{datasets_dir}'.")
        return

    print(f"Found {len(datasets)} datasets:")
    for d in datasets:
        subdirs = [s for s in d.iterdir() if s.is_dir()]
        n_imgs = sum(len(list(s.glob('*.*'))) for s in subdirs)
        print(f"  {d.name}: {len(subdirs)} classes, {n_imgs} images")

    # Load any existing results for resumption
    results_file = results_dir / "all_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for dataset_path in datasets:
        dataset_name = dataset_path.name
        if dataset_name not in all_results:
            all_results[dataset_name] = {}

        print(f"\n{'#' * 70}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#' * 70}")

        for model_name in MODELS:
            # Skip if already done
            if model_name in all_results[dataset_name] and \
                    'accuracy' in all_results[dataset_name].get(model_name, {}):
                acc = all_results[dataset_name][model_name]['accuracy']
                print(f"\n--- {model_name} on {dataset_name}: ALREADY DONE "
                      f"(acc={acc:.4f}) ---")
                continue

            save_dir = results_dir / dataset_name / model_name
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- Training {model_name} on {dataset_name} ---")
            try:
                if model_name == 'ecamgnet':
                    test_metrics, history, info = run_two_phase_experiment(
                        str(dataset_path), str(save_dir),
                        img_size=config['img_size'],
                        batch_size=config['batch_size'],
                        width_mult=config['width_mult'],
                    )
                else:
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

                model = get_model(model_name, info['num_classes'],
                                  pretrained=False, width_mult=config['width_mult'])
                params = count_parameters(model)

                all_results[dataset_name][model_name] = {
                    'accuracy': test_metrics['accuracy'],
                    'precision': test_metrics['precision_macro'],
                    'recall': test_metrics['recall_macro'],
                    'f1': test_metrics['f1_macro'],
                    'auc': test_metrics['auc'],
                    'parameters': params,
                    'confusion_matrix': test_metrics['confusion_matrix'],
                    'history': history,
                    'dataset_info': info,
                }
                print(f"  Result: Acc={test_metrics['accuracy']:.4f}, "
                      f"F1={test_metrics['f1_macro']:.4f}, Params={params:,}")

                # Incremental save
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                all_results[dataset_name][model_name] = {'error': str(e)}

    # Final save
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 100}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 100}")
    for dataset_name, results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'Params':>12}")
        print("-" * 52)
        for model_name in MODELS:
            if model_name in results and 'accuracy' in results.get(model_name, {}):
                r = results[model_name]
                print(f"{model_name:<20} {r['accuracy']:>10.4f} "
                      f"{r['f1']:>10.4f} {r['parameters']:>12,}")
            elif model_name in results:
                print(f"{model_name:<20} {'ERROR':>10}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run all training experiments (with resume support).'
    )
    parser.add_argument('--datasets_dir', type=str, default='datasets',
                        help='Root directory containing dataset subdirectories. '
                             'Default: datasets/')
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
