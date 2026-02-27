"""
Training Engine for Image Classification Models
Supports training on GPU with comprehensive logging and evaluation
"""
import os
import sys
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders
from models import get_model, count_parameters


class Trainer:
    """Training engine with comprehensive metrics tracking."""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 num_classes, class_names, device, save_dir,
                 lr=0.001, epochs=50, patience=10, model_name='model'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.epochs = epochs
        self.patience = patience

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        # Tracking
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.epochs_no_improve = 0

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        total = len(all_labels)
        epoch_loss = running_loss / total
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = {
            'loss': epoch_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probs.tolist(),
        }

        # Per-class metrics
        report = classification_report(all_labels, all_preds,
                                       target_names=self.class_names,
                                       output_dict=True, zero_division=0)
        metrics['per_class'] = report

        # AUC if binary or multiclass
        try:
            if self.num_classes == 2:
                metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                metrics['auc'] = roc_auc_score(all_labels, all_probs,
                                               multi_class='ovr', average='macro')
        except Exception:
            metrics['auc'] = 0.0

        return metrics

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"Training {self.model_name}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}, Patience: {self.patience}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_metrics = self.evaluate(self.val_loader)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']

            # Learning rate step
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Print progress
            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                torch.save(self.best_model_state,
                           self.save_dir / f"{self.model_name}_best.pth")
                print(f"  -> New best val accuracy: {val_acc:.4f}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        # Load best model for testing
        self.model.load_state_dict(self.best_model_state)
        return self.best_val_acc

    def test(self):
        """Run final evaluation on test set."""
        print(f"\n{'=' * 60}")
        print(f"Testing {self.model_name}")
        print(f"{'=' * 60}\n")

        test_metrics = self.evaluate(self.test_loader)

        print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision_macro']:.4f}")
        print(f"Test Recall:    {test_metrics['recall_macro']:.4f}")
        print(f"Test F1-Score:  {test_metrics['f1_macro']:.4f}")
        print(f"Test AUC:       {test_metrics['auc']:.4f}")

        # Save results
        results = {
            'model_name': self.model_name,
            'parameters': count_parameters(self.model),
            'best_val_acc': self.best_val_acc,
            'test_metrics': {k: v for k, v in test_metrics.items()
                             if k not in ['predictions', 'labels', 'probabilities']},
            'history': self.history,
        }

        with open(self.save_dir / f"{self.model_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Save detailed predictions
        predictions = {
            'predictions': test_metrics['predictions'],
            'labels': test_metrics['labels'],
            'probabilities': test_metrics['probabilities'],
            'class_names': self.class_names,
        }
        with open(self.save_dir / f"{self.model_name}_predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2)

        return test_metrics


def run_experiment(data_dir, model_name, save_dir, img_size=224,
                   batch_size=32, epochs=50, lr=0.001, patience=10,
                   width_mult=1.0, pretrained=True):
    """Run a complete training experiment (single-phase, used for baselines)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader, info = create_dataloaders(
        data_dir, img_size=img_size, batch_size=batch_size, num_workers=0
    )
    print(f"Dataset: {data_dir}")
    print(f"Classes: {info['num_classes']} - {info['class_names']}")
    print(f"Samples: Train={info['train_samples']}, Val={info['val_samples']}, Test={info['test_samples']}")

    # Create model
    model = get_model(model_name, info['num_classes'],
                      pretrained=pretrained, width_mult=width_mult)
    print(f"Model: {model_name}, Parameters: {count_parameters(model):,}")

    # Train
    trainer = Trainer(
        model, train_loader, val_loader, test_loader,
        info['num_classes'], info['class_names'], device,
        save_dir, lr=lr, epochs=epochs, patience=patience,
        model_name=model_name
    )
    trainer.train()
    test_metrics = trainer.test()

    return test_metrics, trainer.history, info


def run_two_phase_experiment(data_dir, save_dir, img_size=224, batch_size=32,
                             width_mult=1.0):
    """Run two-phase transfer learning for ECA-MGNet.

    Phase 1: Freeze pretrained backbone, train custom head only.
             5 epochs, LR=3e-3, cosine annealing (eta_min=1e-4).
    Phase 2: Unfreeze all parameters, fine-tune end-to-end.
             Up to 60 epochs, LR=1e-3, cosine annealing (eta_min=1e-6),
             early stopping with patience 20.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, info = create_dataloaders(
        data_dir, img_size=img_size, batch_size=batch_size, num_workers=0
    )
    print(f"Dataset: {data_dir}")
    print(f"Classes: {info['num_classes']} - {info['class_names']}")
    print(f"Samples: Train={info['train_samples']}, Val={info['val_samples']}, Test={info['test_samples']}")

    model = get_model('ecamgnet', info['num_classes'],
                      pretrained=True, width_mult=width_mult)
    model = model.to(device)
    print(f"Model: ECA-MGNet, Parameters: {count_parameters(model):,}")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    combined_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }

    # ---- Phase 1: Freeze backbone, train head only ----
    print(f"\n{'=' * 60}")
    print("Phase 1: Training custom head (backbone frozen)")
    print(f"{'=' * 60}")

    for param in model.stem.parameters():
        param.requires_grad = False
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters (Phase 1): {sum(p.numel() for p in trainable):,}")

    optimizer1 = optim.AdamW(trainable, lr=3e-3, weight_decay=1e-4)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=5, eta_min=1e-4)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(5):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer1.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer1.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        lr_now = optimizer1.param_groups[0]['lr']
        scheduler1.step()

        combined_history['train_loss'].append(train_loss)
        combined_history['train_acc'].append(train_acc)
        combined_history['val_loss'].append(val_loss)
        combined_history['val_acc'].append(val_acc)
        combined_history['lr'].append(lr_now)

        print(f"  Phase1 Epoch [{epoch+1}/5] "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # ---- Phase 2: Unfreeze all, fine-tune end-to-end ----
    print(f"\n{'=' * 60}")
    print("Phase 2: Fine-tuning all parameters")
    print(f"{'=' * 60}")

    for param in model.parameters():
        param.requires_grad = True
    print(f"Trainable parameters (Phase 2): {count_parameters(model):,}")

    optimizer2 = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=60, eta_min=1e-6)

    patience_counter = 0

    for epoch in range(60):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer2.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer2.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        lr_now = optimizer2.param_groups[0]['lr']
        scheduler2.step()

        combined_history['train_loss'].append(train_loss)
        combined_history['train_acc'].append(train_acc)
        combined_history['val_loss'].append(val_loss)
        combined_history['val_acc'].append(val_acc)
        combined_history['lr'].append(lr_now)

        print(f"  Phase2 Epoch [{epoch+1}/60] "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, save_path / "ecamgnet_best.pth")
            print(f"    -> New best val accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"\n  Early stopping at Phase 2 epoch {epoch+1}")
                break

    # Load best model and test
    model.load_state_dict(best_state)
    torch.save(best_state, save_path / "ecamgnet_best.pth")

    # Create a temporary Trainer just for testing
    trainer = Trainer(
        model, train_loader, val_loader, test_loader,
        info['num_classes'], info['class_names'], device,
        str(save_path), lr=1e-3, epochs=1, patience=1,
        model_name='ecamgnet'
    )
    trainer.best_val_acc = best_val_acc
    trainer.best_model_state = best_state
    trainer.history = combined_history
    test_metrics = trainer.test()

    return test_metrics, combined_history, info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='ecamgnet',
                        choices=['ecamgnet', 'mobilenetv2', 'efficientnet_b0',
                                 'shufflenetv2', 'resnet18', 'ghostnet'])
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--two_phase', action='store_true',
                        help='Use two-phase transfer learning (for ECA-MGNet)')
    args = parser.parse_args()

    if args.two_phase or args.model == 'ecamgnet':
        run_two_phase_experiment(
            args.data_dir, args.save_dir,
            img_size=args.img_size, batch_size=args.batch_size,
            width_mult=args.width_mult
        )
    else:
        run_experiment(
            args.data_dir, args.model, args.save_dir,
            img_size=args.img_size, batch_size=args.batch_size,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            width_mult=args.width_mult, pretrained=True
        )
