
"""
Training Pipeline for Multimodal EEG-Eye Emotion Recognition
Implements training, validation, and testing procedures for MAET model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time
from datetime import datetime
import json
import os

from ..models.maet_model import MAET
from ..data.seedvii_dataset import SEEDVII_Dataset
from ..utils.safe_forward import safe_model_forward
from .evaluation_utils import compute_metrics, plot_confusion_matrix


class MultimodalTrainer:
    """
    Main trainer class for multimodal emotion recognition
    
    Handles training loop, validation, model saving, and experiment logging
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config (dict): Training configuration parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_val_acc = 0.0
        self.training_history = []
        
        # Set random seeds for reproducibility
        self._set_seeds(config.get('seed', 42))
    
    def _set_seeds(self, seed):
        """Set random seeds for reproducible results"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def prepare_data(self, data_dir, modality='multimodal', subset_ratio=1.0):
        """
        Load and prepare dataset with train/validation split
        
        Args:
            data_dir (str): Path to dataset
            modality (str): Data modality to use
            subset_ratio (float): Fraction of dataset to use
            
        Returns:
            tuple: (train_loader, val_loader, dataset)
        """
        print(f"Loading {modality} dataset from {data_dir}")
        
        # Load dataset
        dataset = SEEDVII_Dataset(
            data_dir=data_dir,
            modality=modality,
            subset_ratio=subset_ratio
        )
        
        # Create train/validation split
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=self.config.get('val_split', 0.2),
            stratify=dataset.emotion_labels,
            random_state=self.config.get('seed', 42)
        )
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader, dataset
    
    def build_model(self, dataset):
        """
        Build MAET model based on dataset specifications
        
        Args:
            dataset: Dataset object with feature dimensions
            
        Returns:
            MAET: Initialized model
        """
        model_config = self.config.get('model', {})
        
        model = MAET(
            eeg_dim=dataset.eeg_feature_dim if hasattr(dataset, 'eeg_feature_dim') else 310,
            eye_dim=dataset.eye_feature_dim if hasattr(dataset, 'eye_feature_dim') else 33,
            num_classes=dataset.num_classes if hasattr(dataset, 'num_classes') else 7,
            embed_dim=model_config.get('embed_dim', 32),
            depth=model_config.get('depth', 3),
            num_heads=model_config.get('num_heads', 4),
            domain_generalization=model_config.get('domain_generalization', False),
            num_domains=dataset.num_subjects if hasattr(dataset, 'num_subjects') else 20
        )
        
        return model.to(self.device)
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """
        Train model for one epoch
        
        Args:
            model: MAET model
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, additional = safe_model_forward(
                model, 
                eeg=batch.get('eeg'), 
                eye=batch.get('eye'), 
                alpha_=self.config.get('gradient_reversal_alpha', 0.5)
            )
            
            # Compute losses
            emotion_loss = criterion(predictions, batch['label'])
            total_loss_batch = emotion_loss
            
            # Domain adversarial loss if enabled
            if additional and len(additional) > 0:
                domain_loss = criterion(additional[0], batch['subject'])
                domain_weight = self.config.get('domain_loss_weight', 0.1)
                total_loss_batch = emotion_loss + domain_weight * domain_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping for stability
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config['grad_clip']
                )
            
            optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            _, predicted = torch.max(predictions.data, 1)
            total += batch['label'].size(0)
            correct += (predicted == batch['label']).sum().item()
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss_batch.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, model, val_loader, criterion):
        """
        Validate model performance
        
        Args:
            model: MAET model
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            tuple: (loss, accuracy, f1_score, predictions, labels)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                predictions, _ = safe_model_forward(
                    model, 
                    eeg=batch.get('eeg'), 
                    eye=batch.get('eye')
                )
                
                loss = criterion(predictions, batch['label'])
                total_loss += loss.item()
                
                _, predicted = torch.max(predictions.data, 1)
                total += batch['label'].size(0)
                correct += (predicted == batch['label']).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
        
        return avg_loss, accuracy, f1, all_predictions, all_labels
    
    def train(self, data_dir, modality='multimodal', subset_ratio=1.0):
        """
        Main training loop
        
        Args:
            data_dir (str): Path to dataset
            modality (str): Data modality
            subset_ratio (float): Dataset fraction to use
            
        Returns:
            dict: Training results and model
        """
        print(f"Starting training with {modality} modality")
        print(f"Device: {self.device}")
        
        # Prepare data
        train_loader, val_loader, dataset = self.prepare_data(
            data_dir, modality, subset_ratio
        )
        
        # Build model
        model = self.build_model(dataset)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup optimization
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('num_epochs', 100),
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.get('num_epochs', 100)):
            print(f"\\nEpoch {epoch+1}/{self.config.get('num_epochs', 100)}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion
            )
            
            # Validate
            val_loss, val_acc, val_f1, val_preds, val_labels = self.validate(
                model, val_loader, criterion
            )
            
            # Update learning rate
            scheduler.step()
            
            # Log results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            self.training_history.append(epoch_results)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.config.get('save_model', True):
                    self._save_model(model, epoch, val_acc, modality)
            
            # Early stopping
            if self.config.get('early_stopping', 0) > 0:
                if self._should_early_stop():
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        print(f"\\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return {
            'model': model,
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history,
            'training_time': training_time
        }
    
    def _save_model(self, model, epoch, accuracy, modality):
        """Save model checkpoint"""
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoints/maet_{modality}_{timestamp}_acc{accuracy:.2f}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'config': self.config,
            'training_history': self.training_history
        }, filename)
        
        print(f"Model saved: {filename}")
    
    def _should_early_stop(self):
        """Check if training should stop early based on validation performance"""
        patience = self.config.get('early_stopping', 0)
        if patience <= 0:
            return False
        
        if len(self.training_history) < patience:
            return False
        
        recent_accs = [h['val_acc'] for h in self.training_history[-patience:]]
        return all(acc <= self.best_val_acc for acc in recent_accs)


def train_subject_dependent(config, data_dir, modality='multimodal', subset_ratio=0.01):
    """
    Subject-dependent training experiment
    
    Args:
        config (dict): Training configuration
        data_dir (str): Dataset directory
        modality (str): Data modality
        subset_ratio (float): Dataset subset ratio
        
    Returns:
        dict: Experiment results
    """
    print(f"\\n=== SUBJECT-DEPENDENT EXPERIMENT ({modality.upper()}) ===")
    
    # Load full dataset to get subject information
    dataset = SEEDVII_Dataset(data_dir, modality, subset_ratio)
    unique_subjects = np.unique(dataset.subject_labels)
    all_results = []
    
    for subject_id in unique_subjects[:5]:  # Test first 5 subjects for speed
        print(f"\\nSubject {subject_id}...")
        
        # Get subject-specific data
        subject_mask = dataset.subject_labels == subject_id
        subject_indices = np.where(subject_mask)[0]
        
        if len(subject_indices) < 10:  # Skip subjects with too few samples
            continue
        
        # Subject-specific train/validation split
        train_indices, val_indices = train_test_split(
            subject_indices, 
            test_size=0.3, 
            random_state=42,
            stratify=dataset.emotion_labels[subject_indices]
        )
        
        # Create subject-specific dataset
        subject_dataset = Subset(dataset, list(subject_indices))
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        # Train model
        trainer = MultimodalTrainer(config)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        model = trainer.build_model(dataset)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        for epoch in range(20):  # Quick training
            train_loss, train_acc = trainer.train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_f1, _, _ = trainer.validate(model, val_loader, criterion)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Train {train_acc:.1f}%, Val {val_acc:.1f}%")
        
        all_results.append(best_val_acc)
        print(f"  Subject {subject_id} best accuracy: {best_val_acc:.2f}%")
    
    # Calculate overall results
    overall_avg = np.mean(all_results)
    overall_std = np.std(all_results)
    
    print(f"\\nRESULTS: {overall_avg:.2f}% ± {overall_std:.2f}%")
    return overall_avg, overall_std


def train_cross_subject(config, data_dir, modality='eeg', subset_ratio=0.01):
    """
    Cross-subject (Leave-One-Subject-Out) training experiment
    
    Args:
        config (dict): Training configuration
        data_dir (str): Dataset directory
        modality (str): Data modality
        subset_ratio (float): Dataset subset ratio
        
    Returns:
        dict: Experiment results
    """
    print(f"\\n=== CROSS-SUBJECT EXPERIMENT ({modality.upper()}) ===")
    
    dataset = SEEDVII_Dataset(data_dir, modality, subset_ratio)
    unique_subjects = np.unique(dataset.subject_labels)
    all_results = []
    
    for test_subject in unique_subjects[:3]:  # Test first 3 subjects for speed
        print(f"\\nTest Subject {test_subject}...")
        
        try:
            # Leave-one-subject-out split
            train_mask = dataset.subject_labels != test_subject
            test_mask = dataset.subject_labels == test_subject
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(test_indices) < 5:
                continue
            
            # Create domain mapping for training subjects
            training_subjects = np.unique(dataset.subject_labels[train_mask])
            subject_to_domain = {subject_id: domain_id for domain_id, subject_id in enumerate(training_subjects)}
            
            print(f"  Training subjects: {len(training_subjects)}")
            
            # Create data loaders
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
            
            # Create model with domain adaptation
            model = MAET(
                eeg_dim=310 if modality in ['eeg', 'multimodal'] else 0,
                eye_dim=33 if modality in ['eye', 'multimodal'] else 0,
                embed_dim=32,
                depth=3,
                num_heads=4,
                domain_generalization=True,
                num_domains=len(training_subjects)
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            trainer = MultimodalTrainer(config)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Training with domain adaptation
            for epoch in range(30):
                train_loss, train_acc = trainer.train_epoch(model, train_loader, optimizer, criterion)
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}: Train {train_acc:.1f}%")
            
            # Test
            _, test_acc, test_f1, _, _ = trainer.validate(model, test_loader, criterion)
            all_results.append(test_acc)
            print(f"  Subject {test_subject} test accuracy: {test_acc:.2f}%")
            
        except Exception as e:
            print(f"  Error with subject {test_subject}: {e}")
            continue
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
    
    # Calculate results
    if all_results:
        overall_avg = np.mean(all_results)
        overall_std = np.std(all_results)
        print(f"\\nRESULTS: {overall_avg:.2f}% ± {overall_std:.2f}%")
        return overall_avg, overall_std
    else:
        return 0.0, 0.0
```
