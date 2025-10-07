
"""
Evaluation Utilities for Multimodal Emotion Recognition
Implements metrics computation, visualization, and analysis functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd


def compute_metrics(y_true, y_pred, num_classes=7, class_names=None):
    """
    Compute comprehensive evaluation metrics
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels  
        num_classes (int): Number of classes
        class_names (list): Names of emotion classes
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    if class_names is None:
        class_names = ['Neutral', 'Sad', 'Fear', 'Happy', 'Disgust', 'Surprise', 'Angry']
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro')
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None)
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name.lower()}_f1'] = f1_per_class[i]
        metrics[f'{class_name.lower()}_precision'] = precision_per_class[i]
        metrics[f'{class_name.lower()}_recall'] = recall_per_class[i]
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, 
                         title='Confusion Matrix', figsize=(10, 8)):
    """
    Plot confusion matrix with proper formatting
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): Class names for labels
        normalize (bool): Whether to normalize the matrix
        title (str): Plot title
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The confusion matrix plot
    """
    if class_names is None:
        class_names = ['Neutral', 'Sad', 'Fear', 'Happy', 'Disgust', 'Surprise', 'Angry']
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics over epochs
    
    Args:
        history (list): List of epoch results dictionaries
        save_path (str): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The training history plot
    """
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_per_subject_performance(predictions, labels, subjects, class_names=None):
    """
    Analyze performance across different subjects
    
    Args:
        predictions (np.ndarray): Model predictions
        labels (np.ndarray): True labels
        subjects (np.ndarray): Subject IDs
        class_names (list): Names of emotion classes
        
    Returns:
        pd.DataFrame: Per-subject performance results
    """
    if class_names is None:
        class_names = ['Neutral', 'Sad', 'Fear', 'Happy', 'Disgust', 'Surprise', 'Angry']
    
    unique_subjects = np.unique(subjects)
    results = []
    
    for subject in unique_subjects:
        mask = subjects == subject
        subj_pred = predictions[mask]
        subj_true = labels[mask]
        
        if len(subj_pred) > 0:
            metrics = compute_metrics(subj_true, subj_pred, class_names=class_names)
            
            result = {
                'subject': subject,
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'weighted_f1': metrics['weighted_f1'],
                'samples': len(subj_pred)
            }
            
            # Add per-class F1 scores
            for class_name in class_names:
                result[f'{class_name.lower()}_f1'] = metrics.get(f'{class_name.lower()}_f1', 0)
            
            results.append(result)
    
    return pd.DataFrame(results)


def plot_per_subject_performance(subject_results, metric='accuracy', figsize=(12, 6)):
    """
    Plot performance metric across subjects
    
    Args:
        subject_results (pd.DataFrame): Per-subject results from analyze_per_subject_performance
        metric (str): Metric to plot
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The performance plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    subjects = subject_results['subject']
    values = subject_results[metric]
    
    bars = ax.bar(subjects, values, alpha=0.7)
    
    # Add mean line
    mean_val = values.mean()
    ax.axhline(y=mean_val, color='r', linestyle='--', 
               label=f'Mean {metric}: {mean_val:.3f}')
    
    # Color bars based on performance
    colors = ['green' if v > mean_val else 'orange' for v in values]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title(f'{metric.capitalize()} Across Subjects')
    ax.set_xlabel('Subject ID')
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_emotion_confusion(y_true, y_pred, class_names=None):
    """
    Analyze which emotions are most commonly confused
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): Names of emotion classes
        
    Returns:
        pd.DataFrame: Confusion analysis results
    """
    if class_names is None:
        class_names = ['Neutral', 'Sad', 'Fear', 'Happy', 'Disgust', 'Surprise', 'Angry']
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Find most confused pairs
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true_emotion': class_names[i],
                    'predicted_emotion': class_names[j],
                    'count': cm[i, j],
                    'percentage': cm[i, j] / cm[i].sum() * 100
                })
    
    # Sort by confusion count
    confusion_df = pd.DataFrame(confusion_pairs)
    confusion_df = confusion_df.sort_values('count', ascending=False)
    
    return confusion_df


def evaluate_model_robustness(model, test_loaders, device, noise_levels=[0.0, 0.1, 0.2]):
    """
    Evaluate model robustness to input noise
    
    Args:
        model: Trained model
        test_loaders (dict): Dictionary of test dataloaders
        device: Computing device
        noise_levels (list): Noise levels to test
        
    Returns:
        dict: Robustness evaluation results
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for noise_level in noise_levels:
            all_predictions = []
            all_labels = []
            
            for loader_name, loader in test_loaders.items():
                for batch in loader:
                    # Move to device
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    
                    # Add noise
                    if noise_level > 0:
                        if 'eeg' in batch:
                            noise = torch.randn_like(batch['eeg']) * noise_level
                            batch['eeg'] = batch['eeg'] + noise
                        if 'eye' in batch:
                            noise = torch.randn_like(batch['eye']) * noise_level
                            batch['eye'] = batch['eye'] + noise
                    
                    # Forward pass
                    from ..utils.safe_forward import safe_model_forward
                    predictions, _ = safe_model_forward(
                        model, eeg=batch.get('eeg'), eye=batch.get('eye')
                    )
                    
                    _, predicted = torch.max(predictions, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch['label'].cpu().numpy())
            
            # Compute metrics for this noise level
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            
            results[f'noise_{noise_level}'] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': all_predictions,
                'labels': all_labels
            }
    
    return results


def generate_classification_report(y_true, y_pred, class_names=None, save_path=None):
    """
    Generate and save detailed classification report
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): Names of emotion classes
        save_path (str): Path to save the report
        
    Returns:
        str: Classification report string
    """
    if class_names is None:
        class_names = ['Neutral', 'Sad', 'Fear', 'Happy', 'Disgust', 'Surprise', 'Angry']
    
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        digits=4
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report


def compare_model_performance(results_dict, metric='accuracy', title='Model Comparison'):
    """
    Compare performance of different models or configurations
    
    Args:
        results_dict (dict): Dictionary of model names to results
        metric (str): Metric to compare
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: Comparison plot
    """
    model_names = list(results_dict.keys())
    values = [results_dict[name][metric] for name in model_names]
    errors = [results_dict[name].get(f'{metric}_std', 0) for name in model_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(model_names, values, yerr=errors, capsize=5, alpha=0.7)
    
    # Color bars based on performance
    best_value = max(values)
    colors = ['gold' if v == best_value else 'skyblue' for v in values]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title(title)
    ax.set_ylabel(metric.capitalize())
    ax.set_xlabel('Model')
    
    # Add value labels on bars
    for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


class EvaluationSuite:
    """
    Comprehensive evaluation suite for multimodal emotion recognition models
    """
    
    def __init__(self, class_names=None):
        """
        Initialize evaluation suite
        
        Args:
            class_names (list): Names of emotion classes
        """
        if class_names is None:
            self.class_names = ['Neutral', 'Sad', 'Fear', 'Happy', 'Disgust', 'Surprise', 'Angry']
        else:
            self.class_names = class_names
    
    def run_full_evaluation(self, model, test_loader, device, save_dir=None):
        """
        Run complete evaluation suite
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Computing device
            save_dir (str): Directory to save results
            
        Returns:
            dict: Complete evaluation results
        """
        model.eval()
        all_predictions = []
        all_labels = []
        all_subjects = []
        
        # Collect predictions
        with torch.no_grad():
            for batch in test_loader:
                for key in batch:
                    batch[key] = batch[key].to(device)
                
                from ..utils.safe_forward import safe_model_forward
                predictions, _ = safe_model_forward(
                    model, eeg=batch.get('eeg'), eye=batch.get('eye')
                )
                
                _, predicted = torch.max(predictions, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_subjects.extend(batch['subject'].cpu().numpy())
        
        # Compute comprehensive metrics
        results = {
            'overall_metrics': compute_metrics(
                all_labels, all_predictions, class_names=self.class_names
            ),
            'per_subject_results': analyze_per_subject_performance(
                np.array(all_predictions), np.array(all_labels), 
                np.array(all_subjects), self.class_names
            ),
            'confusion_analysis': analyze_emotion_confusion(
                all_labels, all_predictions, self.class_names
            ),
            'predictions': all_predictions,
            'labels': all_labels,
            'subjects': all_subjects
        }
        
        # Save results if directory provided
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Save confusion matrix plot
            fig_cm = plot_confusion_matrix(
                all_labels, all_predictions, self.class_names
            )
            fig_cm.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close(fig_cm)
            
            # Save per-subject performance plot
            fig_subj = plot_per_subject_performance(results['per_subject_results'])
            fig_subj.savefig(f'{save_dir}/per_subject_performance.png', dpi=300, bbox_inches='tight')
            plt.close(fig_subj)
            
            # Save classification report
            report = generate_classification_report(
                all_labels, all_predictions, self.class_names,
                f'{save_dir}/classification_report.txt'
            )
            
            # Save results as JSON
            import json
            results_json = {
                'overall_metrics': {k: float(v) if isinstance(v, np.ndarray) and v.size == 1 
                                   else v.tolist() if isinstance(v, np.ndarray) else v
                                   for k, v in results['overall_metrics'].items()
                                   if k != 'confusion_matrix'},
                'per_subject_summary': {
                    'mean_accuracy': float(results['per_subject_results']['accuracy'].mean()),
                    'std_accuracy': float(results['per_subject_results']['accuracy'].std()),
                    'mean_f1': float(results['per_subject_results']['macro_f1'].mean()),
                    'std_f1': float(results['per_subject_results']['macro_f1'].std())
                }
            }
            
            with open(f'{save_dir}/evaluation_results.json', 'w') as f:
                json.dump(results_json, f, indent=2)
        
        return results
```
