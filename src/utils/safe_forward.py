
"""
Safe Forward Pass Utilities for MAET Model
Handles flexible model forward passes with different modality combinations
"""

import torch
import torch.nn.functional as F


def safe_model_forward(model, eeg=None, eye=None, alpha_=None):
    """
    Safe forward pass through MAET model with flexible input handling
    
    This function provides a standardized interface for calling the MAET model
    with different combinations of modalities and parameters, handling cases
    where some inputs might be None or unavailable.
    
    Args:
        model (torch.nn.Module): MAET model instance
        eeg (torch.Tensor, optional): EEG features [batch_size, eeg_dim]
        eye (torch.Tensor, optional): Eye tracking features [batch_size, eye_dim]
        alpha_ (float, optional): Gradient reversal strength for domain adaptation
        
    Returns:
        tuple: (predictions, additional_outputs)
            - predictions: Main emotion classification logits
            - additional_outputs: Domain predictions or other auxiliary outputs (may be None)
    """
    # Build keyword arguments based on available inputs
    kwargs = {}
    
    if eeg is not None:
        kwargs['eeg'] = eeg
    
    if eye is not None:
        kwargs['eye'] = eye
        
    if alpha_ is not None:
        kwargs['alpha_'] = alpha_
    
    # Ensure at least one modality is provided
    if not kwargs or ('eeg' not in kwargs and 'eye' not in kwargs):
        raise ValueError("At least one modality (eeg or eye) must be provided")
    
    # Forward pass through model
    try:
        output = model(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Error during model forward pass: {str(e)}")
    
    # Handle different output formats
    if isinstance(output, tuple):
        # Model returned multiple outputs (e.g., predictions + domain predictions)
        predictions = output[0]
        additional_outputs = output[1:] if len(output) > 1 else None
        return predictions, additional_outputs
    else:
        # Model returned single output (just predictions)
        return output, None


def batch_safe_forward(model, batch, device=None, alpha_=None):
    """
    Safe forward pass with automatic batch processing and device handling
    
    Args:
        model (torch.nn.Module): MAET model
        batch (dict): Batch dictionary containing 'eeg', 'eye', 'label', etc.
        device (torch.device, optional): Target device for computation
        alpha_ (float, optional): Gradient reversal strength
        
    Returns:
        tuple: (predictions, additional_outputs, processed_batch)
    """
    # Move batch to device if specified
    if device is not None:
        processed_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                processed_batch[key] = value.to(device)
            else:
                processed_batch[key] = value
    else:
        processed_batch = batch
    
    # Extract modality data
    eeg = processed_batch.get('eeg', None)
    eye = processed_batch.get('eye', None)
    
    # Forward pass
    predictions, additional = safe_model_forward(
        model, eeg=eeg, eye=eye, alpha_=alpha_
    )
    
    return predictions, additional, processed_batch


def validate_model_inputs(eeg=None, eye=None, eeg_dim=310, eye_dim=33):
    """
    Validate input tensors for MAET model
    
    Args:
        eeg (torch.Tensor, optional): EEG features
        eye (torch.Tensor, optional): Eye tracking features  
        eeg_dim (int): Expected EEG feature dimension
        eye_dim (int): Expected eye tracking feature dimension
        
    Raises:
        ValueError: If input validation fails
    """
    if eeg is None and eye is None:
        raise ValueError("At least one modality must be provided")
    
    if eeg is not None:
        if not torch.is_tensor(eeg):
            raise ValueError("EEG input must be a torch.Tensor")
        if eeg.dim() != 2:
            raise ValueError(f"EEG input must be 2D (batch_size, features), got {eeg.dim()}D")
        if eeg.size(1) != eeg_dim:
            raise ValueError(f"EEG feature dimension must be {eeg_dim}, got {eeg.size(1)}")
    
    if eye is not None:
        if not torch.is_tensor(eye):
            raise ValueError("Eye input must be a torch.Tensor")
        if eye.dim() != 2:
            raise ValueError(f"Eye input must be 2D (batch_size, features), got {eye.dim()}D")
        if eye.size(1) != eye_dim:
            raise ValueError(f"Eye feature dimension must be {eye_dim}, got {eye.size(1)}")
    
    # Check batch size consistency
    if eeg is not None and eye is not None:
        if eeg.size(0) != eye.size(0):
            raise ValueError(f"Batch sizes must match: EEG {eeg.size(0)}, Eye {eye.size(0)}")


def ensemble_forward(models, eeg=None, eye=None, method='average', weights=None):
    """
    Forward pass through multiple models for ensemble prediction
    
    Args:
        models (list): List of MAET model instances
        eeg (torch.Tensor, optional): EEG features
        eye (torch.Tensor, optional): Eye tracking features
        method (str): Ensemble method - 'average', 'weighted', or 'voting'
        weights (list, optional): Model weights for weighted averaging
        
    Returns:
        torch.Tensor: Ensemble predictions
    """
    if not models:
        raise ValueError("At least one model must be provided")
    
    predictions = []
    
    # Collect predictions from all models
    for model in models:
        model.eval()
        with torch.no_grad():
            pred, _ = safe_model_forward(model, eeg=eeg, eye=eye)
            predictions.append(pred)
    
    predictions = torch.stack(predictions, dim=0)  # [n_models, batch_size, n_classes]
    
    # Apply ensemble method
    if method == 'average':
        ensemble_pred = torch.mean(predictions, dim=0)
    elif method == 'weighted':
        if weights is None:
            raise ValueError("Weights must be provided for weighted ensemble")
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        weights = torch.tensor(weights, dtype=predictions.dtype, device=predictions.device)
        weights = weights.view(-1, 1, 1)  # [n_models, 1, 1]
        weights = F.softmax(weights, dim=0)  # Normalize weights
        
        ensemble_pred = torch.sum(predictions * weights, dim=0)
    elif method == 'voting':
        # Hard voting - take mode of predicted classes
        predicted_classes = torch.argmax(predictions, dim=-1)  # [n_models, batch_size]
        ensemble_pred = torch.mode(predicted_classes, dim=0).values
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_pred


def gradual_modality_dropout(model, eeg, eye, dropout_schedule, epoch):
    """
    Apply gradual modality dropout during training for robustness
    
    Args:
        model (torch.nn.Module): MAET model
        eeg (torch.Tensor): EEG features
        eye (torch.Tensor): Eye tracking features
        dropout_schedule (dict): Schedule for modality dropout probabilities
        epoch (int): Current training epoch
        
    Returns:
        tuple: (predictions, additional_outputs)
    """
    # Get dropout probabilities for current epoch
    eeg_dropout_prob = dropout_schedule.get('eeg', {}).get(epoch, 0.0)
    eye_dropout_prob = dropout_schedule.get('eye', {}).get(epoch, 0.0)
    
    # Apply dropout to modalities
    use_eeg = torch.rand(1).item() > eeg_dropout_prob
    use_eye = torch.rand(1).item() > eye_dropout_prob
    
    # Ensure at least one modality is kept
    if not use_eeg and not use_eye:
        if torch.rand(1).item() > 0.5:
            use_eeg = True
        else:
            use_eye = True
    
    # Forward pass with selected modalities
    input_eeg = eeg if use_eeg else None
    input_eye = eye if use_eye else None
    
    return safe_model_forward(model, eeg=input_eeg, eye=input_eye)


def adaptive_forward(model, eeg, eye, confidence_threshold=0.8):
    """
    Adaptive forward pass that uses confidence to determine modality usage
    
    Args:
        model (torch.nn.Module): MAET model
        eeg (torch.Tensor): EEG features
        eye (torch.Tensor): Eye tracking features  
        confidence_threshold (float): Threshold for high-confidence predictions
        
    Returns:
        tuple: (predictions, additional_outputs, modality_used)
    """
    model.eval()
    
    with torch.no_grad():
        # Try multimodal first
        pred_multi, additional = safe_model_forward(model, eeg=eeg, eye=eye)
        confidence_multi = torch.max(F.softmax(pred_multi, dim=1), dim=1)[0]
        
        # If confidence is high, use multimodal prediction
        if torch.mean(confidence_multi) > confidence_threshold:
            return pred_multi, additional, 'multimodal'
        
        # Try EEG-only
        pred_eeg, _ = safe_model_forward(model, eeg=eeg, eye=None)
        confidence_eeg = torch.max(F.softmax(pred_eeg, dim=1), dim=1)[0]
        
        # Try eye-only
        pred_eye, _ = safe_model_forward(model, eeg=None, eye=eye)
        confidence_eye = torch.max(F.softmax(pred_eye, dim=1), dim=1)[0]
        
        # Choose modality with highest confidence
        avg_confidence = {
            'multimodal': torch.mean(confidence_multi),
            'eeg': torch.mean(confidence_eeg), 
            'eye': torch.mean(confidence_eye)
        }
        
        best_modality = max(avg_confidence, key=avg_confidence.get)
        
        if best_modality == 'eeg':
            return pred_eeg, None, 'eeg'
        elif best_modality == 'eye':
            return pred_eye, None, 'eye'
        else:
            return pred_multi, additional, 'multimodal'


class ForwardPassLogger:
    """
    Logger for tracking model forward pass statistics and debugging
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset logging statistics"""
        self.total_calls = 0
        self.modality_usage = {'eeg': 0, 'eye': 0, 'multimodal': 0}
        self.error_count = 0
        self.avg_prediction_time = 0.0
    
    def log_forward_pass(self, eeg, eye, prediction_time=None):
        """
        Log a forward pass
        
        Args:
            eeg (torch.Tensor): EEG input (or None)
            eye (torch.Tensor): Eye input (or None)  
            prediction_time (float): Time taken for prediction
        """
        self.total_calls += 1
        
        # Track modality usage
        if eeg is not None and eye is not None:
            self.modality_usage['multimodal'] += 1
        elif eeg is not None:
            self.modality_usage['eeg'] += 1
        elif eye is not None:
            self.modality_usage['eye'] += 1
        
        # Track timing if provided
        if prediction_time is not None:
            self.avg_prediction_time = (
                (self.avg_prediction_time * (self.total_calls - 1) + prediction_time) / 
                self.total_calls
            )
    
    def log_error(self):
        """Log an error during forward pass"""
        self.error_count += 1
    
    def get_statistics(self):
        """Get logging statistics"""
        if self.total_calls == 0:
            return {}
        
        return {
            'total_calls': self.total_calls,
            'error_rate': self.error_count / self.total_calls,
            'modality_distribution': {
                k: v / self.total_calls for k, v in self.modality_usage.items()
            },
            'avg_prediction_time': self.avg_prediction_time
        }


# Global logger instance
forward_logger = ForwardPassLogger()
```
