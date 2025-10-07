"""
Gradient Reversal Layer for Domain Adversarial Training
Implements the gradient reversal technique from the DANN paper
"""

import torch


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer implementation.
    
    This layer acts as an identity function during forward pass but reverses
    (and scales) the gradient during backward pass. Used in domain adversarial
    training to learn domain-invariant features.
    
    Based on "Domain-Adversarial Training of Neural Networks" (Ganin et al.)
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass - acts as identity function
        
        Args:
            ctx: PyTorch context for backward pass
            x (torch.Tensor): Input tensor
            alpha (float): Gradient reversal strength
            
        Returns:
            torch.Tensor: Output tensor (same as input)
        """
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod  
    def backward(ctx, grad_output):
        """
        Backward pass - reverses and scales gradient
        
        Args:
            ctx: PyTorch context from forward pass
            grad_output (torch.Tensor): Gradient from next layer
            
        Returns:
            tuple: (reversed_gradient, None)
        """
        return -ctx.alpha * grad_output, None


def gradient_reversal(x, alpha=1.0):
    """
    Convenience function to apply gradient reversal
    
    Args:
        x (torch.Tensor): Input tensor
        alpha (float): Gradient reversal strength (default: 1.0)
        
    Returns:
        torch.Tensor: Tensor with gradient reversal applied
    """
    return GradientReversalLayer.apply(x, alpha)


class AdversarialLoss(torch.nn.Module):
    """
    Adversarial loss for domain adaptation
    
    Combines emotion classification loss with domain classification loss
    using gradient reversal for domain-invariant feature learning.
    """
    
    def __init__(self, emotion_weight=1.0, domain_weight=0.1):
        """
        Initialize adversarial loss
        
        Args:
            emotion_weight (float): Weight for emotion classification loss
            domain_weight (float): Weight for domain classification loss
        """
        super().__init__()
        self.emotion_weight = emotion_weight
        self.domain_weight = domain_weight
        self.emotion_criterion = torch.nn.CrossEntropyLoss()
        self.domain_criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, emotion_pred, emotion_true, domain_pred=None, domain_true=None):
        """
        Compute combined adversarial loss
        
        Args:
            emotion_pred (torch.Tensor): Emotion predictions
            emotion_true (torch.Tensor): True emotion labels
            domain_pred (torch.Tensor): Domain predictions (optional)
            domain_true (torch.Tensor): True domain labels (optional)
            
        Returns:
            torch.Tensor: Combined loss
        """
        emotion_loss = self.emotion_criterion(emotion_pred, emotion_true)
        
        if domain_pred is not None and domain_true is not None:
            domain_loss = self.domain_criterion(domain_pred, domain_true)
            total_loss = self.emotion_weight * emotion_loss + self.domain_weight * domain_loss
            return total_loss, emotion_loss, domain_loss
        else:
            return emotion_loss


class DomainClassifier(torch.nn.Module):
    """
    Domain classifier for adversarial training
    
    Takes features and predicts which domain/subject they come from.
    Used with gradient reversal to learn domain-invariant features.
    """
    
    def __init__(self, input_dim, num_domains, hidden_dim=None):
        """
        Initialize domain classifier
        
        Args:
            input_dim (int): Input feature dimension
            num_domains (int): Number of source domains
            hidden_dim (int): Hidden layer dimension (default: same as input_dim)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim // 2, num_domains)
        )
    
    def forward(self, x, alpha=1.0):
        """
        Forward pass with gradient reversal
        
        Args:
            x (torch.Tensor): Input features
            alpha (float): Gradient reversal strength
            
        Returns:
            torch.Tensor: Domain predictions
        """
        reversed_features = gradient_reversal(x, alpha)
        return self.classifier(reversed_features)


class ScheduledGradientReversal:
    """
    Scheduler for gradient reversal strength (alpha parameter)
    
    Implements common scheduling strategies for domain adaptation training
    """
    
    def __init__(self, strategy='linear', max_alpha=1.0, warmup_epochs=0):
        """
        Initialize gradient reversal scheduler
        
        Args:
            strategy (str): Scheduling strategy - 'linear', 'exponential', or 'constant'
            max_alpha (float): Maximum alpha value
            warmup_epochs (int): Number of warmup epochs before scheduling begins
        """
        self.strategy = strategy
        self.max_alpha = max_alpha
        self.warmup_epochs = warmup_epochs
    
    def get_alpha(self, epoch, total_epochs):
        """
        Get alpha value for current epoch
        
        Args:
            epoch (int): Current epoch
            total_epochs (int): Total training epochs
            
        Returns:
            float: Alpha value for gradient reversal
        """
        if epoch < self.warmup_epochs:
            return 0.0
        
        progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)
        
        if self.strategy == 'linear':
            return self.max_alpha * progress
        elif self.strategy == 'exponential':
            return self.max_alpha * (1 - torch.exp(torch.tensor(-progress * 5))).item()
        elif self.strategy == 'constant':
            return self.max_alpha
        else:
            raise ValueError(f"Unknown scheduling strategy: {self.strategy}")


def create_domain_mapping(subject_ids):
    """
    Create mapping from subject IDs to domain indices
    
    Args:
        subject_ids (list or np.ndarray): List of subject identifiers
        
    Returns:
        dict: Mapping from subject_id to domain_id
    """
    unique_subjects = sorted(list(set(subject_ids)))
    return {subject_id: domain_id for domain_id, subject_id in enumerate(unique_subjects)}


def apply_domain_adaptation_loss(emotion_logits, emotion_labels, domain_logits, 
                                domain_labels, emotion_weight=1.0, domain_weight=0.1):
    """
    Apply domain adaptation loss combining emotion and domain objectives
    
    Args:
        emotion_logits (torch.Tensor): Emotion classification logits
        emotion_labels (torch.Tensor): True emotion labels
        domain_logits (torch.Tensor): Domain classification logits
        domain_labels (torch.Tensor): True domain labels
        emotion_weight (float): Weight for emotion loss
        domain_weight (float): Weight for domain loss
        
    Returns:
        tuple: (total_loss, emotion_loss, domain_loss)
    """
    emotion_criterion = torch.nn.CrossEntropyLoss()
    domain_criterion = torch.nn.CrossEntropyLoss()
    
    emotion_loss = emotion_criterion(emotion_logits, emotion_labels)
    domain_loss = domain_criterion(domain_logits, domain_labels)
    
    total_loss = emotion_weight * emotion_loss + domain_weight * domain_loss
    
    return total_loss, emotion_loss, domain_loss


class MultiSourceDomainAdaptation:
    """
    Multi-source domain adaptation utilities
    
    Handles adaptation when training on multiple source domains
    and testing on a target domain.
    """
    
    def __init__(self, source_domains, lambda_domain=0.1):
        """
        Initialize multi-source domain adaptation
        
        Args:
            source_domains (list): List of source domain identifiers
            lambda_domain (float): Domain adaptation loss weight
        """
        self.source_domains = source_domains
        self.lambda_domain = lambda_domain
        self.num_domains = len(source_domains)
    
    def compute_domain_weights(self, domain_labels):
        """
        Compute weights for different domains based on sample distribution
        
        Args:
            domain_labels (torch.Tensor): Domain labels for current batch
            
        Returns:
            torch.Tensor: Domain weights
        """
        domain_counts = torch.bincount(domain_labels, minlength=self.num_domains)
        total_samples = domain_labels.size(0)
        
        # Inverse frequency weighting
        weights = total_samples / (self.num_domains * domain_counts.float())
        return weights
    
    def compute_adapted_loss(self, emotion_logits, emotion_labels, 
                           domain_logits, domain_labels):
        """
        Compute domain-adapted loss with source domain weighting
        
        Args:
            emotion_logits (torch.Tensor): Emotion predictions
            emotion_labels (torch.Tensor): True emotion labels
            domain_logits (torch.Tensor): Domain predictions  
            domain_labels (torch.Tensor): True domain labels
            
        Returns:
            tuple: (total_loss, emotion_loss, domain_loss)
        """
        # Standard emotion classification loss
        emotion_loss = torch.nn.functional.cross_entropy(emotion_logits, emotion_labels)
        
        # Weighted domain classification loss
        domain_weights = self.compute_domain_weights(domain_labels)
        domain_loss = torch.nn.functional.cross_entropy(
            domain_logits, domain_labels, weight=domain_weights
        )
        
        # Combined loss
        total_loss = emotion_loss + self.lambda_domain * domain_loss
        
        return total_loss, emotion_loss, domain_loss
```
