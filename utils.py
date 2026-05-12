import torch
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def format_time(seconds):
    """Format seconds into readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {save_path}")

def plot_attention_distribution(attention_weights, save_path):
    """Plot attention weight distribution"""
    confidence = np.abs(attention_weights[:, 0] - attention_weights[:, 1])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of confidence scores
    axes[0].hist(confidence, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.2, color='r', linestyle='--', label='Low confidence threshold')
    axes[0].axvline(x=0.5, color='g', linestyle='--', label='High confidence threshold')
    axes[0].set_xlabel('Attention Confidence |α₁ - α₂|')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Attention Confidence Distribution')
    axes[0].legend()
    
    # Scatter plot of α₁ vs α₂
    axes[1].scatter(attention_weights[:, 0], attention_weights[:, 1], 
                   alpha=0.3, s=10)
    axes[1].plot([0, 1], [0, 1], 'r--', label='α₁ = α₂')
    axes[1].set_xlabel('α₁ (Encoder 1 weight)')
    axes[1].set_ylabel('α₂ (Encoder 2 weight)')
    axes[1].set_title('Attention Weight Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Attention distribution saved to {save_path}")

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training and validation curves"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {save_path}")

def save_results_table(results, save_path):
    """Save results as formatted table"""
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        # Overall metrics
        f.write("Overall Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1: {results['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {results['weighted_f1']:.4f}\n\n")
        
        # Per-class metrics
        f.write("Per-Class Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}\n")
        f.write("-"*70 + "\n")
        for class_name, metrics in results['per_class'].items():
            f.write(f"{class_name:<20} {metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} "
                   f"{metrics['support']:<10}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Results table saved to {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"✓ Loaded checkpoint from epoch {epoch}")
    
    return checkpoint

def save_checkpoint(checkpoint_path, model, optimizer=None, epoch=0, **kwargs):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        **kwargs
    }
    
    if optimizer:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved checkpoint to {checkpoint_path}")

def print_model_summary(model, config):
    """Print model architecture summary"""
    trainable, total = count_parameters(model)
    
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable %: {100 * trainable / total:.2f}%")
    print("="*70 + "\n")