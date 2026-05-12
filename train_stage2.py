import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
import os
from models import TransformerEncoder, ChannelAttentionFusion, IntentClassifier
from data_loader import load_data, create_dataloaders
from config import Config
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
def create_balanced_dataloader(df, tokenizer, max_length, class_to_idx, batch_size, num_workers):
    """Create a dataloader with balanced class sampling"""
    from torch.utils.data import WeightedRandomSampler
    
    # Calculate class weights for sampling
    class_counts = df['intent'].value_counts()
    total_samples = len(df)
    
    # Weight inversely proportional to class frequency
    class_weights = {intent: total_samples / count for intent, count in class_counts.items()}
    
    # Assign weight to each sample
    sample_weights = df['intent'].map(class_weights).values
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True
    )
    
    # Create dataset
    from data_loader import ChatbotDataset
    dataset = ChatbotDataset(df, tokenizer, max_length, class_to_idx)
    
    # Create dataloader with sampler
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers
    )
    
    return dataloader

def train_fusion(encoder1, encoder2, fusion, classifier, train_df, val_df, 
                tokenizer, class_to_idx, config):
    """Train fusion module with frozen encoders"""
    
    # Move to device
    encoder1 = encoder1.to(config.device)
    encoder2 = encoder2.to(config.device)
    fusion = fusion.to(config.device)
    classifier = classifier.to(config.device)
    
    # Freeze encoders
    for param in encoder1.parameters():
        param.requires_grad = False
    for param in encoder2.parameters():
        param.requires_grad = False
    
    encoder1.eval()
    encoder2.eval()
    
    # Calculate class weights for Focal Loss
    class_counts = train_df['intent'].value_counts()
    total_samples = len(train_df)
    class_weights = torch.tensor([
        total_samples / class_counts[config.classes[i]] 
        for i in range(config.num_classes)
    ], dtype=torch.float32).to(config.device)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * config.num_classes
    
    print(f"\nClass Weights for Focal Loss:")
    for i, cls in enumerate(config.classes):
        print(f"  {cls}: {class_weights[i]:.4f}")
    
    # Create balanced dataloaders
    train_loader = create_balanced_dataloader(
        train_df, tokenizer, config.max_seq_length, class_to_idx,
        config.stage2_batch_size, config.num_workers
    )
    
    # Regular validation loader (no balancing needed)
    from data_loader import ChatbotDataset
    val_dataset = ChatbotDataset(val_df, tokenizer, config.max_seq_length, class_to_idx)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.stage2_batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Optimizer (only fusion + classifier parameters)
    optimizer = AdamW(
        list(fusion.parameters()) + list(classifier.parameters()),
        lr=config.stage2_lr,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon
    )
    
    # Loss function: Focal Loss with reduced entropy regularization
    focal_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    entropy_weight = 0.01  # REDUCED from 0.1 to allow soft blending
    
    best_val_f1 = 0.0
    patience = 3
    patience_counter = 0
    
    print(f"\n{'='*50}")
    print(f"Training Fusion Module with Balanced Sampling")
    print(f"{'='*50}")
    print(f"Trainable parameters: {sum(p.numel() for p in fusion.parameters() if p.requires_grad):,}")
    print(f"Entropy regularization: {entropy_weight}")
    
    for epoch in range(config.stage2_epochs):
        # Training
        fusion.train()
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Track per-class accuracy
        class_correct = {cls: 0 for cls in config.classes}
        class_total = {cls: 0 for cls in config.classes}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.stage2_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['intent'].to(config.device)
            
            # Forward pass through frozen encoders
            with torch.no_grad():
                h1 = encoder1(input_ids, attention_mask)
                h2 = encoder2(input_ids, attention_mask)
            
            # Fusion with attention
            h_fused, attention_weights = fusion(h1, h2)
            
            # Classification
            logits = classifier(h_fused)
            
            # Focal Loss for classification
            focal_loss = focal_loss_fn(logits, labels)
            
            # Entropy regularization (REDUCED to allow soft attention)
            entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
            
            # Total loss
            loss = focal_loss + entropy_weight * entropy
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(fusion.parameters()) + list(classifier.parameters()),
                config.max_grad_norm
            )
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Per-class tracking
            for i in range(len(labels)):
                label_name = config.classes[labels[i].item()]
                class_total[label_name] += 1
                if predicted[i] == labels[i]:
                    class_correct[label_name] += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        encoder1.eval()
        encoder2.eval()
        fusion.eval()
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        attention_confidence = []
        
        # Per-class validation metrics
        val_class_correct = {cls: 0 for cls in config.classes}
        val_class_total = {cls: 0 for cls in config.classes}
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['intent'].to(config.device)
                
                h1 = encoder1(input_ids, attention_mask)
                h2 = encoder2(input_ids, attention_mask)
                h_fused, attention_weights = fusion(h1, h2)
                logits = classifier(h_fused)
                
                focal_loss = focal_loss_fn(logits, labels)
                entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
                loss = focal_loss + entropy_weight * entropy
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Per-class tracking
                for i in range(len(labels)):
                    label_name = config.classes[labels[i].item()]
                    val_class_total[label_name] += 1
                    if predicted[i] == labels[i]:
                        val_class_correct[label_name] += 1
                
                # Track attention confidence
                confidence = torch.abs(attention_weights[:, 0] - attention_weights[:, 1])
                attention_confidence.extend(confidence.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        avg_confidence = np.mean(attention_confidence)
        uncertain_pct = 100 * sum(1 for c in attention_confidence if c < 0.2) / len(attention_confidence)
        
        # Calculate macro F1
        from sklearn.metrics import f1_score
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
        print(f"  Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Macro F1={val_macro_f1:.4f}")
        print(f"  Attention Confidence: {avg_confidence:.3f}, Uncertain (<0.2): {uncertain_pct:.1f}%")
        
        # Show per-class accuracy for rare classes
        print(f"\n  Rare Class Performance:")
        for cls in config.rare_classes:
            if val_class_total[cls] > 0:
                acc = 100 * val_class_correct[cls] / val_class_total[cls]
                print(f"    {cls}: {val_class_correct[cls]}/{val_class_total[cls]} ({acc:.1f}%)")
        
        # Save best model based on macro F1
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            os.makedirs(f"{config.checkpoint_dir}/stage2_fusion", exist_ok=True)
            torch.save({
                'fusion': fusion.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'val_f1': val_macro_f1
            }, f"{config.checkpoint_dir}/stage2_fusion/fusion_classifier.pt")
            print(f"  ✓ Saved best model (Macro F1: {val_macro_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered!")
                break
    
    return fusion, classifier

def main():
    config = Config()
    torch.manual_seed(config.seed)
    
    print("="*70)
    print("STAGE 2: FUSION TRAINING (WITH BALANCED SAMPLING)")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data_dict = load_data(config)
    
    # Load trained encoders from Stage 1
    print("\nLoading trained encoders...")
    encoder1 = TransformerEncoder(config)
    encoder2 = TransformerEncoder(config)
    
    checkpoint1 = torch.load(f"{config.checkpoint_dir}/stage1_encoders/encoder1_frequent.pt")
    checkpoint2 = torch.load(f"{config.checkpoint_dir}/stage1_encoders/encoder2_rare.pt")
    
    encoder1.load_state_dict(checkpoint1['encoder'])
    encoder2.load_state_dict(checkpoint2['encoder'])
    print("✓ Encoders loaded")
    
    # Initialize fusion and classifier
    fusion = ChannelAttentionFusion(config)
    classifier = IntentClassifier(config)
    
    print(f"\nFusion module parameters: {sum(p.numel() for p in fusion.parameters()):,}")
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # Train fusion with DATAFRAMES (not dataloaders)
    fusion, classifier = train_fusion(
        encoder1, encoder2, fusion, classifier,
        data_dict['train_full'],  # Pass DataFrame
        data_dict['val_full'],    # Pass DataFrame
        data_dict['tokenizer'],
        data_dict['class_to_idx'],
        config
    )
    
    print("\n" + "="*70)
    print("STAGE 2 COMPLETE!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}/stage2_fusion/")
    print("="*70)

if __name__ == "__main__":
    main()