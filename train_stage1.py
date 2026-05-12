import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
from models import TransformerEncoder
from data_loader import load_data, create_dataloaders
from config import Config

def train_encoder(encoder, train_loader, val_loader, config, encoder_name):
    """Train a single encoder"""
    encoder = encoder.to(config.device)
    
    # Classification head for this encoder
    classifier = nn.Linear(config.hidden_dim, config.num_classes).to(config.device)
    
    # Optimizer
    optimizer = AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=config.stage1_lr,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*50}")
    print(f"Training {encoder_name}")
    print(f"{'='*50}")
    
    for epoch in range(config.stage1_epochs):
        # Training
        encoder.train()
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.stage1_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['intent'].to(config.device)
            
            # Forward pass
            encoder_output = encoder(input_ids, attention_mask)
            logits = classifier(encoder_output)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()),
                config.max_grad_norm
            )
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        encoder.eval()
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['intent'].to(config.device)
                
                encoder_output = encoder(input_ids, attention_mask)
                logits = classifier(encoder_output)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(f"{config.checkpoint_dir}/stage1_encoders", exist_ok=True)
            torch.save({
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, f"{config.checkpoint_dir}/stage1_encoders/{encoder_name}.pt")
            print(f"✓ Saved best {encoder_name}")
    
    return encoder

def main():
    config = Config()
    torch.manual_seed(config.seed)
    
    print("="*70)
    print("STAGE 1: ENCODER TRAINING")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data_dict = load_data(config)
    dataloaders = create_dataloaders(data_dict, config, stage='stage1')
    
    # Initialize encoders
    encoder1 = TransformerEncoder(config)
    encoder2 = TransformerEncoder(config)
    
    print(f"\nEncoder architecture:")
    print(f"- Hidden dim: {config.hidden_dim}")
    print(f"- Layers: {config.num_layers}")
    print(f"- Heads: {config.num_heads}")
    print(f"- Parameters: ~{sum(p.numel() for p in encoder1.parameters()) / 1e6:.1f}M per encoder")
    
    # Train Encoder 1 (Frequent classes)
    encoder1 = train_encoder(
        encoder1,
        dataloaders['train_enc1'],
        dataloaders['val_enc1'],
        config,
        encoder_name='encoder1_frequent'
    )
    
    # Train Encoder 2 (Rare classes)
    encoder2 = train_encoder(
        encoder2,
        dataloaders['train_enc2'],
        dataloaders['val_enc2'],
        config,
        encoder_name='encoder2_rare'
    )
    
    print("\n" + "="*70)
    print("STAGE 1 COMPLETE!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}/stage1_encoders/")
    print("="*70)

if __name__ == "__main__":
    main()