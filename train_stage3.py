import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
from transformers import AutoTokenizer
from models import TransformerEncoder, ChannelAttentionFusion, IntentClassifier, create_lora_decoder
from data_loader import load_data, create_dataloaders
from config import Config

def train_decoder(decoder, train_loader, config, decoder_name, tokenizer):
    """Train decoder with LoRA"""
    decoder = decoder.to(config.device)
    
    # Optimizer (only LoRA parameters)
    optimizer = AdamW(
        [p for p in decoder.parameters() if p.requires_grad],
        lr=config.stage3_lr,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon
    )
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*50}")
    print(f"Training {decoder_name}")
    print(f"{'='*50}")
    
    for epoch in range(config.stage3_epochs):
        # Training
        decoder.train()
        train_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.stage3_epochs}")
        
        for step, batch in enumerate(pbar):
            response_ids = batch['response_ids'].to(config.device)
            response_mask = batch['response_mask'].to(config.device)
            
            # Shift for causal language modeling
            input_ids = response_ids[:, :-1]
            labels = response_ids[:, 1:]
            attention_mask = response_mask[:, :-1]
            
            # Forward pass
            outputs = decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Gradient accumulation
            loss = loss / config.stage3_grad_accum
            loss.backward()
            
            # Update weights after accumulation steps
            if (step + 1) % config.stage3_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in decoder.parameters() if p.requires_grad],
                    config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config.stage3_grad_accum
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item() * config.stage3_grad_accum:.4f}'})
        
        train_loss /= num_batches
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")
        
        # Save checkpoint
        os.makedirs(f"{config.checkpoint_dir}/stage3_decoders", exist_ok=True)
        decoder.save_pretrained(f"{config.checkpoint_dir}/stage3_decoders/{decoder_name}")
        print(f"✓ Saved {decoder_name}")
    
    return decoder

def main():
    config = Config()
    torch.manual_seed(config.seed)
    
    print("="*70)
    print("STAGE 3: DECODER TRAINING")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data_dict = load_data(config)
    dataloaders = create_dataloaders(data_dict, config, stage='stage3')
    
    # Load tokenizers for decoders
    smollm_tokenizer = AutoTokenizer.from_pretrained(config.smollm_model)
    phi_tokenizer = AutoTokenizer.from_pretrained(config.phi_model)
    
    if smollm_tokenizer.pad_token is None:
        smollm_tokenizer.pad_token = smollm_tokenizer.eos_token
    if phi_tokenizer.pad_token is None:
        phi_tokenizer.pad_token = phi_tokenizer.eos_token
    
    # Create decoders with LoRA
    print("\nInitializing SmolLM with LoRA...")
    smollm_decoder = create_lora_decoder(config, config.smollm_model, config.device)
    
    print("\nInitializing Phi-1.5 with LoRA...")
    phi_decoder = create_lora_decoder(config, config.phi_model, config.device)
    
    # Train SmolLM (Domain-specific)
    print("\n" + "="*50)
    print("Training SmolLM for Domain-Specific Queries")
    print("="*50)
    smollm_decoder = train_decoder(
        smollm_decoder,
        dataloaders['train_smollm'],
        config,
        decoder_name='smollm_lora',
        tokenizer=smollm_tokenizer
    )
    
    # Train Phi-1.5 (Open Domain)
    print("\n" + "="*50)
    print("Training Phi-1.5 for Open Domain Queries")
    print("="*50)
    phi_decoder = train_decoder(
        phi_decoder,
        dataloaders['train_phi'],
        config,
        decoder_name='phi_lora',
        tokenizer=phi_tokenizer
    )
    
    print("\n" + "="*70)
    print("STAGE 3 COMPLETE!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}/stage3_decoders/")
    print("="*70)

if __name__ == "__main__":
    main()