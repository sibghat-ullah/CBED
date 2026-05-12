import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import nlpaug.augmenter.word as naw

class ChatbotDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, class_to_idx):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        query = row['query']
        response = row['response']
        intent = row['intent']
        
        # DEBUG: Check if intent exists in mapping
        if intent not in self.class_to_idx:
            raise KeyError(f"Intent '{intent}' not found in class_to_idx mapping. "
                          f"Available intents: {list(self.class_to_idx.keys())}")
        
        # Tokenize query
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize response
        response_encoding = self.tokenizer(
            response,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': query_encoding['input_ids'].squeeze(0),
            'attention_mask': query_encoding['attention_mask'].squeeze(0),
            'response_ids': response_encoding['input_ids'].squeeze(0),
            'response_mask': response_encoding['attention_mask'].squeeze(0),
            'intent': self.class_to_idx[intent],
            'intent_name': intent
        }
    

def load_data(config):
    """Load and split dataset according to encoder/decoder partitioning"""
    # Load CSV (columns: query, response, intent)
    df = pd.read_csv(config.data_path)
    
    # CRITICAL FIX: Normalize intent names to title case
    df['intent'] = df['intent'].str.strip().str.title()
    
    # Get all unique intents from dataset
    all_intents = sorted(df['intent'].unique())
    
    print("\n" + "="*70)
    print("INTENT CLASSES FOUND IN DATASET:")
    print("="*70)
    class_counts = df['intent'].value_counts()
    for intent in all_intents:
        count = class_counts[intent]
        print(f"  - {intent}: {count} samples")
    
    # Update config.classes to include ALL intents from dataset
    config.classes = all_intents
    config.num_classes = len(config.classes)
    
    # Auto-configure class assignments based on frequency threshold
    frequent = class_counts[class_counts >= config.frequency_threshold].index.tolist()
    rare = class_counts[class_counts < config.frequency_threshold].index.tolist()
    
    config.frequent_classes = frequent
    config.rare_classes = rare
    
    # For decoder routing: find "Open Domain" class
    open_domain_candidates = [c for c in config.classes if 'open' in c.lower() and 'domain' in c.lower()]
    
    if open_domain_candidates:
        config.open_domain_classes = open_domain_candidates
        config.domain_classes = [c for c in config.classes if c not in config.open_domain_classes]
    else:
        # If no "Open Domain" class found, use the most frequent as open domain
        most_frequent = class_counts.idxmax()
        config.open_domain_classes = [most_frequent]
        config.domain_classes = [c for c in config.classes if c != most_frequent]
    
    print("\n" + "="*70)
    print("AUTO-CONFIGURED CLASS ASSIGNMENTS")
    print("="*70)
    print(f"Frequency threshold: {config.frequency_threshold} samples\n")
    
    print(f"Frequent classes (Encoder 1): {len(config.frequent_classes)}")
    for cls in config.frequent_classes:
        print(f"  - {cls}: {class_counts[cls]} samples")
    
    print(f"\nRare classes (Encoder 2): {len(config.rare_classes)}")
    for cls in config.rare_classes:
        print(f"  - {cls}: {class_counts[cls]} samples")
    
    print(f"\nOpen Domain classes (Phi-1.5): {config.open_domain_classes}")
    print(f"Domain-specific classes (SmolLM): {config.domain_classes}")
    print("="*70)
    
    # Create class to index mapping - THIS IS CRITICAL
    class_to_idx = {cls: idx for idx, cls in enumerate(config.classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Verify all intents in dataframe are in mapping
    missing_intents = set(df['intent'].unique()) - set(class_to_idx.keys())
    if missing_intents:
        print(f"\n❌ ERROR: Found intents in data not in mapping: {missing_intents}")
        raise ValueError(f"Missing intents in class_to_idx: {missing_intents}")
    
    # Split train/val stratified by class
    train_df, val_df = train_test_split(
        df, 
        test_size=config.val_ratio, 
        stratify=df['intent'],
        random_state=config.seed
    )
    
    # Create encoder-specific datasets
    # Handle empty lists gracefully
    if config.frequent_classes:
        train_enc1 = train_df[train_df['intent'].isin(config.frequent_classes)]
        val_enc1 = val_df[val_df['intent'].isin(config.frequent_classes)]
    else:
        train_enc1 = train_df.iloc[0:0]  # Empty dataframe
        val_enc1 = val_df.iloc[0:0]
    
    if config.rare_classes:
        train_enc2 = train_df[train_df['intent'].isin(config.rare_classes)]
        val_enc2 = val_df[val_df['intent'].isin(config.rare_classes)]
    else:
        train_enc2 = train_df.iloc[0:0]
        val_enc2 = val_df.iloc[0:0]
    
    # Create decoder-specific datasets
    if config.domain_classes:
        train_smollm = train_df[train_df['intent'].isin(config.domain_classes)]
        val_smollm = val_df[val_df['intent'].isin(config.domain_classes)]
    else:
        train_smollm = train_df.iloc[0:0]
        val_smollm = val_df.iloc[0:0]
    
    if config.open_domain_classes:
        train_phi = train_df[train_df['intent'].isin(config.open_domain_classes)]
        val_phi = val_df[val_df['intent'].isin(config.open_domain_classes)]
    else:
        train_phi = train_df.iloc[0:0]
        val_phi = val_df.iloc[0:0]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"\nEncoder 1 (Frequent): Train={len(train_enc1)}, Val={len(val_enc1)}")
    print(f"Encoder 2 (Rare): Train={len(train_enc2)}, Val={len(val_enc2)}")
    print(f"\nSmolLM (Domain): Train={len(train_smollm)}, Val={len(val_smollm)}")
    print(f"Phi-1.5 (Open): Train={len(train_phi)}, Val={len(val_phi)}")
    
    return {
        'tokenizer': tokenizer,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'train_enc1': train_enc1,
        'val_enc1': val_enc1,
        'train_enc2': train_enc2,
        'val_enc2': val_enc2,
        'train_smollm': train_smollm,
        'val_smollm': val_smollm,
        'train_phi': train_phi,
        'val_phi': val_phi,
        'train_full': train_df,
        'val_full': val_df
    }

def create_dataloaders(data_dict, config, stage='stage1'):
    """Create dataloaders for specific training stage"""
    tokenizer = data_dict['tokenizer']
    class_to_idx = data_dict['class_to_idx']
    
    if stage == 'stage1':
        # Stage 1: Encoder training
        train_enc1_dataset = ChatbotDataset(
            data_dict['train_enc1'], tokenizer, config.max_seq_length, class_to_idx
        )
        val_enc1_dataset = ChatbotDataset(
            data_dict['val_enc1'], tokenizer, config.max_seq_length, class_to_idx
        )
        
        train_enc2_dataset = ChatbotDataset(
            data_dict['train_enc2'], tokenizer, config.max_seq_length, class_to_idx
        )
        val_enc2_dataset = ChatbotDataset(
            data_dict['val_enc2'], tokenizer, config.max_seq_length, class_to_idx
        )
        
        train_loader_enc1 = DataLoader(
            train_enc1_dataset, 
            batch_size=config.stage1_batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader_enc1 = DataLoader(
            val_enc1_dataset,
            batch_size=config.stage1_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        train_loader_enc2 = DataLoader(
            train_enc2_dataset,
            batch_size=config.stage1_batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader_enc2 = DataLoader(
            val_enc2_dataset,
            batch_size=config.stage1_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        return {
            'train_enc1': train_loader_enc1,
            'val_enc1': val_loader_enc1,
            'train_enc2': train_loader_enc2,
            'val_enc2': val_loader_enc2
        }
    
    elif stage == 'stage2':
        # Stage 2: Fusion training (use full dataset)
        train_dataset = ChatbotDataset(
            data_dict['train_full'], tokenizer, config.max_seq_length, class_to_idx
        )
        val_dataset = ChatbotDataset(
            data_dict['val_full'], tokenizer, config.max_seq_length, class_to_idx
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.stage2_batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.stage2_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        return {'train': train_loader, 'val': val_loader}
    
    elif stage == 'stage3':
        # Stage 3: Decoder training
        train_smollm_dataset = ChatbotDataset(
            data_dict['train_smollm'], tokenizer, config.max_seq_length, class_to_idx
        )
        val_smollm_dataset = ChatbotDataset(
            data_dict['val_smollm'], tokenizer, config.max_seq_length, class_to_idx
        )
        
        train_phi_dataset = ChatbotDataset(
            data_dict['train_phi'], tokenizer, config.max_seq_length, class_to_idx
        )
        val_phi_dataset = ChatbotDataset(
            data_dict['val_phi'], tokenizer, config.max_seq_length, class_to_idx
        )
        
        train_loader_smollm = DataLoader(
            train_smollm_dataset,
            batch_size=config.stage3_batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader_smollm = DataLoader(
            val_smollm_dataset,
            batch_size=config.stage3_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        train_loader_phi = DataLoader(
            train_phi_dataset,
            batch_size=config.stage3_batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader_phi = DataLoader(
            val_phi_dataset,
            batch_size=config.stage3_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        return {
            'train_smollm': train_loader_smollm,
            'val_smollm': val_loader_smollm,
            'train_phi': train_loader_phi,
            'val_phi': val_loader_phi
        }

def augment_rare_classes(df, rare_classes, target_samples=500):
    """
    Augment rare classes using back-translation and synonym replacement
    """
    try:
        # Synonym replacement augmenter
        aug = naw.SynonymAug(aug_src='wordnet')
        
        augmented_dfs = [df]
        
        for cls in rare_classes:
            class_df = df[df['intent'] == cls]
            current_count = len(class_df)
            
            if current_count < target_samples:
                needed = target_samples - current_count
                print(f"  Augmenting {cls}: {current_count} → {target_samples} samples")
                
                # Augment by repeating and modifying
                aug_samples = []
                while len(aug_samples) < needed:
                    for _, row in class_df.iterrows():
                        if len(aug_samples) >= needed:
                            break
                        try:
                            aug_query = aug.augment(row['query'])
                            aug_samples.append({
                                'query': aug_query[0] if isinstance(aug_query, list) else aug_query,
                                'response': row['response'],
                                'intent': row['intent']
                            })
                        except:
                            # If augmentation fails, just duplicate
                            aug_samples.append({
                                'query': row['query'],
                                'response': row['response'],
                                'intent': row['intent']
                            })
                
                augmented_dfs.append(pd.DataFrame(aug_samples))
        
        return pd.concat(augmented_dfs, ignore_index=True)
    
    except ImportError:
        print("  ⚠️ nlpaug not installed. Install with: pip install nlpaug")
        print("  Skipping augmentation...")
        return df