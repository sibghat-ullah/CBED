import torch

class Config:
    # Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4
    
    # Data paths
    data_path = "data/chatbot_dataset.csv"  # Update this
    checkpoint_dir = "checkpoints"
    results_dir = "results"
    
    # Data split
    train_ratio = 0.8
    val_ratio = 0.2
    frequency_threshold = 1000  # Classes >= 1000 samples go to Encoder1
    
    # Class names (7 intents)
    classes = ['Open Domain', 'Transport', 'Faculty', 'Miscellaneous', 
               'Admission', 'Programs', 'Hostels']
    num_classes = 7
    
    # Encoder1: frequent classes
    frequent_classes = ['Transport', 'Faculty', 'Open Domain']
    # Encoder2: rare classes  
    rare_classes = ['Admission', 'Programs', 'Hostels', 'Miscellaneous']
    
    # Domain-specific classes for SmolLM
    domain_classes = ['Transport', 'Faculty', 'Admission', 'Programs', 
                      'Hostels', 'Miscellaneous']
    # Open Domain for Phi-1.5
    open_domain_classes = ['Open Domain']
    
    # Encoder architecture
    vocab_size = 30522
    hidden_dim = 384
    num_layers = 3
    num_heads = 6
    max_seq_length = 128
    dropout = 0.1
    
    # Fusion module
    fusion_output_dim = 192
    fusion_hidden_dim = 256
    
    # LoRA configuration
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Stage 1: Encoder training
    stage1_epochs = 15
    stage1_lr = 3e-4
    stage1_batch_size = 32
    
    # Stage 2: Fusion training
    stage2_epochs = 10
    stage2_lr = 1e-4
    stage2_batch_size = 32
    
    # Stage 3: Decoder training
    stage3_epochs = 5
    stage3_lr = 2e-4
    stage3_batch_size = 8  # Small batch due to decoder size
    stage3_grad_accum = 4  # Effective batch size: 32
    
    # Optimization
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    
    # Decoder models
    smollm_model = "HuggingFaceTB/SmolLM-135M"
    phi_model = "microsoft/phi-1_5"
    
    # Evaluation
    k_folds = 5
    confidence_interval = 0.95
    
    # Random seed for reproducibility
    seed = 42