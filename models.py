import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import math

class TransformerEncoder(nn.Module):
    """Lightweight Transformer Encoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids, attention_mask):
        batch_size, seq_length = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = self.dropout(token_embeds + position_embeds)
        
        # Create attention mask for transformer (True = masked position)
        extended_attention_mask = (1.0 - attention_mask.unsqueeze(1)) * -10000.0
        
        # Transformer encoding
        encoder_output = self.transformer(
            embeddings,
            src_key_padding_mask=(attention_mask == 0)
        )
        
        # Pool: take [CLS] token (first token)
        pooled_output = encoder_output[:, 0, :]
        pooled_output = self.layer_norm(pooled_output)
        
        return pooled_output

class ChannelAttentionFusion(nn.Module):
    """Channel Attention Fusion Module"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Attention network: [h1 ⊕ h2] → hidden → attention_weights
        self.projection = nn.Linear(config.hidden_dim * 2, config.fusion_hidden_dim)
        self.attention = nn.Linear(config.fusion_hidden_dim, 2)
        self.relu = nn.ReLU()
        
        # Output projection to reduce dimensionality
        self.output_projection = nn.Linear(config.hidden_dim, config.fusion_output_dim)
        
    def forward(self, h1, h2):
        # Concatenate encoder outputs
        concat = torch.cat([h1, h2], dim=-1)  # [batch, 768]
        
        # Compute attention weights
        hidden = self.relu(self.projection(concat))  # [batch, 256]
        attention_logits = self.attention(hidden)    # [batch, 2]
        attention_weights = F.softmax(attention_logits, dim=-1)  # [batch, 2]
        
        # Weighted combination
        alpha1 = attention_weights[:, 0].unsqueeze(-1)  # [batch, 1]
        alpha2 = attention_weights[:, 1].unsqueeze(-1)  # [batch, 1]
        h_fused = alpha1 * h1 + alpha2 * h2  # [batch, 384]
        
        # Project to fusion output dimension
        h_fused = self.output_projection(h_fused)  # [batch, 192]
        
        return h_fused, attention_weights

class IntentClassifier(nn.Module):
    """Classification head"""
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(config.fusion_output_dim, config.num_classes)
        
    def forward(self, h_fused):
        logits = self.classifier(h_fused)
        return logits

class DualEncoderSystem(nn.Module):
    """Complete dual encoder + fusion + classification system"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder1 = TransformerEncoder(config)
        self.encoder2 = TransformerEncoder(config)
        self.fusion = ChannelAttentionFusion(config)
        self.classifier = IntentClassifier(config)
        
    def forward(self, input_ids, attention_mask, return_attention=False):
        # Dual encoding
        h1 = self.encoder1(input_ids, attention_mask)
        h2 = self.encoder2(input_ids, attention_mask)
        
        # Fusion with attention
        h_fused, attention_weights = self.fusion(h1, h2)
        
        # Classification
        logits = self.classifier(h_fused)
        
        if return_attention:
            return logits, h_fused, attention_weights
        return logits, h_fused

def create_lora_decoder(config, model_name, device):
    """Create decoder with LoRA fine-tuning"""
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model