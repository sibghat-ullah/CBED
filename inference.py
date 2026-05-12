from config import Config
from models import DualEncoderSystem
from transformers import AutoTokenizer
import torch

config = Config()
model = DualEncoderSystem(config)

# Load trained weights
# (See evaluate.py for complete loading code)

# Inference
query = "When does the library close?"
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(query, return_tensors='pt', max_length=128, padding='max_length')

with torch.no_grad():
    logits, h_fused, attention = model(
        inputs['input_ids'].to(config.device),
        inputs['attention_mask'].to(config.device),
        return_attention=True
    )
    
intent = config.classes[torch.argmax(logits, dim=-1).item()]
print(f"Predicted Intent: {intent}")
print(f"Attention: Enc1={attention[0,0]:.3f}, Enc2={attention[0,1]:.3f}")