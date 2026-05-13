"""
Stage 3: Decoder LoRA fine-tuning (SmolLM for domain, Phi-1.5 for open-domain).

Key fixes vs. the original code:
  1. Uses each decoder's OWN tokenizer (not BERT). Feeding BERT ids into a
     causal LM is the silent killer of generation quality.
  2. Causal-LM labels = input_ids with pad tokens masked to -100.
  3. Gradient accumulation, mixed precision when CUDA is available.
  4. Skips a decoder cleanly if its split is empty (e.g. no Open Domain data).
  5. Loss is computed on prompt + " " + response, with the prompt tokens
     also masked to -100 so the model is only trained to produce the response.
"""

import os
import gc
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from models import create_lora_decoder
from data_loader import load_data
from config import Config


# ---------------------------------------------------------------------------
# Dataset built around the decoder's own tokenizer
# ---------------------------------------------------------------------------
class CausalLMDataset(Dataset):
    """
    Builds (input_ids, attention_mask, labels) where:
      - tokens belonging to the prompt are labelled -100 (ignored in loss)
      - tokens belonging to the response are kept as their token ids
      - pad tokens are labelled -100
    """
    def __init__(self, df, tokenizer, max_length, prompt_template="Q: {q}\nA: "):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = self.prompt_template.format(q=str(row["query"]))
        response = str(row["response"])

        # Tokenise prompt and full text separately so we know the prompt length
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]
        full_text = prompt + response + self.tokenizer.eos_token
        full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = full["input_ids"].squeeze(0)
        attn = full["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        # mask prompt tokens
        prompt_len = min(len(prompt_ids), self.max_length)
        labels[:prompt_len] = -100
        # mask pad tokens
        labels[attn == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Training loop for one decoder
# ---------------------------------------------------------------------------
def train_one_decoder(model, tokenizer, train_df, val_df, config, name):
    if len(train_df) == 0:
        print(f"\n[stage3] skipping {name}: no training data in this split.")
        return None

    device = config.device
    train_ds = CausalLMDataset(train_df, tokenizer, config.max_seq_length)
    val_ds = CausalLMDataset(val_df, tokenizer, config.max_seq_length) \
        if len(val_df) > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=config.stage3_batch_size,
        shuffle=True, num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.stage3_batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    ) if val_ds is not None else None

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable, lr=config.stage3_lr,
        weight_decay=config.weight_decay, eps=config.adam_epsilon,
    )

    steps_per_epoch = max(1, len(train_loader) // config.stage3_grad_accum)
    total_steps = max(1, steps_per_epoch * config.stage3_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.stage3_lr,
        total_steps=total_steps, pct_start=0.1, anneal_strategy="cos",
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    os.makedirs(f"{config.checkpoint_dir}/stage3_decoders/{name}", exist_ok=True)

    for epoch in range(config.stage3_epochs):
        model.train()
        running, n = 0.0, 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"[{name}] epoch {epoch+1}/{config.stage3_epochs}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss / config.stage3_grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % config.stage3_grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            running += loss.item() * config.stage3_grad_accum
            n += 1
            pbar.set_postfix({"loss": f"{(running/max(n,1)):.4f}"})

        train_loss = running / max(n, 1)

        # validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            v_running, v_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attn = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                    v_running += out.loss.item()
                    v_n += 1
            val_loss = v_running / max(v_n, 1)

        msg = f"  epoch {epoch+1}: train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.4f}"
        print(msg)

        # Save best (val if available, else train)
        score = val_loss if val_loss is not None else train_loss
        if score < best_val:
            best_val = score
            save_dir = f"{config.checkpoint_dir}/stage3_decoders/{name}"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  saved {name} (score={best_val:.4f})")

    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = config.device

    print("="*70)
    print("STAGE 3: DECODER LORA FINE-TUNING (fixed)")
    print("="*70)

    data_dict = load_data(config)
    train_smollm = data_dict["train_smollm"]
    val_smollm = data_dict["val_smollm"]
    train_phi = data_dict["train_phi"]
    val_phi = data_dict["val_phi"]

    print(f"\nSplit sizes:")
    print(f"  SmolLM  : train={len(train_smollm)} val={len(val_smollm)}")
    print(f"  Phi-1.5 : train={len(train_phi)} val={len(val_phi)}")

    # ----- SmolLM (domain) -----
    print("\n--- Loading SmolLM ---")
    smollm_tok = AutoTokenizer.from_pretrained(config.smollm_model)
    if smollm_tok.pad_token is None:
        smollm_tok.pad_token = smollm_tok.eos_token
    smollm = create_lora_decoder(config, config.smollm_model, device)
    train_one_decoder(smollm, smollm_tok, train_smollm, val_smollm, config, "smollm")

    # free memory before loading the second decoder
    del smollm, smollm_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----- Phi-1.5 (open domain) -----
    print("\n--- Loading Phi-1.5 ---")
    phi_tok = AutoTokenizer.from_pretrained(config.phi_model)
    if phi_tok.pad_token is None:
        phi_tok.pad_token = phi_tok.eos_token
    phi = create_lora_decoder(config, config.phi_model, device)
    train_one_decoder(phi, phi_tok, train_phi, val_phi, config, "phi")

    print("\n" + "="*70)
    print("STAGE 3 COMPLETE")
    print(f"Checkpoints: {config.checkpoint_dir}/stage3_decoders/")
    print("="*70)


if __name__ == "__main__":
    main()