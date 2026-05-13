"""
Stage 2: Channel-attention fusion + classifier training.

Key fixes vs. original:
  1. Loads encoder1 + encoder2 from Stage 1 and FREEZES them
     (set FREEZE_ENCODERS=False below if you want to fine-tune).
  2. Focal loss with gamma=2.0 + class weights. Focal loss naturally
     down-weights well-classified majority samples; you do not need to
     guess perfect class weights to escape the collapse.
  3. WeightedRandomSampler so each epoch sees a balanced label distribution.
  4. Early stop on macro-F1.
  5. Cosine LR schedule with warmup.
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

from models import (
    TransformerEncoder, ChannelAttentionFusion, IntentClassifier,
)
from data_loader import load_data, ChatbotDataset
from config import Config


FREEZE_ENCODERS = True   # safest default; flip to False for end-to-end fine-tune


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional class weighting.
    L = -alpha_c * (1 - p_c)^gamma * log(p_c)
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        # p_t = exp(-ce) is valid when label_smoothing=0; with smoothing it
        # is a good enough proxy and keeps the focal modulator well-defined.
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_balanced_sampler(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    per_class = 1.0 / np.sqrt(counts)
    sample_w = per_class[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / np.sqrt(counts)
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def load_encoder(config, ckpt_path):
    encoder = TransformerEncoder(config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    return encoder


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def evaluate(encoder1, encoder2, fusion, classifier, loader, device, num_classes):
    encoder1.eval(); encoder2.eval(); fusion.eval(); classifier.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["intent"].to(device)
            h1 = encoder1(input_ids, attn)
            h2 = encoder2(input_ids, attn)
            h_fused, _ = fusion(h1, h2)
            logits = classifier(h_fused)
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            gts.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(gts, preds)
    macro_f1 = f1_score(gts, preds, average="macro",
                        labels=list(range(num_classes)), zero_division=0)
    return acc, macro_f1, preds, gts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = config.device

    print("="*70)
    print("STAGE 2: FUSION + CLASSIFIER TRAINING (fixed)")
    print("="*70)

    data_dict = load_data(config)
    tokenizer = data_dict["tokenizer"]
    class_to_idx = data_dict["class_to_idx"]
    train_df = data_dict["train_full"]
    val_df = data_dict["val_full"]

    train_ds = ChatbotDataset(train_df, tokenizer, config.max_seq_length, class_to_idx)
    val_ds = ChatbotDataset(val_df, tokenizer, config.max_seq_length, class_to_idx)

    train_labels = np.array([class_to_idx[i] for i in train_df["intent"]])
    class_weights = compute_class_weights(train_labels, config.num_classes).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=config.stage2_batch_size,
        sampler=make_balanced_sampler(train_labels, config.num_classes),
        num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.stage2_batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    # Load encoders from Stage 1
    enc1_path = f"{config.checkpoint_dir}/stage1_encoders/encoder1_frequent.pt"
    enc2_path = f"{config.checkpoint_dir}/stage1_encoders/encoder2_rare.pt"
    if not (os.path.exists(enc1_path) and os.path.exists(enc2_path)):
        raise FileNotFoundError(
            "Stage 1 checkpoints missing. Run train_stage1.py first."
        )
    encoder1 = load_encoder(config, enc1_path).to(device)
    encoder2 = load_encoder(config, enc2_path).to(device)

    fusion = ChannelAttentionFusion(config).to(device)
    classifier = IntentClassifier(config).to(device)

    if FREEZE_ENCODERS:
        for p in encoder1.parameters():
            p.requires_grad = False
        for p in encoder2.parameters():
            p.requires_grad = False
        encoder1.eval(); encoder2.eval()
        trainable_params = list(fusion.parameters()) + list(classifier.parameters())
    else:
        trainable_params = (
            list(encoder1.parameters()) + list(encoder2.parameters())
            + list(fusion.parameters()) + list(classifier.parameters())
        )

    optimizer = AdamW(
        trainable_params,
        lr=config.stage2_lr,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )

    total_steps = max(1, len(train_loader) * config.stage2_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.stage2_lr,
        total_steps=total_steps, pct_start=0.1, anneal_strategy="cos",
    )

    criterion = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=0.05)

    best_f1, patience, bad = -1.0, 4, 0
    os.makedirs(f"{config.checkpoint_dir}/stage2_fusion", exist_ok=True)

    for epoch in range(config.stage2_epochs):
        if not FREEZE_ENCODERS:
            encoder1.train(); encoder2.train()
        fusion.train(); classifier.train()

        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"stage2 epoch {epoch+1}/{config.stage2_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["intent"].to(device)

            if FREEZE_ENCODERS:
                with torch.no_grad():
                    h1 = encoder1(input_ids, attn)
                    h2 = encoder2(input_ids, attn)
            else:
                h1 = encoder1(input_ids, attn)
                h2 = encoder2(input_ids, attn)

            h_fused, _ = fusion(h1, h2)
            logits = classifier(h_fused)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            n += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running / max(n, 1)
        acc, macro_f1, preds, gts = evaluate(
            encoder1, encoder2, fusion, classifier,
            val_loader, device, config.num_classes,
        )
        print(f"  epoch {epoch+1}: train_loss={train_loss:.4f} | "
              f"val acc={acc:.4f} macroF1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            bad = 0
            torch.save(
                {
                    "fusion": fusion.state_dict(),
                    "classifier": classifier.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": macro_f1,
                    "val_acc": acc,
                    # Persist these too so eval can re-build the exact label order
                    "class_to_idx": class_to_idx,
                },
                f"{config.checkpoint_dir}/stage2_fusion/fusion_classifier.pt",
            )
            # Snapshot a readable per-class report
            report = classification_report(
                gts, preds,
                labels=list(range(config.num_classes)),
                target_names=[k for k, _ in sorted(class_to_idx.items(),
                                                   key=lambda x: x[1])],
                zero_division=0,
                output_dict=True,
            )
            os.makedirs(config.results_dir, exist_ok=True)
            with open(f"{config.results_dir}/stage2_val_report.json", "w") as f:
                json.dump(report, f, indent=2)
            print(f"  saved (macroF1={macro_f1:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    print("\n" + "="*70)
    print(f"STAGE 2 COMPLETE | best val macroF1: {best_f1:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()