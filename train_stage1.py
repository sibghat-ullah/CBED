"""
Stage 1: Encoder pre-training.

Key fixes vs. original:
  1. Both encoders are trained on the FULL label space (num_classes outputs),
     so the classifier neurons for every class actually receive gradient.
  2. Encoder1 sees a sampler biased toward FREQUENT classes,
     Encoder2 sees a sampler biased toward RARE classes.
     This preserves the "channel-boosting" intuition without starving any
     output neuron of training signal.
  3. Class-weighted cross-entropy with sqrt-inverse-frequency weights
     (gentler than 1/N, which previously collapsed the majority class).
  4. Label smoothing 0.05 to fight over-confidence
     (avg_confidence was 0.9998 in your last run -> textbook miscalibration).
  5. Early-stop on macro-F1, not val loss. With heavy imbalance, val loss
     is dominated by the majority class and hides minority collapse.
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from models import TransformerEncoder
from data_loader import load_data, ChatbotDataset
from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_class_weights(labels, num_classes, scheme="sqrt_inv"):
    """
    Returns a tensor of shape [num_classes] suitable for CrossEntropyLoss.

    schemes:
      - "sqrt_inv":     w_c = 1 / sqrt(count_c)        (recommended default)
      - "effective":    Cui et al. 2019, beta = 0.999
      - "inv":          w_c = 1 / count_c              (aggressive, can collapse)
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for absent classes

    if scheme == "sqrt_inv":
        w = 1.0 / np.sqrt(counts)
    elif scheme == "effective":
        beta = 0.999
        eff_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / eff_num
    elif scheme == "inv":
        w = 1.0 / counts
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # Normalise so weights have mean 1 (keeps loss magnitude familiar)
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def build_biased_sampler(labels, focus_class_ids, num_classes,
                         focus_boost=3.0):
    """
    Per-sample weights for WeightedRandomSampler.

    Every sample first gets weight 1/sqrt(class_count) (rebalances classes),
    then samples whose class is in `focus_class_ids` get multiplied by
    `focus_boost` so the encoder spends more steps on its specialty.
    Critically, ALL classes still appear (no neuron is starved).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    per_class_weight = 1.0 / np.sqrt(counts)

    sample_weights = per_class_weight[labels]
    focus_set = set(focus_class_ids)
    boost_mask = np.array([lbl in focus_set for lbl in labels], dtype=np.float64)
    sample_weights = sample_weights * (1.0 + (focus_boost - 1.0) * boost_mask)

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def evaluate_encoder(encoder, classifier, loader, device, num_classes):
    encoder.eval()
    classifier.eval()
    preds, gts = [], []
    total_loss, n_batches = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["intent"].to(device)
            feats = encoder(input_ids, attn)
            logits = classifier(feats)
            total_loss += criterion(logits, labels).item()
            n_batches += 1
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            gts.extend(labels.cpu().numpy().tolist())
    if n_batches == 0:
        return float("inf"), 0.0, 0.0
    acc = accuracy_score(gts, preds)
    macro_f1 = f1_score(gts, preds, average="macro",
                        labels=list(range(num_classes)), zero_division=0)
    return total_loss / n_batches, acc, macro_f1


# ---------------------------------------------------------------------------
# Single-encoder training loop
# ---------------------------------------------------------------------------
def train_one_encoder(encoder, train_loader, val_loader,
                      class_weights, config, encoder_name):
    device = config.device
    encoder = encoder.to(device)
    classifier = nn.Linear(config.hidden_dim, config.num_classes).to(device)

    optimizer = AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=config.stage1_lr,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )

    total_steps = max(1, len(train_loader) * config.stage1_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.stage1_lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.05,
    )

    best_macro_f1 = -1.0
    patience, bad_epochs = 4, 0

    print(f"\n{'='*60}\nTraining {encoder_name}\n{'='*60}")

    for epoch in range(config.stage1_epochs):
        encoder.train()
        classifier.train()
        running_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[{encoder_name}] epoch {epoch+1}/{config.stage1_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["intent"].to(device)

            feats = encoder(input_ids, attn)
            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / max(n, 1)
        val_loss, val_acc, val_f1 = evaluate_encoder(
            encoder, classifier, val_loader, device, config.num_classes
        )
        print(f"  epoch {epoch+1}: train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} macroF1={val_f1:.4f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            bad_epochs = 0
            os.makedirs(f"{config.checkpoint_dir}/stage1_encoders", exist_ok=True)
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc,
                },
                f"{config.checkpoint_dir}/stage1_encoders/{encoder_name}.pt",
            )
            print(f"  saved (macroF1={val_f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    return best_macro_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("="*70)
    print("STAGE 1: ENCODER PRE-TRAINING (fixed)")
    print("="*70)

    data_dict = load_data(config)
    tokenizer = data_dict["tokenizer"]
    class_to_idx = data_dict["class_to_idx"]
    train_df = data_dict["train_full"]
    val_df = data_dict["val_full"]

    # Build datasets on the FULL label space (this is the key fix).
    train_ds = ChatbotDataset(train_df, tokenizer, config.max_seq_length, class_to_idx)
    val_ds = ChatbotDataset(val_df, tokenizer, config.max_seq_length, class_to_idx)

    train_labels = np.array([class_to_idx[i] for i in train_df["intent"]])

    # Class weights for loss (shared)
    class_weights = compute_class_weights(
        train_labels, config.num_classes, scheme="sqrt_inv"
    )
    print("\nClass weights (sqrt-inverse-frequency, mean-normalised):")
    for cls_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx:>2} {cls_name:<20} weight={class_weights[idx].item():.4f}")

    # Encoder1: biased toward frequent classes
    freq_ids = [class_to_idx[c] for c in config.frequent_classes if c in class_to_idx]
    rare_ids = [class_to_idx[c] for c in config.rare_classes if c in class_to_idx]

    sampler_enc1 = build_biased_sampler(
        train_labels, focus_class_ids=freq_ids,
        num_classes=config.num_classes, focus_boost=2.0,
    )
    sampler_enc2 = build_biased_sampler(
        train_labels, focus_class_ids=rare_ids,
        num_classes=config.num_classes, focus_boost=4.0,  # stronger boost for rare
    )

    train_loader_enc1 = DataLoader(
        train_ds, batch_size=config.stage1_batch_size,
        sampler=sampler_enc1, num_workers=config.num_workers, pin_memory=True,
    )
    train_loader_enc2 = DataLoader(
        train_ds, batch_size=config.stage1_batch_size,
        sampler=sampler_enc2, num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.stage1_batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    encoder1 = TransformerEncoder(config)
    encoder2 = TransformerEncoder(config)

    print(f"\nEncoder params: ~{sum(p.numel() for p in encoder1.parameters())/1e6:.1f}M each")

    f1_enc1 = train_one_encoder(
        encoder1, train_loader_enc1, val_loader,
        class_weights, config, "encoder1_frequent",
    )
    f1_enc2 = train_one_encoder(
        encoder2, train_loader_enc2, val_loader,
        class_weights, config, "encoder2_rare",
    )

    print("\n" + "="*70)
    print("STAGE 1 COMPLETE")
    print(f"  encoder1_frequent best macroF1: {f1_enc1:.4f}")
    print(f"  encoder2_rare     best macroF1: {f1_enc2:.4f}")
    print(f"Checkpoints: {config.checkpoint_dir}/stage1_encoders/")
    print("="*70)


if __name__ == "__main__":
    main()