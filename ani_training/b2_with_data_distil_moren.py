#!/usr/bin/env python3
"""
Fast Text‑only price predictor (Train/Test) - Adapted for multi-column data
Key speedups: fast tokenizer + pretokenization + dynamic padding,
mixed precision on GPU, fused AdamW (when available), torch.compile,
TF32 on Ampere+, tuned DataLoader, optional layer freezing.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_path": "../augmented_catalog_final_2.csv",
    "total_samples_to_use": 74999, # Use 20k for quick tests; set to None to use all
    "test_split": 0.1, # Adjusted to match 15k/5k split from 20k samples
    "random_seed": 2,

    # Model / tokenization
    "model_name": "distilbert-base-uncased",
    "max_length": 160,
    "dropout": 0.2,

    # Training
    "batch_size": 32, # Can often be increased with AMP
    "epochs": 10,
    "lr_encoder": 2e-5,
    "lr_head": 1e-3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "gradient_clip": 1.0,

    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": max(0, min(8, (os.cpu_count() or 2) // 2)),
    "use_amp": True,
    "use_torch_compile": True,
    "use_fused_adamw": True,
    "enable_tf32": True,

    # Optional: freeze lower layers for speed (DistilBERT has 6 layers)
    "freeze_n_layers": 0,
    "freeze_all_encoder": False
}

# ─────────────────────────────────────────────────────────────────────────────
# Custom Metric & Data Preparation Function
# ─────────────────────────────────────────────────────────────────────────────
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    epsilon = 1e-8 # To avoid division by zero
    return np.mean(numerator / (denominator + epsilon)) * 100

def create_combined_text(row):
    """
    Combines structured data from a row into a single descriptive string
    that a language model can understand.
    """
    category = str(row['product_category']).strip()
    description = str(row['description']).strip()
    value = row['value']
    unit = str(row['unit']).strip().capitalize()

    # Use a clear, structured format
    text_parts = [f"Category: {category}"]

    if pd.notna(description) and description:
        text_parts.append(f"description: {description}")

    # Add value and unit information, handling the imputed 'Count' case
    if unit.lower() != 'count' or value != 1:
        text_parts.append(f"Amount: {value} {unit}")

    return " [SEP] ".join(text_parts)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset (pre-tokenized; dynamic padding happens in collate_fn)
# ─────────────────────────────────────────────────────────────────────────────
class PriceDataset(Dataset):
    def __init__(self, encodings, prices):
        self.encodings = encodings
        self.prices = prices

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx):
        item = {k: self.encodings[k][idx] for k in ('input_ids', 'attention_mask')}
        item['price'] = torch.tensor(self.prices[idx], dtype=torch.float32)
        return item

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class PriceRegressor(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),  # First hidden layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),         # Second hidden layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.regressor(pooled).squeeze(-1)

def _try_freeze_layers(encoder, n_layers: int, freeze_all: bool):
    """Freeze first n_layers of the transformer stack."""
    if freeze_all and hasattr(encoder, "embeddings"):
        for p in encoder.embeddings.parameters():
            p.requires_grad = False

    layers = None
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
        layers = encoder.transformer.layer
    elif hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer

    if layers is not None:
        if freeze_all:
            n_layers = len(layers)
        for layer in list(layers)[:max(0, n_layers)]:
            for p in layer.parameters():
                p.requires_grad = False

# ─────────────────────────────────────────────────────────────────────────────
# Training / Evaluation Loops
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, scaler, amp_enabled):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        targets = batch['price'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
            preds = model(input_ids, attention_mask)
            loss = nn.functional.huber_loss(preds, targets, delta=0.5)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += float(loss)
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, amp_enabled):
    model.eval()
    preds_log, targets_log = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        targets = batch['price'].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
            preds = model(input_ids, attention_mask)

        preds_log.append(preds.detach().cpu().numpy())
        targets_log.append(targets.detach().cpu().numpy())

    preds_log = np.concatenate(preds_log)
    targets_log = np.concatenate(targets_log)

    preds_orig = np.expm1(preds_log)
    targets_orig = np.expm1(targets_log)

    mae = mean_absolute_error(targets_orig, preds_orig)
    r2 = r2_score(targets_orig, preds_orig)
    smape_val = smape(targets_orig, preds_orig)
    return mae, r2, smape_val, preds_orig, targets_orig

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    device = torch.device(CONFIG["device"])
    amp_enabled = (device.type == "cuda") and CONFIG["use_amp"]

    if device.type == "cuda" and CONFIG["enable_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    print("="*70)
    print("Fast Text-only Product Price Predictor (Adapted for Multi-Column CSV)")
    print("="*70)

    # 1) Load & preprocess (ADAPTED SECTION)
    print("\n[1/5] Loading and preprocessing data...")
    try:
        df_full = pd.read_csv(CONFIG["data_path"])
    except FileNotFoundError:
        print(f"Error: Data file not found at {CONFIG['data_path']}")
        return

    # **Data cleaning from original script**
    df_full.rename(columns={'misc_info': 'description'}, inplace=True)
    df_full.dropna(subset=['price', 'product_category'], inplace=True)
    missing_value_mask = df_full['value'].isna()
    df_full.loc[missing_value_mask, 'value'] = 1
    df_full.loc[missing_value_mask, 'unit'] = 'Count'

    # **Sample the cleaned dataframe**
    df = df_full.sample(n=CONFIG["total_samples_to_use"], random_state=CONFIG["random_seed"])
    print(f"Loaded and cleaned data. Using a random sample of {len(df)} rows.")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"Price mean: ${df['price'].mean():.2f}, median: ${df['price'].median():.2f}")

    # **Feature Engineering: Combine columns into a single text input**
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    df['log_price'] = np.log1p(df['price'])

    train_df, test_df = train_test_split(
        df,
        test_size=CONFIG["test_split"],
        random_state=CONFIG["random_seed"],
        shuffle=True
    )
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    train_texts = train_df['combined_text'].tolist()
    test_texts  = test_df['combined_text'].tolist()
    train_prices = train_df['log_price'].values
    test_prices  = test_df['log_price'].values

    # 2) Tokenizer + pre-tokenization + dynamic padding via collator
    print("\n[2/5] Initializing tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=True)

    train_enc = tokenizer(train_texts, truncation=True, max_length=CONFIG["max_length"], padding=False)
    test_enc = tokenizer(test_texts, truncation=True, max_length=CONFIG["max_length"], padding=False)

    train_dataset = PriceDataset(train_enc, train_prices)
    test_dataset  = PriceDataset(test_enc, test_prices)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")

    def collate_fn(batch):
        prices = torch.tensor([ex['price'] for ex in batch], dtype=torch.float32)
        features = [{k: ex[k] for k in ('input_ids', 'attention_mask')} for ex in batch]
        batch_out = collator(features)
        batch_out['price'] = prices
        return batch_out

    loader_common = dict(
        batch_size=CONFIG["batch_size"],
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )
    if CONFIG["num_workers"] > 0:
        loader_common.update(
            num_workers=CONFIG["num_workers"],
            persistent_workers=True,
            prefetch_factor=2
        )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_common)
    test_loader  = DataLoader(test_dataset, shuffle=False, **loader_common)

    # 3) Build model
    print("\n[3/5] Building model...")
    model = PriceRegressor(CONFIG["model_name"], dropout=CONFIG["dropout"]).to(device)

    _try_freeze_layers(model.encoder, CONFIG["freeze_n_layers"], CONFIG["freeze_all_encoder"])
    if CONFIG["freeze_all_encoder"]:
        print("Encoder frozen: ALL layers")
    elif CONFIG["freeze_n_layers"] > 0:
        print(f"Encoder partially frozen: first {CONFIG['freeze_n_layers']} layers")

    if CONFIG["use_torch_compile"] and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("✓ torch.compile enabled")
        except Exception as e:
            print(f"× torch.compile not enabled ({e})")

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params    = list(model.regressor.parameters())

    opt_kwargs = {"weight_decay": CONFIG["weight_decay"]}
    if device.type == "cuda" and CONFIG["use_fused_adamw"]:
        opt_kwargs["fused"] = True
    try:
        optimizer = AdamW([
            {'params': encoder_params, 'lr': CONFIG["lr_encoder"]},
            {'params': head_params,    'lr': CONFIG["lr_head"]}
        ], **opt_kwargs)
        if opt_kwargs.get("fused", False):
            print("✓ Fused AdamW enabled")
    except TypeError:
        optimizer = AdamW([
            {'params': encoder_params, 'lr': CONFIG["lr_encoder"]},
            {'params': head_params,    'lr': CONFIG["lr_head"]}
        ], weight_decay=CONFIG["weight_decay"])
        if device.type == "cuda":
            print("× Fused AdamW not available; using standard AdamW")

    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 4) Train
    print("\n[4/5] Training...")
    print("-"*70)
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, amp_enabled)
        print(f"  Train Loss (Huber): {train_loss:.4f}")

    # 5) Evaluate
    print("\n[5/5] Evaluating on test set...")
    print("-"*70)
    test_mae, test_r2, test_smape, preds, actuals = evaluate(model, test_loader, device, amp_enabled)

    print("\n" + "="*70)
    print("FINAL TEST SET RESULTS")
    print("="*70)
    print(f"Mean Absolute Error (MAE):    ${test_mae:.2f}")
    print(f"R-squared (R2):               {test_r2:.2f}")
    print(f"SMAPE:                        {test_smape:.2f}%")
    print("="*70)

    # Save model & predictions
    model_name = CONFIG["model_name"].replace('/', '_')
    model_save_path = f"{model_name}_price_predictor_{CONFIG['total_samples_to_use']}_{CONFIG['epochs']}_moren.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✓ Model saved to {model_save_path}")

    results_df = pd.DataFrame({
        'actual_price': actuals,
        'predicted_price': preds,
        'absolute_error': np.abs(actuals - preds)
    }, index=test_df.index) # Use original index
    results_df.to_csv('test_predictions_distil.csv')
    print(f"✓ Test predictions saved to test_predictions.csv")

if __name__ == "__main__":
    main()