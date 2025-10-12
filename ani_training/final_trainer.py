#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
# MODIFICATION: sklearn imports are no longer needed
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score

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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_path": "../augmented_catalog_final_2.csv", # Your full dataset
    "random_seed": 42,

    # --- Use your best found hyperparameters here ---
    "model_name": "microsoft/deberta-v3-base",
    "max_length": 160,
    "dropout": 0.2,

    # Training
    "batch_size": 32,
    "epochs": 10, # Choose the number of epochs that worked best in your experiments
    "lr_encoder": 3e-5,
    "lr_head": 1e-3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "gradient_clip": 1.0,

    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": max(0, min(8, (os.cpu_count() or 2) // 2)),
    "use_amp": True,
    "use_torch_compile": False, # Set to False for broader compatibility
    "use_fused_adamw": True,
    "enable_tf32": True,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions & Model Definition (Keep these unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def create_combined_text(row):
    category = str(row['product_category']).strip()
    description = str(row['description']).strip()
    value = row['value']
    unit = str(row['unit']).strip().capitalize()
    text_parts = [f"Category: {category}"]
    if pd.notna(description) and description:
        text_parts.append(f"Description: {description}")
    if unit.lower() != 'count' or value != 1:
        text_parts.append(f"Amount: {value} {unit}")
    return " [SEP] ".join(text_parts)

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

class PriceRegressor(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
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

# --- MODIFICATION: The 'evaluate' function is no longer needed and has been removed ---

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])
    device = torch.device(CONFIG["device"])
    amp_enabled = (device.type == "cuda") and CONFIG["use_amp"]

    print("="*70)
    print("Final Model Training on Full Dataset")
    print("="*70)

    # 1) Load & preprocess
    print("\n[1/4] Loading and preprocessing full dataset...")
    df = pd.read_csv(CONFIG["data_path"])
    df.rename(columns={'misc_info': 'description'}, inplace=True)
    df.dropna(subset=['price', 'product_category'], inplace=True)
    missing_value_mask = df['value'].isna()
    df.loc[missing_value_mask, 'value'] = 1
    df.loc[missing_value_mask, 'unit'] = 'Count'

    print(f"Total samples for training: {len(df)}")
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    df['log_price'] = np.log1p(df['price'])

    # --- MODIFICATION: No train-test split. Use the entire dataframe. ---
    train_texts = df['combined_text'].tolist()
    train_prices = df['log_price'].values

    # 2) Tokenizer + Datasets
    print("\n[2/4] Initializing tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=True)
    train_enc = tokenizer(train_texts, truncation=True, max_length=CONFIG["max_length"], padding=False)
    train_dataset = PriceDataset(train_enc, train_prices)

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
        loader_common.update(num_workers=CONFIG["num_workers"], persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_common)
    # --- MODIFICATION: test_loader is removed. ---

    # 3) Build model
    print("\n[3/4] Building model...")
    model = PriceRegressor(CONFIG["model_name"], dropout=CONFIG["dropout"]).to(device)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params    = list(model.regressor.parameters())
    opt_kwargs = {"weight_decay": CONFIG["weight_decay"]}
    if device.type == "cuda" and CONFIG["use_fused_adamw"]:
        opt_kwargs["fused"] = True

    optimizer = AdamW([
        {'params': encoder_params, 'lr': CONFIG["lr_encoder"]},
        {'params': head_params,    'lr': CONFIG["lr_head"]}
    ], **opt_kwargs)

    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 4) Train
    print("\n[4/4] Training the final model...")
    print("-"*70)
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, amp_enabled)
        print(f"  Train Loss (Huber): {train_loss:.4f}")

    # --- MODIFICATION: Evaluation step is removed. ---

    # --- FINAL STEP: Save the trained model ---
    final_model_path = 'final_price_predictor_msft_10.pt'
    torch.save(model.state_dict(), final_model_path)
    print("\n" + "="*70)
    print(f"✅ Final model trained on all data and saved to '{final_model_path}'")
    print("="*70)


if __name__ == "__main__":
    main()