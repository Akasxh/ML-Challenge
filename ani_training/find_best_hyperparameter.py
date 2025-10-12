#!/usr/bin/env python3
"""
Manual Grid Search for Hyperparameter Tuning
Simple, interpretable approach to find best hyperparameters
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
    AutoTokenizer, AutoModel, DataCollatorWithPadding,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import itertools
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER GRID - CUSTOMIZE THIS!
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    # Model choices
    "model_name": [
        "distilbert-base-uncased",
    ],
    
    # Architecture
    "max_length": [128, 160, 192],
    "dropout": [0.2, 0.3, 0.4],
    "hidden_dims": [
        [512, 256],      # Current (good baseline)
        [768, 384],      # Larger
        [256],           # Simpler
    ],
    
    # Training
    "batch_size": [32, 64],
    "epochs": [5, 7, 10],
    "lr_encoder": [1e-5, 2e-5, 3e-5],  # Most important!
    "lr_head": [5e-4, 1e-3, 2e-3],
    "weight_decay": [0.01, 0.05],
    "warmup_ratio": [0.05, 0.1, 0.15],
}

# Fixed config
BASE_CONFIG = {
    "data_path": "../augmented_catalog_final_2.csv",
    "total_samples": 20000,
    "test_split": 0.2,
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": max(0, min(8, (os.cpu_count() or 2) // 2)),
    "use_amp": True,
    "gradient_clip": 1.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def create_combined_text(row):
    category = str(row['product_category']).strip()
    description = str(row['Description']).strip()
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
    def __init__(self, model_name: str, dropout: float = 0.2, hidden_dims: list = [512, 256]):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        layers = [nn.LayerNorm(hidden_size), nn.Dropout(dropout)]
        prev_dim = hidden_size
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.regressor = nn.Sequential(*layers)

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
        nn.utils.clip_grad_norm_(model.parameters(), BASE_CONFIG["gradient_clip"])
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
        
        preds_log.append(preds.cpu().numpy())
        targets_log.append(targets.cpu().numpy())
    
    preds_log = np.concatenate(preds_log)
    targets_log = np.concatenate(targets_log)
    
    preds_orig = np.expm1(preds_log)
    targets_orig = np.expm1(targets_log)
    
    mae = mean_absolute_error(targets_orig, preds_orig)
    r2 = r2_score(targets_orig, preds_orig)
    
    # SMAPE
    numerator = np.abs(preds_orig - targets_orig)
    denominator = (np.abs(targets_orig) + np.abs(preds_orig)) / 2
    smape = np.mean(numerator / (denominator + 1e-8)) * 100
    
    return mae, r2, smape

# ─────────────────────────────────────────────────────────────────────────────
# Training Function for One Configuration
# ─────────────────────────────────────────────────────────────────────────────
def train_with_config(config, train_texts, test_texts, train_prices, test_prices, device):
    """Train model with given hyperparameters and return test performance"""
    
    print(f"\n{'─'*70}")
    print(f"Testing configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"{'─'*70}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
    train_enc = tokenizer(train_texts, truncation=True, max_length=config['max_length'], padding=False)
    test_enc = tokenizer(test_texts, truncation=True, max_length=config['max_length'], padding=False)
    
    train_dataset = PriceDataset(train_enc, train_prices)
    test_dataset = PriceDataset(test_enc, test_prices)
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")
    
    def collate_fn(batch):
        prices = torch.tensor([ex['price'] for ex in batch], dtype=torch.float32)
        features = [{k: ex[k] for k in ('input_ids', 'attention_mask')} for ex in batch]
        batch_out = collator(features)
        batch_out['price'] = prices
        return batch_out
    
    loader_kwargs = {
        'batch_size': config['batch_size'],
        'collate_fn': collate_fn,
        'pin_memory': (device.type == 'cuda'),
    }
    if BASE_CONFIG["num_workers"] > 0:
        loader_kwargs.update(num_workers=BASE_CONFIG["num_workers"], persistent_workers=True, prefetch_factor=2)
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    # Model
    model = PriceRegressor(
        config['model_name'],
        dropout=config['dropout'],
        hidden_dims=config['hidden_dims']
    ).to(device)
    
    # Optimizer
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.regressor.parameters())
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': config['lr_encoder']},
        {'params': head_params, 'lr': config['lr_head']}
    ], weight_decay=config['weight_decay'])
    
    # Scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=BASE_CONFIG["use_amp"])
    
    # Training loop
    best_mae = float('inf')
    best_r2 = 0.0
    best_smape = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, BASE_CONFIG["use_amp"])
        mae, r2, smape = evaluate(model, test_loader, device, BASE_CONFIG["use_amp"])
        
        print(f"Epoch {epoch}/{config['epochs']} - Train Loss: {train_loss:.4f}, "
              f"Test MAE: ${mae:.2f}, R²: {r2:.4f}, SMAPE: {smape:.2f}%")
        
        best_mae = min(best_mae, mae)
        best_r2 = max(best_r2, r2)
        best_smape = min(best_smape, smape)
    
    # Clean up
    del model, optimizer, scheduler, train_loader, test_loader
    torch.cuda.empty_cache()
    
    return best_mae, best_r2, best_smape

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("Manual Hyperparameter Grid Search")
    print("="*70)
    
    device = torch.device(BASE_CONFIG["device"])
    
    # Load data once
    print("\nLoading data...")
    df = pd.read_csv(BASE_CONFIG["data_path"])

    df.rename(columns={'misc_info': 'description'}, inplace=True)
    df.dropna(subset=['price', 'product_category'], inplace=True)
    
    missing_value_mask = df['value'].isna()
    df.loc[missing_value_mask, 'value'] = 1
    df.loc[missing_value_mask, 'unit'] = 'Count'
    
    df = df.sample(n=BASE_CONFIG["total_samples"], random_state=BASE_CONFIG["random_seed"])
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    df['log_price'] = np.log1p(df['price'])
    
    train_df, test_df = train_test_split(
        df, test_size=BASE_CONFIG["test_split"],
        random_state=BASE_CONFIG["random_seed"]
    )
    
    train_texts = train_df['combined_text'].tolist()
    test_texts = test_df['combined_text'].tolist()
    train_prices = train_df['log_price'].values
    test_prices = test_df['log_price'].values
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    
    print(f"\nTotal configurations to test: {len(combinations)}")
    print("="*70)
    
    # Track results
    results = []
    
    # Test each configuration
    for i, combo in enumerate(combinations, 1):
        config = dict(zip(keys, combo))
        
        print(f"\n[{i}/{len(combinations)}] Starting configuration...")
        start_time = datetime.now()
        
        try:
            mae, r2, smape = train_with_config(config, train_texts, test_texts, 
                                               train_prices, test_prices, device)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            result = {
                **config,
                'test_mae': mae,
                'test_r2': r2,
                'test_smape': smape,
                'time_seconds': elapsed
            }
            results.append(result)
            
            print(f"\n✅ Configuration {i} completed in {elapsed:.0f}s")
            print(f"   Best MAE: ${mae:.2f}, R²: {r2:.4f}, SMAPE: {smape:.2f}%")
            
        except Exception as e:
            print(f"\n❌ Configuration {i} failed: {str(e)}")
            continue
    
    # Save and display results
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE")
    print("="*70)
    
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('test_mae')
        df_results.to_csv('grid_search_results.csv', index=False)
        print(f"\n✅ Results saved to 'grid_search_results.csv'")
        
        print("\n🏆 TOP 5 CONFIGURATIONS BY MAE:")
        print("-"*70)
        for idx, row in df_results.head(5).iterrows():
            print(f"\nRank {df_results.index.get_loc(idx) + 1}:")
            print(f"  MAE: ${row['test_mae']:.2f} | R²: {row['test_r2']:.4f} | SMAPE: {row['test_smape']:.2f}%")
            print(f"  model: {row['model_name']}")
            print(f"  max_length: {row['max_length']}, dropout: {row['dropout']}, hidden_dims: {row['hidden_dims']}")
            print(f"  batch_size: {row['batch_size']}, epochs: {row['epochs']}")
            print(f"  lr_encoder: {row['lr_encoder']}, lr_head: {row['lr_head']}")
            print(f"  weight_decay: {row['weight_decay']}, warmup_ratio: {row['warmup_ratio']}")
        
        print("\n" + "="*70)
        print("BEST HYPERPARAMETERS:")
        print("="*70)
        best = df_results.iloc[0]
        print(f"MAE: ${best['test_mae']:.2f}")
        print(f"R²: {best['test_r2']:.4f}")
        print(f"SMAPE: {best['test_smape']:.2f}%")
        print(f"\nConfiguration:")
        for key in PARAM_GRID.keys():
            print(f"  {key}: {best[key]}")
    else:
        print("\n❌ No successful configurations")

if __name__ == "__main__":
    main()