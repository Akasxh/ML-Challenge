#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
CONFIG = {
    "model_name": "microsoft/deberta-v3-base",
    "saved_model_path": "final_price_predictor_msft_10.pt",  # ✅ Fixed: matches Code 1 output
    "test_csv_path": "augmented_test_final.csv",
    "output_csv_path": "prediction_out.csv",
    "max_length": 160,
    "dropout": 0.2,
    "batch_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"Using device: {CONFIG['device']}")

# --- Helper Functions ---
def create_combined_text(row):
    """Must match training exactly"""
    category = str(row.get('product_category', '')).strip()
    description = str(row.get('description', '')).strip()  # Note: Capital 'D'
    value = row.get('value', 1.0)
    unit = str(row.get('unit', 'Count')).strip().capitalize()

    if pd.isna(value):
        value = 1.0
        unit = 'Count'

    text_parts = [f"Category: {category}"]
    if pd.notna(description) and description:
        text_parts.append(f"Description: {description}")
    if unit.lower() != 'count' or value != 1:
        text_parts.append(f"Amount: {value} {unit}")

    return " [SEP] ".join(text_parts)


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


class TestDataset(Dataset):
    """Dataset with dynamic padding support"""
    def __init__(self, encodings, sample_ids):
        self.encodings = encodings
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        item = {k: self.encodings[k][idx] for k in ('input_ids', 'attention_mask')}
        item['sample_id'] = self.sample_ids[idx]
        return item


# --- Main Execution ---
def main():
    # 1. Load tokenizer and model
    print(f"Loading tokenizer for '{CONFIG['model_name']}'...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], use_fast=True)

    print(f"Loading model from '{CONFIG['saved_model_path']}'...")
    if not os.path.exists(CONFIG['saved_model_path']):
        raise FileNotFoundError(f"Model file not found: {CONFIG['saved_model_path']}")

    model = PriceRegressor(
        model_name=CONFIG['model_name'],
        dropout=CONFIG['dropout']
    )
    model.load_state_dict(
        torch.load(CONFIG['saved_model_path'], map_location=torch.device(CONFIG['device']))
    )
    model.to(CONFIG['device'])
    model.eval()
    print("✅ Model loaded successfully.")

    # 2. Load and prepare test data
    print(f"Loading test data from '{CONFIG['test_csv_path']}'...")
    if not os.path.exists(CONFIG['test_csv_path']):
        raise FileNotFoundError(f"Test file not found: {CONFIG['test_csv_path']}")

    df_test = pd.read_csv(CONFIG['test_csv_path'])
    
    # ✅ Fixed: Apply same preprocessing as training
    # Check if column renaming is needed (depends on your test.csv structure)
    
    # Handle missing values
    missing_value_mask = df_test['value'].isna()
    df_test.loc[missing_value_mask, 'value'] = 1
    df_test.loc[missing_value_mask, 'unit'] = 'Count'

    # Create combined text
    df_test['combined_text'] = df_test.apply(create_combined_text, axis=1)
    print(f"Prepared {len(df_test)} samples for prediction.")

    # 3. Tokenize (without padding - done dynamically in collator)
    print("Tokenizing test data...")
    test_encodings = tokenizer(
        df_test['combined_text'].tolist(),
        truncation=True,
        max_length=CONFIG['max_length'],
        padding=False  # ✅ Fixed: No padding here
    )

    # 4. Create dataset and dataloader with dynamic padding
    test_dataset = TestDataset(test_encodings, df_test['sample_id'].values)
    
    # ✅ Fixed: Use DataCollator for dynamic padding
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")
    
    def collate_fn(batch):
        sample_ids = [ex['sample_id'] for ex in batch]
        features = [{k: ex[k] for k in ('input_ids', 'attention_mask')} for ex in batch]
        batch_out = collator(features)
        batch_out['sample_id'] = sample_ids
        return batch_out

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(CONFIG['device'] == "cuda")
    )

    # 5. Run predictions
    all_predictions = []
    all_sample_ids = []

    print("Running predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(CONFIG['device'], non_blocking=True)
            attention_mask = batch['attention_mask'].to(CONFIG['device'], non_blocking=True)
            sample_ids = batch['sample_id']

            # Get predictions in log space
            log_preds = model(input_ids, attention_mask)

            # Transform back to original price scale
            actual_preds = np.expm1(log_preds.cpu().numpy())

            all_predictions.extend(actual_preds)
            all_sample_ids.extend(sample_ids)

    print("✅ Predictions complete.")

    # 6. Create submission file
    print(f"Saving predictions to '{CONFIG['output_csv_path']}'...")
    df_submission = pd.DataFrame({
        'sample_id': all_sample_ids,
        'price': all_predictions
    })

    df_submission = df_submission[['sample_id', 'price']]
    df_submission.to_csv(CONFIG['output_csv_path'], index=False)

    print(f"✅ Submission file created successfully!")
    print(f"   Predicted prices range: ${df_submission['price'].min():.2f} - ${df_submission['price'].max():.2f}")
    print(f"   Mean predicted price: ${df_submission['price'].mean():.2f}")


if __name__ == "__main__":
    main()