"""Fine-tuning pipeline for TimesFM on statistical arbitrage spreads.

This script fetches historical data for a universe of pairs, calculates their 
historical spreads, and performs fine-tuning on TimesFM using LoRA.
"""

from __future__ import annotations

import logging
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model

from pipeline.data.fetcher import StockDataFetcher
from pipeline.data.universe import list_pairs
from pipeline.stats.spread import SpreadCalculator
from pipeline.model.loader import TimesFMLoader

load_dotenv()
logger = logging.getLogger(__name__)

# Training Parameters
CONTEXT_LEN = 512
HORIZON_LEN = 64
STRIDE = 128
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 3

class SpreadDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_values": torch.tensor(item["input_values"], dtype=torch.float32),
            "label_values": torch.tensor(item["label_values"], dtype=torch.float32),
        }

def prepare_spread_dataset() -> HFDataset:
    """Fetches pair data and calculates spreads to build a fine-tuning dataset."""
    logger.info("Fetching universe pairs for training data...")
    fetcher = StockDataFetcher()
    universe = list_pairs()
    all_windows = []

    for pair in universe:
        try:
            logger.info(f"Processing {pair.ticker_a}/{pair.ticker_b}...")
            df = fetcher.fetch([pair.ticker_a, pair.ticker_b], start=TRAIN_START, end=TRAIN_END)
            prices_a = df[pair.ticker_a]["Close"]
            prices_b = df[pair.ticker_b]["Close"]
            
            # Use log spread with beta=1.0 for generalisation
            spread = np.log(prices_a.values) - np.log(prices_b.values)
            
            # Per-series z-score normalisation (TimesFM expects this)
            mean = np.mean(spread)
            std = np.std(spread)
            norm_spread = (spread - mean) / (std + 1e-9)
            
            total_len = len(norm_spread)
            for i in range(0, total_len - CONTEXT_LEN - HORIZON_LEN, STRIDE):
                window = norm_spread[i : i + CONTEXT_LEN + HORIZON_LEN]
                all_windows.append({
                    "input_values": window[:CONTEXT_LEN].tolist(),
                    "label_values": window[CONTEXT_LEN : CONTEXT_LEN + HORIZON_LEN].tolist(),
                })
        except Exception as e:
            logger.warning(f"Failed to process {pair.ticker_a}/{pair.ticker_b}: {e}")
            continue

    logger.info(f"Dataset preparation complete. Total windows: {len(all_windows)}")
    return HFDataset.from_list(all_windows)

def finetune_model(hf_dataset: HFDataset):
    """Fine-tunes the TimesFM model using LoRA."""
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    loader = TimesFMLoader.get_instance()
    if not loader.is_loaded():
        loader.load(max_context=CONTEXT_LEN, max_horizon=HORIZON_LEN)
    
    # TimesFM model (the underlying torch model)
    model = loader.model.model 
    model.to(device)

    # 1. Apply LoRA
    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], # Transformer blocks
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 2. Data Loader
    train_ds = SpreadDataset(hf_dataset)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    logger.info(f"Starting fine-tuning for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # TimesFM forward expects [batch, context_len]
            inputs = batch["input_values"].to(device)
            labels = batch["label_values"].to(device)
            
            # Note: This is a simplified loss calculation. 
            # TimesFM actually outputs mean + quantiles.
            # We target the mean forecast to match label_values.
            outputs = model(inputs) 
            
            # MSE Loss on the horizon
            loss = torch.nn.functional.mse_loss(outputs[:, -HORIZON_LEN:], labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.6f}")

    # 4. Save
    save_path = "./data/models/timesfm-finetuned"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    logger.info(f"Fine-tuned model saved to {save_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = prepare_spread_dataset()
    if len(ds) > 0:
        finetune_model(ds)
    else:
        logger.error("No training data generated.")
