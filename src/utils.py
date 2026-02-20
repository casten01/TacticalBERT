import numpy as np
import pandas as pd
from .config import Config
import torch
import json
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path

def discretize_location(x, y):
    """
    Converts continuous coordinates (0-120, 0-80) in a unique Token ID.
    Row-Major flattening.
    """
    if pd.isna(x) or pd.isna(y):
        return 0 # Token [PAD] / [UNK]
    
    x = np.clip(x, 0, Config.GRID_WIDTH - 1e-6)
    y = np.clip(y, 0, Config.GRID_HEIGHT - 1e-6)
    
    x_int = int(x)
    y_int = int(y)
    
    token_id = (y_int * Config.GRID_WIDTH) + x_int + 1
    return token_id

def discretize_duration(seconds):
    """
    Logaritmic Binning for the duration of an event.
    """
    if pd.isna(seconds):
        return 0 # [NONE]
    
    if seconds < 0.5: return 1
    if seconds < 1.0: return 2  
    if seconds < 1.5: return 3  
    if seconds < 2.0: return 4  
    if seconds < 3.0: return 5  
    if seconds < 4.0: return 6 
    if seconds < 6.0: return 7  
    if seconds < 10.0: return 8 
    return 9                    

def calculate_and_print_weights():
    print(f"Loading data from {Config.PROCESSED_DATA_DIR / 'dataset_tokenized.pt'}...")
    try:
        dataset = torch.load(Config.PROCESSED_DATA_DIR / "dataset_tokenized.pt", weights_only=False)
        
        with open(Config.PROCESSED_DATA_DIR / "vocab.json", 'r') as f:
            full_vocab = json.load(f)
        type_vocab = full_vocab['type_name']
        
    except FileNotFoundError as e:
        print(f"Error: files not found.\n{e}")
        return

    print("Counting frequencies...")
    counts = Counter()
    
    for item in tqdm(dataset, desc="Scanning"):
        counts.update(item['type_ids'])

    max_count = max(counts.values())
    total_samples = sum(counts.values())
    
    print(f"\nTotal Events: {total_samples}")
    print(f"Max Frequency (Class '{[k for k,v in type_vocab.items() if v==counts.most_common(1)[0][0]][0]}'): {max_count}")

    weight_dict = {}
    
    print("\n--- WEIGHTS ---")
    print(f"{'ACTION':<20} | {'COUNT':<10} | {'WEIGHT':<10}")
    print("-" * 45)

    for name, idx in type_vocab.items():
        if idx in counts:
            count = counts[idx]
            # Formula Inverse Frequency: Max / Count
            w = max_count / (count + 1e-6)
            
            # Clamping to 100.0 for stability
            w = min(w, 100.0)
            w = round(w, 2)
            weight_dict[name] = w
            
            if count > 0:
                print(f"{name:<20} | {count:<10} | {w:<10}")
        else:
            weight_dict[name] = 1.0

    print("\n\n" + "="*50)
    print("="*50)
    print("MANUAL_WEIGHTS = {")
    for k, v in weight_dict.items():
        if v > 1.0:
            print(f"    '{k}': {v},")
    print("}")
    print("="*50)