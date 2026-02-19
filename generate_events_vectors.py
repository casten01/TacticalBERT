import torch
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.model import TacticalBert
from src.dataset import FootballDataset, TacticalCollator

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

def extract_individual_events(model, dataloader, target_event_name, vocab):
    """
    Extract individual events of a specific type from the dataloader using the model's hidden states.
    For each occurrence of the target event, calculates the start and end coordinates based on the location IDs and filters out events that are followed by special tokens. 
    It extracts player ID and context features.
    """
    type_vocab = vocab['type_name']
    target_id = type_vocab.get(target_event_name)
    
    # Special tokens to filter out
    special_tokens = {type_vocab.get(tk) for tk in ["[PAD]", "[MASK]", "[SEP]", "[CLS]", "[UNK]"] if tk in type_vocab}

    if target_id is None:
        raise ValueError(f"Event '{target_event_name}' not found in vocabulary.")

    events_data = []

    print(f"--- Extraction Started for: {target_event_name} ---")
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(DEVICE)
            loc_ids = batch['loc_ids'].to(DEVICE)
            dur_ids = batch['duration_ids'].to(DEVICE)
            ctx = batch['context_features'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids, mask, loc_ids=loc_ids, duration_ids=dur_ids, context_features=ctx)
            last_hidden_state = outputs.hidden_states[-1].cpu().numpy()
            
            matches = (input_ids == target_id).nonzero(as_tuple=False)
            
            for b_idx, t_idx in matches:
                b_idx, t_idx = b_idx.item(), t_idx.item()

                if t_idx + 1 >= input_ids.size(1):
                    continue
                
                next_token_id = input_ids[b_idx, t_idx + 1].item()
                if next_token_id in special_tokens:
                    continue 

                loc_start = loc_ids[b_idx, t_idx].item()
                start_x = (loc_start - 1) % Config.GRID_WIDTH
                start_y = (loc_start - 1) // Config.GRID_WIDTH

                loc_end = loc_ids[b_idx, t_idx + 1].item()
                end_x = (loc_end - 1) % Config.GRID_WIDTH
                end_y = (loc_end - 1) // Config.GRID_WIDTH

                player_id = None
                if 'meta' in batch and 'player_ids' in batch['meta']:
                    p_list = batch['meta']['player_ids'][b_idx]
                    meta_idx = t_idx - 1 
                    if 0 <= meta_idx < len(p_list):
                        player_id = p_list[meta_idx]
                        if hasattr(player_id, 'item'): player_id = player_id.item()

                events_data.append({
                    'start_x': start_x,
                    'start_y': start_y,
                    'end_x': end_x,
                    'end_y': end_y,
                    'progressive_dist': end_x - start_x,
                    'pressure': ctx[b_idx, t_idx, 0].item(),
                    'duration': dur_ids[b_idx, t_idx].item(),
                    'player_id': player_id,
                    'embedding': last_hidden_state[b_idx, t_idx, :].astype('float32')
                })

    return pd.DataFrame(events_data)

def main(event_type='Pass'):

    with open(Config.PROCESSED_DATA_DIR / "vocab.json", 'r') as f:
        vocab = json.load(f)

    full_dataset = FootballDataset(Config.PROCESSED_DATA_DIR / "dataset_tokenized.pt")
    
    subset_size = min(10000, len(full_dataset))
    indices = torch.arange(subset_size)
    subset_dataset = Subset(full_dataset, indices)
    
    collator = TacticalCollator(tokenizer_vocab=vocab['type_name'], max_len=Config.MAX_LEN)
    dataloader = DataLoader(subset_dataset, batch_size=32, shuffle=False, collate_fn=collator)

    model = TacticalBert(vocab_size=len(vocab['type_name'])).to(DEVICE)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=DEVICE))
    model.bert.config.output_hidden_states = True 

    df_events = extract_individual_events(model, dataloader, event_type, vocab)
    
    output_path = Config.PROCESSED_DATA_DIR / f"events_{event_type.lower()}_db.parquet"
    df_events.to_parquet(output_path)
    
    print(f"\n--- SUCCESS ---")
    print(f"File saved: {output_path}")
    print(f"Total events in the database: {len(df_events)}")

if __name__ == "__main__":
    #Change the event type as needed (e.g., 'Pass', 'Shot', 'Duel', etc.)
    main('Pass')