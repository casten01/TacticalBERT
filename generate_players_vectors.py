import torch
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from statsbombpy import sb
from torch.utils.data import DataLoader

from src.config import Config
from src.model import TacticalBert
from src.dataset import FootballDataset, TacticalCollator

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = Config.BEST_MODEL_PATH
DATA_PATH = Config.PROCESSED_DATA_DIR / "dataset_tokenized.pt"
VOCAB_PATH = Config.PROCESSED_DATA_DIR / "vocab.json"
METADATA_PATH = Config.PROCESSED_DATA_DIR / "player_metadata.json"
MATCHES_METADATA_PATH = Config.RAW_DATA_DIR / "matches_metadata.csv"
OUTPUT_DB_PATH = Config.PROCESSED_DATA_DIR / "player_embeddings_db.parquet"

warnings.filterwarnings('ignore')

# --- UTILITY ---

def get_most_common(lst):
    """Return the most common element in a list."""
    if not lst: return None
    return Counter(lst).most_common(1)[0][0]

def build_player_metadata():
    """Download player metadata from StatsBomb API and build a mapping."""
    print("--- BUILDING PLAYER METADATA (StatsBomb API) ---")
    
    if not MATCHES_METADATA_PATH.exists():
        print(f"Error: {MATCHES_METADATA_PATH} does not exist.")
        return {}

    df_matches = pd.read_csv(MATCHES_METADATA_PATH)
    if 'match_id' not in df_matches.columns:
        print("Error: match_id column missing.")
        return {}
        
    match_ids = df_matches['match_id'].unique().tolist()
    print(f"Analaysing {len(match_ids)} matches to extract player metadata...")

    raw_storage = {}
    
    for match_id in tqdm(match_ids, desc="Fetching Lineups"):
        try:
            lineups = sb.lineups(match_id=match_id, fmt="dict")
            for team_id, team_data in lineups.items():
                team_name = team_data.get('team_name', 'Unknown')
                for player in team_data.get('lineup', []):
                    pid = int(player['player_id'])
                    pname = str(player['player_name'])
                    
                    current_positions = [p['position'] for p in player.get('positions', []) if 'position' in p]
                    
                    if pid not in raw_storage:
                        raw_storage[pid] = {'name': pname, 'teams': [], 'positions': []}
                    
                    raw_storage[pid]['teams'].append(team_name)
                    raw_storage[pid]['positions'].extend(current_positions)
                    
                    if len(pname) > len(raw_storage[pid]['name']):
                        raw_storage[pid]['name'] = pname
        except Exception:
            continue

    final_map = {}
    for pid, data in raw_storage.items():
        primary_pos = get_most_common(data['positions']) or "Unknown/Substitute"
        primary_team = get_most_common(data['teams'])
        
        final_map[pid] = {
            "player_name": data['name'],
            "position_name": primary_pos,
            "team_name": primary_team
        }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(final_map, f, indent=4)
    
    print(f"Saved: {len(final_map)} players .")
    return final_map

def load_metadata():
    """Load or build player metadata."""
    if METADATA_PATH.exists():
        print(f"Loading from {METADATA_PATH}...")
        with open(METADATA_PATH, 'r') as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}
    else:
        return build_player_metadata()

# --- EXTRACTING AND AGGREGATING ---

def extract_and_aggregate(model, dataloader):
    player_sums = defaultdict(lambda: np.zeros(Config.HIDDEN_SIZE))
    player_counts = defaultdict(int)
    
    print("Starting embedding extraction and aggregation...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):

            input_ids = batch['input_ids'].to(DEVICE)
            loc_ids = batch['loc_ids'].to(DEVICE)
            dur_ids = batch['duration_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            ctx = batch['context_features'].to(DEVICE)
            
            if 'meta' in batch and 'player_ids' in batch['meta']:
                batch_player_ids = batch['meta']['player_ids']
            elif 'player_ids' in batch:
                batch_player_ids = batch['player_ids']
            else:
                continue

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loc_ids=loc_ids,
                duration_ids=dur_ids,
                context_features=ctx
            )
            
            if hasattr(outputs, 'hidden_states'):
                last_hidden_state = outputs.hidden_states[-1]
            else:
                last_hidden_state = outputs[-1]

            hidden_states = last_hidden_state.cpu().numpy()
            
            batch_size = hidden_states.shape[0]
            model_seq_len = hidden_states.shape[1]

            for b in range(batch_size):
                players_in_seq = batch_player_ids[b]
                limit = min(len(players_in_seq), model_seq_len)

                for t in range(limit):
                    pid = players_in_seq[t]
                    
                    if hasattr(pid, 'item'): pid = pid.item()
                    
                    if pid == 0: continue 
                    
                    vector = hidden_states[b, t, :]
                    player_sums[pid] += vector
                    player_counts[pid] += 1
                    
    return player_sums, player_counts

# --- MAIN ---

def main():
    print(f"Working on device: {DEVICE}")
    
    player_meta = load_metadata()
    
    if not VOCAB_PATH.exists():
        raise FileNotFoundError("Vocab not found.")
        
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
        type_vocab_size = len(vocab['type_name'])

    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = TacticalBert(vocab_size=type_vocab_size).to(DEVICE)
    
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint {CHECKPOINT_PATH} not found.")
        
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    model.bert.config.output_hidden_states = True 
    print("Model loaded successfully.")

    print("Loading Dataset...")
    dataset = FootballDataset(DATA_PATH)
    collator = TacticalCollator(tokenizer_vocab=vocab['type_name'], max_len=Config.MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator, shuffle=False)
    
    sums, counts = extract_and_aggregate(model, dataloader)
    
    print("\nCreating DataFrame...")
    rows = []
    
    for pid, total_vector in sums.items():
        count = counts[pid]
        if count == 0:
            continue
        avg_vector = total_vector / count
        meta = player_meta.get(pid, {})
        
        rows.append({
            'player_id': pid,
            'player_name': meta.get('player_name', f"Unknown_{pid}"),
            'position': meta.get('position_name', 'Unknown'),
            'team': meta.get('team_name', 'Unknown'),
            'actions_count': count,
            'embedding': avg_vector
        })

    df = pd.DataFrame(rows)
    
    OUTPUT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(OUTPUT_DB_PATH)
    print(f"Database saved to: {OUTPUT_DB_PATH}")
    print(f"Total Players: {len(df)}")

if __name__ == "__main__":
    main()