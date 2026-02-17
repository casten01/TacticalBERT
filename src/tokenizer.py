import pandas as pd
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from .config import Config
from .utils import discretize_location, discretize_duration

class FootballTokenizer:
    def __init__(self):

        self.vocabs = {}
        
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
        }
        self.next_token_id = 5

    def fit(self, parquet_files):
        
        """Read all chunks and build vocabularies for categorical columns."""

        print("Building Vocabulary from raw files...")
        
        unique_values = {col: set() for col in Config.CATEGORICAL_COLS}

        for file_path in tqdm(parquet_files, desc="Scanning Vocab"):
            df = pd.read_parquet(file_path)

            for col in Config.CATEGORICAL_COLS:
                uniques = df[col].astype(str).unique()
                unique_values[col].update(uniques)

        # (String -> Int)
        for col, values in unique_values.items():

            sorted_values = sorted(list(values))
                        
            vocab_mapping = self.special_tokens.copy()
            current_id = self.next_token_id

            if col == 'type_name':
                vocab_mapping["Goal"] = 5

            for val in sorted_values:
                if val in vocab_mapping:
                    continue
                vocab_mapping[val] = current_id
                current_id += 1
            
            self.vocabs[col] = vocab_mapping
            
        print("Vocabulary built.")
        return self

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.vocabs, f, indent=4)
        print(f"Vocab saved to {path}")

    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.vocabs = json.load(f)
        print(f"Vocab loaded from {path}")

    def encode_match(self, df_match):
        """
        Transform a DataFrame of a match into a LIST of SEQUENCES (Possessions).
        Each sequence is a dictionary of lists of integers.
        """
        sequences = []
        
        grouped = df_match.groupby('possession')
        
        for possession_id, group in grouped:
            group = group.sort_values('index')
            attacking_team = group['is_possession_team'].iloc[0]
            
            dist = self.event_distance(group)
            
            is_carry = group['type_name'] == 'Carry'
            is_short = dist < Config.MIN_CARRY_DISTANCE
            
            mask_keep = ~(is_carry & is_short)
            group = group[mask_keep]
            
            if len(group) < 2:
                continue

            seq_data = {
                'match_id': group['match_id'].iloc[0],
                'possession_id': int(possession_id),

                # Categorical features (tokenized)
                'type_ids': [],
                'loc_ids': [],
                'duration_ids': [],
                'context_features': [],
                # Future extensions:
                'sub_type_ids': [],
                'outcome_ids': [],
                'body_part_ids': [],
                'technique_ids': [],
                'play_pattern_ids': [],
                
                # Meta info (not for training)
                'player_ids': [],
                'team_ids': []
            }

            for _, row in group.iterrows():
                type_name = str(row['type_name'])
                x = row['x']
                y = row['y']

                if row['is_possession_team'] != attacking_team:
                    x = Config.GRID_WIDTH - x
                    y = Config.GRID_HEIGHT - y
                
                for col in Config.CATEGORICAL_COLS:
                    val = str(row[col])
                    vocab = self.vocabs.get(col, {})

                    tid = vocab.get(val, vocab["[UNK]"])
                    
                    key = col.replace('_name', '_ids')
                    seq_data[key].append(tid)

                loc_id = discretize_location(x, y)
                seq_data['loc_ids'].append(loc_id)
                
                dur_id = discretize_duration(row['duration'])
                seq_data['duration_ids'].append(dur_id)

                # Context feature[pressure, counter, poss_team]
                ctx = [
                    int(row['under_pressure']),
                    int(row['counterpress']),
                    int(row['is_possession_team'])
                ]
                seq_data['context_features'].append(ctx)

                pid = row.get('player_id', 0)
                seq_data['player_ids'].append(int(pid) if pd.notna(pid) else 0)

                tid = row.get('team_id', 0)
                seq_data['team_ids'].append(int(tid) if pd.notna(tid) else 0)

            sequences.append(seq_data)
            
        return sequences

    def event_distance(self, group):
        next_x = group['x'].shift(-1)
        next_y = group['y'].shift(-1)
            
        dx = next_x - group['x']
        dy = next_y - group['y']
        dist = np.sqrt(dx**2 + dy**2)
        return dist