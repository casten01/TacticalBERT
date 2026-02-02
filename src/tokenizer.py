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

            if 'shot_outcome_name' in df.columns:
                mask_goal = (df['type_name'] == 'Shot') & (df['shot_outcome_name'] == 'Goal')
                df.loc[mask_goal, 'type_name'] = 'Goal'

            if 'type_name' in df.columns:
                df = df[~df['type_name'].isin(Config.IGNORED_EVENTS)]

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
            if 'type_name' in group.columns:
                group = group[~group['type_name'].isin(Config.IGNORED_EVENTS)].copy()
            
            next_x = group['x'].shift(-1)
            next_y = group['y'].shift(-1)
            
            dx = next_x - group['x']
            dy = next_y - group['y']
            dist = np.sqrt(dx**2 + dy**2)
            
            is_carry = group['type_name'] == 'Carry'
            is_short = dist < Config.MIN_CARRY_DISTANCE
            

            mask_keep = ~(is_carry & is_short)
            group = group[mask_keep]
            
            if len(group) < 2:
                continue

            if 'shot_outcome_name' in group.columns:
                            is_goal = (group['type_name'] == 'Shot') & (group['shot_outcome_name'] == 'Goal')
                            group.loc[is_goal, 'type_name'] = 'Goal'

            seq_data = {
                'match_id': group['match_id'].iloc[0],
                'possession_id': int(possession_id),
                'type_ids': [],
                'sub_type_ids': [],
                'outcome_ids': [],
                'body_part_ids': [],
                'technique_ids': [],
                'play_pattern_ids': [],
                'duration_ids': [],
                'loc_ids': [],
                'end_loc_ids': [],
                'context_features': [],

                'player_ids': [],
                'team_ids': [],
                'event_ids': [],
                'timestamps': []
            }

            for _, row in group.iterrows():
                type_name = str(row['type_name'])
                
                for col in Config.CATEGORICAL_COLS:
                    val = str(row[col])
                    vocab = self.vocabs.get(col, {})

                    tid = vocab.get(val, vocab["[UNK]"])
                    
                    key = col.replace('_name', '_ids')
                    seq_data[key].append(tid)

                loc_id = discretize_location(row['x'], row['y'])
                end_loc_id = discretize_location(row['end_x'], row['end_y'])
                seq_data['loc_ids'].append(loc_id)
                seq_data['end_loc_ids'].append(end_loc_id)
                
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
                
                seq_data['event_ids'].append(str(row.get('id', '')))
                
                seq_data['timestamps'].append(str(row.get('timestamp', '')))

            sequences.append(seq_data)
            
        return sequences