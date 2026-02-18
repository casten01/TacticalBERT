import torch
from torch.utils.data import Dataset
import random
import numpy as np

class FootballDataset(Dataset):
    def __init__(self, data_path, max_len=128):
        """
        Loads a .pt file containing a list of dictionaries.
        """
        print(f"Loading dataset from {data_path}...")
        self.data = torch.load(data_path, weights_only=False)
        self.max_len = max_len
        print(f"Dataset loaded. Total sequences: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        return item

class TacticalCollator:
    def __init__(self, tokenizer_vocab, max_len=128, mask_prob=0.15):
        self.pad_token_id = tokenizer_vocab["[PAD]"]
        self.mask_token_id = tokenizer_vocab["[MASK]"]
        self.cls_token_id = tokenizer_vocab["[CLS]"]
        self.sep_token_id = tokenizer_vocab["[SEP]"]
        self.vocab_len = len(tokenizer_vocab)
        self.max_len = max_len
        self.mask_prob = mask_prob
        

    def __call__(self, batch):
        
        batch_out = {
            'input_ids': [],         # masked type_ids
            'attention_mask': [],     
            'labels': [],  
            'loc_ids': [],
            'duration_ids': [],
            'context_features': [],

            # Meta info (not tensors, not for training)
            'player_ids': [],
            'match_id': [],
            'team_ids': [],
            'possession_id': []
        }

        for item in batch:
            
            max_seq = self.max_len - 2
            
            raw_types = item['type_ids'][:max_seq]
            raw_locs = item['loc_ids'][:max_seq]
            raw_dur = item['duration_ids'][:max_seq] if 'duration_ids' in item else [0]*len(raw_types)
            raw_ctx = item['context_features'][:max_seq] if 'context_features' in item else [[0,0,0]]*len(raw_types)

            input_ids = [self.cls_token_id] + raw_types + [self.sep_token_id]
            loc_ids = [0] + raw_locs + [0]
            dur_ids = [0] + raw_dur + [0]
            ctx_feats = [[0,0,0]] + raw_ctx + [[0,0,0]]

            labels = [-100] * len(input_ids)
            
            for i in range(1, len(input_ids) - 1):
                if random.random() < self.mask_prob:
                    labels[i] = input_ids[i]
                    
                    prob = random.random()
                    if prob < 0.8:
                        input_ids[i] = self.mask_token_id
                    elif prob < 0.9:
                        input_ids[i] = random.randint(5, self.vocab_len - 1)

            pad_len = self.max_len - len(input_ids)
            
            att_mask = [1] * len(input_ids) + [0] * pad_len
            
            input_ids += [self.pad_token_id] * pad_len
            labels += [-100] * pad_len
            loc_ids += [0] * pad_len
            dur_ids += [0] * pad_len
            ctx_feats += [[0,0,0]] * pad_len 

            batch_out['input_ids'].append(input_ids)
            batch_out['labels'].append(labels)
            batch_out['attention_mask'].append(att_mask)
            batch_out['loc_ids'].append(loc_ids)
            batch_out['duration_ids'].append(dur_ids)
            batch_out['context_features'].append(ctx_feats)
            
            batch_out['player_ids'].append(item.get('player_ids', []))
            batch_out['match_id'].append(item.get('match_id', 0))
            batch_out['team_ids'].append(item.get('team_ids', []))
            batch_out['possession_id'].append(item.get('possession_id', 0))

        return {
            'input_ids': torch.tensor(batch_out['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(batch_out['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(batch_out['labels'], dtype=torch.long),
            'loc_ids': torch.tensor(batch_out['loc_ids'], dtype=torch.long),
            'duration_ids': torch.tensor(batch_out['duration_ids'], dtype=torch.long),
            'context_features': torch.tensor(batch_out['context_features'], dtype=torch.float), # Float per proiezione lineare
            'meta': {
                'player_ids': batch_out['player_ids'],
                'match_id': batch_out['match_id'],
                'team_ids': batch_out['team_ids'],
                'possession_id': batch_out['possession_id']
            }
        }