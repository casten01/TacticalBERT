import pandas as pd
import numpy as np
from mplsoccer import Sbopen
from tqdm import tqdm
from .config import Config

class DataExtractor:
    def __init__(self):
        self.parser = Sbopen()

    def get_all_match_ids(self):
        """Get all match IDs for the configured competitions and season"""
        match_ids = []
        print(f"Fetching matches for Season {Config.SEASON_ID}...")
        
        for comp_id in Config.COMPETITION_IDS:
            try:
                df_matches = self.parser.match(competition_id=comp_id, season_id=Config.SEASON_ID)
                ids = df_matches['match_id'].tolist()
                match_ids.extend(ids)
                print(f" -> Competition {comp_id}: {len(ids)} matches found.")
            except Exception as e:
                print(f"Error fetching competition {comp_id}: {e}")
                
        return match_ids

    def process_match(self, match_id):
        """
        Download, clean, and format events for a single match.
        Returns a DataFrame ready to be saved.
        """
        try:
            events, _, _, _ = self.parser.event(match_id)
        except Exception as e:
            print(f"Skipping match {match_id}: {e}")
            return None

        # Custom column processing
        events['is_possession_team'] = (events['team_id'] == events['possession_team_id'])

        if 'type_name' in events.columns:
            events = events[~events['type_name'].isin(Config.IGNORED_EVENTS)].copy()

        if 'shot_outcome_name' in events.columns:
            is_goal = (events['type_name'] == 'Shot') & (events['outcome_name'] == 'Goal')
            events.loc[is_goal, 'type_name'] = 'Goal'

        cols_to_keep = (
            Config.CATEGORICAL_COLS + 
            Config.BOOLEAN_COLS + 
            Config.NUMERICAL_COLS + 
            Config.META_COLS
        )
        
        existing_cols = [c for c in cols_to_keep if c in events.columns]
        df_clean = events[existing_cols].copy()
        
        # Booleans: NaN -> False
        for col in Config.BOOLEAN_COLS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(False).astype(int)

        if 'duration' in df_clean.columns:
            df_clean['duration'] = df_clean['duration'].fillna(0.0)

        # Categorical: NaN -> "[NONE]" or handled by tokenizer after
        for col in Config.CATEGORICAL_COLS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("NaN")

        return df_clean
    
    def save_matches_metadata(self):
        """. Saves a CSV mapping match_id -> competition_id."""
        Config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        all_meta = []
        print("Saving metadata index...")
        for comp_id in Config.COMPETITION_IDS:
            try:
                df = self.parser.match(competition_id=comp_id, season_id=Config.SEASON_ID)
                df_small = df[['match_id', 'competition_id', 'match_date', 'home_team_name', 'away_team_name']].copy()
                all_meta.append(df_small)
            except:
                pass
        
        if all_meta:
            full_meta = pd.concat(all_meta)
            full_meta.to_csv(Config.RAW_DATA_DIR / "matches_metadata.csv", index=False)
            print("Metadata index saved.")

    def run_extraction_pipeline(self):
        """Executes the full extraction pipeline and saves to Parquet."""
        all_matches = self.get_all_match_ids()
        print(f"Total matches to process: {len(all_matches)}")
        
        buffer = []
        chunk_size = 100 
        chunk_counter = 0

        for m_id in tqdm(all_matches, desc="Extracting Data"):
            df_match = self.process_match(m_id)
            if df_match is not None:
                buffer.append(df_match)
            
            if len(buffer) >= chunk_size:
                self._save_chunk(buffer, chunk_counter)
                buffer = []
                chunk_counter += 1
        
        if buffer:
            self._save_chunk(buffer, chunk_counter)
        
        self.save_matches_metadata()
            
    def _save_chunk(self, df_list, chunk_id):
        Config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not df_list: return
        full_df = pd.concat(df_list, ignore_index=True)
        filename = Config.RAW_DATA_DIR / f"events_chunk_{chunk_id}.parquet"
        full_df.to_parquet(filename, index=False)