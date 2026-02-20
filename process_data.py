import torch
import glob
import pandas as pd
from src.config import Config
from src.tokenizer import FootballTokenizer
from tqdm import tqdm

def main():
    print("TOKENIZATION & PROCESSING")
    Config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(Config.RAW_DATA_DIR.glob("*.parquet")))
    if not files:
        print("No parquet files found! Run step 1 first.")
        return

    tokenizer = FootballTokenizer()
    vocab_path = Config.PROCESSED_DATA_DIR / "vocab.json"

    if vocab_path.exists():
        print("Vocab file found, loading...")
        tokenizer.load_vocab(vocab_path)
    else:
        print("Vocab file not found, building from scratch...")
        tokenizer.fit(files)
        tokenizer.save_vocab(vocab_path)

    all_sequences = []
    
    print("Transforming data into sequences...")
    for f in tqdm(files, desc="Processing chunks"):
        df = pd.read_parquet(f)
        
        for match_id, match_df in df.groupby('match_id'):
            seqs = tokenizer.encode_match(match_df)
            all_sequences.extend(seqs)

    print(f"Total sequences (possessions) created: {len(all_sequences)}")
    
    output_path = Config.PROCESSED_DATA_DIR / "dataset_tokenized.pt"
    print(f"Saving dataset to {output_path}...")
    torch.save(all_sequences, output_path)
    
    print("--- STEP 2 COMPLETED ---")
    print(f"Vocab saved at: {vocab_path}")
    print(f"Dataset saved at: {output_path}")

if __name__ == "__main__":
    main()