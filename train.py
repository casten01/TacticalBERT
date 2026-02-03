import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import json
from pathlib import Path

from src.config import Config
from src.dataset import FootballDataset, TacticalCollator
from src.model import TacticalBert

def get_class_weights(vocab, device):
    """Assigns higher weight to rare/important actions. Used for weighted loss."""
    weights = torch.ones(len(vocab), device=device)
    
    high_value_actions = {
        'Goal': 100.0,
        'Bad Behaviour': 100.0,
        'Ball Recovery': 9.07,
        'Block': 24.6,
        'Carry': 2.1,
        'Clearance': 20.15,
        'Dispossessed': 37.05,
        'Dribble': 28.71,
        'Dribbled Past': 46.98,
        'Duel': 11.67,
        'Error': 100.0,
        'Foul Committed': 31.83,
        'Foul Won': 33.41,
        'Goal Keeper': 33.47,
        'Interception': 40.46,
        'Miscontrol': 32.8,
        'Offside': 100.0,
        'Own Goal Against': 100.0,
        'Own Goal For': 100.0,
        'Pressure': 2.96,
        'Shield': 100.0,
        'Shot': 38.67,
    }
    
    for action, weight in high_value_actions.items():
        if action in vocab:
            idx = vocab[action]
            weights[idx] = weight
            
    return weights

def train():
    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Vocab & Dataset
    vocab_path = Config.PROCESSED_DATA_DIR / "vocab.json"
    with open(vocab_path, 'r') as f:
        full_vocab = json.load(f)
    
    type_vocab = full_vocab['type_name'] 
    
    dataset_path = Config.PROCESSED_DATA_DIR / "dataset_tokenized.pt"
    dataset = FootballDataset(dataset_path)

    # Collator & DataLoader
    collator = TacticalCollator(tokenizer_vocab=type_vocab, max_len=Config.MAX_LEN)
    
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collator)

    # Model
    class_weights = get_class_weights(type_vocab, device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = TacticalBert(vocab_size=len(type_vocab)).to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # --- SANITY CHECK MODE ---
    
    SANITY_CHECK = False

    epochs = 100 if SANITY_CHECK else 5
    print(f"Starting training for {epochs} epochs. Sanity Check: {SANITY_CHECK}")

    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loc_ids = batch['loc_ids'].to(device)
            duration_ids = batch['duration_ids'].to(device)
            context_features = batch['context_features'].to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                #labels=labels,
                loc_ids=loc_ids,
                duration_ids=duration_ids,
                context_features=context_features
            )
            
            logits = outputs.logits.view(-1, len(type_vocab))
            targets = labels.view(-1)
            
            loss = loss_fct(logits, targets)
            
            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            if SANITY_CHECK:
                break
        
        avg_loss = total_loss / (1 if SANITY_CHECK else len(train_loader))
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        if not SANITY_CHECK:
            save_path = Config.BASE_DIR / "checkpoints"
            save_path.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path / f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()