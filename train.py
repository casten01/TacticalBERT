import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
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

def train_one_epoch(model, dataloader, optimizer, loss_fct, device, type_vocab, desc):
    """Performs a single training epoch and returns the average loss."""
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc=desc)
    
    for batch in loop:
        # Remove meta info before passing to the model
        _ = batch.pop('meta')
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        
        logits = outputs.logits.view(-1, len(type_vocab))
        targets = batch['labels'].view(-1)
        
        loss = loss_fct(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fct, device, type_vocab, desc):
    """Evaluates the model and returns loss, F1, accuracy, targets, and preds."""
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    loop = tqdm(dataloader, desc=desc)
    
    with torch.no_grad():
        for batch in loop:
            # Remove meta info before passing to the model
            _ = batch.pop('meta')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits.view(-1, len(type_vocab))
            targets = batch['labels'].view(-1)
            
            loss = loss_fct(logits, targets)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            valid_mask = targets != -100
            
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_targets.extend(targets[valid_mask].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    
    # Calculate F1 and Accuracy
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_targets, all_preds)
    
    return avg_loss, f1, acc, all_targets, all_preds

def run_validation_phase(dataset, type_vocab, device, class_weights, save_dir):
    """Finds the best epoch using an 80/20 split and early stopping."""
    print("\n" + "="*50 + "\nTRAINING WITH VALIDATION\n" + "="*50)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    collator = TacticalCollator(tokenizer_vocab=type_vocab, max_len=Config.MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collator)

    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    model = TacticalBert(vocab_size=len(type_vocab)).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    epochs = 100
    patience = 15
    patience_counter = 0
    best_val_f1 = 0.0
    best_epoch = 1

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fct, device, type_vocab, f"Epoch {epoch+1} [TRAIN]"
        )
        
        val_loss, val_f1, val_acc, all_targets, all_preds = evaluate(
            model, val_loader, loss_fct, device, type_vocab, f"Epoch {epoch+1} [VALID]"
        )

        print(f"\n[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            print(f" -> New Best F1: saving model at epoch {best_epoch}...")
            torch.save(model.state_dict(), save_dir / "model_best.pth")
            
            max_id = max(type_vocab.values())
            target_names = [f"UNUSED_ID_{i}" for i in range(max_id + 1)]
            for name, idx in type_vocab.items():
                target_names[idx] = name
                
            report = classification_report(
                all_targets, all_preds, 
                target_names=target_names, 
                labels=range(max_id + 1), 
                zero_division=0
            )
            
            print("\n--- CLASSIFICATION REPORT ---")
            print(report)
            print("-" * 30 + "\n")
            
            with open(save_dir / "best_report.txt", "w") as f:
                f.write(f"BEST EPOCH: {best_epoch}\n")
                f.write(f"VALIDATION F1 (MACRO): {best_val_f1:.4f}\n")
                f.write(f"VALIDATION ACCURACY: {val_acc:.4f}\n\n")
                f.write(report)
            
        else:
            patience_counter += 1
            print(f" -> No improvement ({patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print(f"\nEARLY STOPPING triggered at epoch {epoch+1}.")
            break

    
    state = {"best_epoch": best_epoch, "best_val_f1": best_val_f1}
    with open(save_dir / "training_state.json", 'w') as f:
        json.dump(state, f)
        
    return best_epoch

def run_production_phase(dataset, type_vocab, device, class_weights, target_epochs, save_dir):
    """Retrains on the 100% full dataset up to the target epochs."""
    print("\n" + "="*50 + f"\nFULL RETRAINING (Target: {target_epochs} epochs)\n" + "="*50)
    
    collator = TacticalCollator(tokenizer_vocab=type_vocab, max_len=Config.MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collator)

    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    model = TacticalBert(vocab_size=len(type_vocab)).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    for epoch in range(target_epochs):
        train_one_epoch(
            model, train_loader, optimizer, loss_fct, device, type_vocab, f"Epoch {epoch+1}/{target_epochs} [FULL TRAIN]"
        )

    print("\nProduction training complete: saving model...")
    torch.save(model.state_dict(), save_dir / "model_production.pth")
    print(f"Model successfully saved to {save_dir / 'model_production.pth'}")

def main():
    checkpoints_dir = Config.BASE_DIR / "models"
    checkpoints_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load vocab and dataset
    vocab_path = Config.PROCESSED_DATA_DIR / "vocab.json"
    with open(vocab_path, 'r') as f:
        type_vocab = json.load(f)['type_name'] 
    
    dataset_path = Config.PROCESSED_DATA_DIR / "dataset_tokenized.pt"
    dataset = FootballDataset(dataset_path)
    
    class_weights = get_class_weights(type_vocab, device)

    state_file = checkpoints_dir / "training_state.json"
    prod_model_file = checkpoints_dir / "model_production.pth"

    # STATE LOGIC ORCHESTRATOR
    if prod_model_file.exists():
        print("\n[OK] Pipeline already completed. Production model is ready.")
        print("To restart from scratch, manually delete the 'checkpoints' directory.")
        return

    if not state_file.exists():
        # Validation missing, run 80/20 split
        best_epoch = run_validation_phase(dataset, type_vocab, device, class_weights, checkpoints_dir)
    else:
        # Validation done, read best epoch from JSON
        with open(state_file, 'r') as f:
            best_epoch = json.load(f)['best_epoch']
        print(f"\n[INFO] Pre-existing validation found. Best Epoch recorded: {best_epoch}")

    # Run full production retraining
    run_production_phase(dataset, type_vocab, device, class_weights, target_epochs=best_epoch, save_dir=checkpoints_dir)

if __name__ == "__main__":
    main()