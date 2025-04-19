
import os
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader
from NER_dataset import load_ner_dataset
from NER_config import *
from NER_trainer import train_model, eval_model 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=True)

    # Load datasets
    train_dataset = load_ner_dataset(
        data_path=PATH_TRAIN,
        tokenizer=tokenizer,
        label2id=LABELS2ID,
        max_length=MAX_LENGTH
    )
    val_dataset = load_ner_dataset(
        data_path=PATH_VAL,
        tokenizer=tokenizer,
        label2id=LABELS2ID,
        max_length=MAX_LENGTH
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # Load model phobert base v2
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base-v2",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABELS,
        label2id=LABELS2ID
    )

    # Fine-tune
    best_val_loss, best_val_acc, std_acc = train_model(
        phobert_model    = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        epochs           = NUM_TRAIN_EPOCHS,
        learning_rate    = LEARNING_RATE,
        run_dir          = RUN_DIR,
        log_dir          = LOG_DIR,
        save_epochfreq   = SAVE_EPOCH_FREQ,
        save_pred_epoch  = SAVE_EPOCH_FREQ,
        save_pred_dir    = SAVE_PRED_DIR,
        lr_step_size     = LR_STEP_SIZE,
        lr_gamma         = LR_GAMMA,
        device           = device
    )

    print(f"Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f} ± {std_acc:.4f}")

    val_loss, avg_acc, std_acc = eval_model(
        phobert_model = model,
        val_loader    = val_loader,
        criterion     = None,      
        device        = device,
        save_pred_dir = None,
        save_pred     = False
    )

    print(f"Final Eval -- Loss: {val_loss:.4f}, Acc: {avg_acc:.4f} ± {std_acc:.4f}")

if __name__ == "__main__":
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_PRED_DIR, exist_ok=True)
    main()