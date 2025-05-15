import os
import torch
from torch import nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerFast, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from NER_dataset import load_ner_dataset
from NER_config import *
from NER_trainer import train_model, eval_model 

def finetuning_NER():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "vinai/phobert-base-v2", use_fast=True
        )
    
    added_new_token = False
    if tokenizer.eos_token is None:
        print("Warning: eos_token is None. Adding a default one '</s>'.")
        tokenizer.add_special_tokens({'eos_token': '</s>'}) # phoBERT 
        added_new_token = True 

    if tokenizer.pad_token is None:
        # print(f"Setting pad_token to eos_token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print(f"Tokenizer already has pad_token: {tokenizer.pad_token}")

    # print(f"Tokenizer pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    # print(f"Tokenizer vocabulary size: {len(tokenizer)}")


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

    data_collator = DataCollatorForTokenClassification(
        tokenizer, label_pad_token_id=-100
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

    # Load model phobert base v2
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base-v2",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABELS,
        label2id=LABELS2ID
    )

    # print(f"Original model embedding size: {model.config.vocab_size}")

    if len(tokenizer) > model.config.vocab_size:
        # print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        # model.config.vocab_size = len(tokenizer) # Thường không cần thiết nếu dùng resize_token_embeddings
    elif added_new_token and len(tokenizer) == model.config.vocab_size :
        # print(f"Resizing model embeddings to {len(tokenizer)} (even if size seems unchanged after adding token).")
        model.resize_token_embeddings(len(tokenizer))


    if tokenizer.pad_token_id is not None:
         model.config.pad_token_id = tokenizer.pad_token_id
        #  print(f"Updated model config pad_token_id to: {model.config.pad_token_id}")

    tokenizer.save_pretrained(PATH_SAVE_TOKENIZER)

    model.to(device)
    # Fine-tune
    best_val_loss, best_val_acc = train_model(
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

    print(f"Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")

    model.load_state_dict(torch.load("./NER/out/checkpoints/checkpoint_best.pkl", map_location=torch.device(device))["model_state"])
    val_loss, avg_acc = eval_model(
        phobert_model = model,
        val_loader    = val_loader,
        loss_fn       = nn.CrossEntropyLoss(ignore_index=-100),
        device        = device,
        save_pred_dir = None,
        save_pred     = False
    )

    print(f"Final Eval -- Loss: {val_loss:.4f}, Acc: {avg_acc:.4f}")

if __name__ == "__main__":
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_PRED_DIR, exist_ok=True)
    finetuning_NER()