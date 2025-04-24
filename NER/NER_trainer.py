import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn

def train_model(model,
                train_loader,
                val_loader,
                epochs,
                learning_rate,
                run_dir,
                log_dir,
                save_epochfreq,
                save_pred_epoch,
                save_pred_dir,
                lr_step_size,
                lr_gamma,
                device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # Loss function: bỏ qua label = -100 (padding/subword)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100) 

    best_val_loss = float("inf")

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    with open(log_file, "w") as f:
        f.write("Epoch, Train Loss, Val Loss, Val Acc\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]"):
            # batch: input_ids, attention_mask, labels
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels)
            loss = outputs.loss  # HuggingFace trả về loss khi truyền labels
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()

        save_preds = (epoch % save_pred_epoch == 0)
        val_loss, val_acc = eval_model(
            model, val_loader, loss_fn, device,
            save_pred_dir if save_preds else None, save_preds
        )

        os.makedirs(run_dir, exist_ok=True)
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch
        }
        if epoch % save_epochfreq == 0:
            torch.save(state, os.path.join(run_dir, f"checkpoint_epoch{epoch}.pkl"))
       
        torch.save(state, os.path.join(run_dir, "checkpoint_latest.pkl"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(state, os.path.join(run_dir, "checkpoint_best.pkl"))
            print(f"[Epoch {epoch}] Saved new best model (ValLoss={val_loss:.4f})")

        
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        print(f"[Epoch {epoch}/{epochs}] TrainLoss={avg_train_loss:.4f} | "
              f"ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f}")

    return best_val_loss, best_val_acc


def eval_model(model, val_loader, loss_fn, device,
               save_pred_dir=None, save_pred=False):

    model.eval()
    total_loss = 0.0
    acc_all = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels)
            loss = outputs.loss
            logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)

            total_loss += loss.item()

            # Tính token-level accuracy cho mỗi sample
            preds = torch.argmax(logits, dim=-1)
            batch_acc = []
            for p, l in zip(preds, labels):
                mask = (l != -100)
                correct = (p[mask] == l[mask]).sum().item()
                total = mask.sum().item()
                batch_acc.append(correct / total if total > 0 else 0.0)
            acc_all.append(np.array(batch_acc))

    avg_loss = total_loss / len(val_loader)
    acc_all = np.concatenate(acc_all)
    val_acc = float(np.mean(acc_all))

    if save_pred and save_pred_dir is not None:
        os.makedirs(save_pred_dir, exist_ok=True)
        np.save(os.path.join(save_pred_dir, "val_rewards.npy"), val_acc)

    return avg_loss, val_acc
