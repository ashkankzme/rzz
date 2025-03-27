import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import get_train_val_datasets, CONTEXT_SIZE
from transformer import DecoderOnlyTransformer
import numpy as np


def save_checkpoint(_model, _optimizer, interval, step, checkpoint_path):
    torch.save({
        "model_state": _model.state_dict(),
        "optimizer_state": _optimizer.state_dict(),
        "current_interval": interval,
        "global_step": step
    }, checkpoint_path)


def xavier_model_param_initialization(_model):
    # Xavier Initialization for model parameters to get uniform distribution of loss in initial training
    for param in _model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param, gain=0.1)  # Reduce weight scale


if __name__ == "__main__":
    # Load config and best hyperparameters
    with open("config.json") as f:
        config = json.load(f)

    K = config.get("K", 10)
    checkpoint_interval = config.get("checkpoint_update_interval", 1000)
    # Load hyperparameters from config file w/ defaults:
    best_hparams = config.get("best_hparams", {
        "num_epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1
    })
    best_hparams["d_ff"] = best_hparams["d_model"] * 4

    # Paths for checkpoints and logs
    CHECKPOINT_DIR = os.path.join("../checkpoints")
    LOG_DIR = os.path.join("../logs")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(best_hparams["d_model"],
                                   best_hparams["num_heads"],
                                   best_hparams["num_layers"],
                                   best_hparams["d_ff"],
                                   best_hparams["dropout"]).to(device)

    xavier_model_param_initialization(model)
    optimizer = optim.Adam(model.parameters(), lr=best_hparams["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    # State for resuming training
    start_interval = 0
    global_step = 0
    resume_checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if os.path.exists(resume_checkpoint):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_interval = checkpoint["current_interval"]
        global_step = checkpoint["global_step"]

    log_file = open(os.path.join(LOG_DIR, "train_log.txt"), "a")
    log_file.write("-" * 50 + "\n")
    log_file.write(f"starting training at interval {start_interval}\n")
    log_file.flush()

    for t in range(start_interval, K):
        print(f"Training on interval {t}")
        train_dataset, val_dataset = get_train_val_datasets(t, K)
        train_loader = DataLoader(train_dataset, batch_size=best_hparams["batch_size"], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=best_hparams["batch_size"], shuffle=False)
        best_val_loss = float("inf")
        best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_of_{t + 1}-th_millions.pt")

        for epoch in range(best_hparams["num_epochs"]):
            model.train()
            for inp, target in tqdm(train_loader, desc=f"Interval {t} Epoch {epoch + 1}"):
                inp = torch.tensor(inp, dtype=torch.long).to(device)
                target = torch.tensor(target, dtype=torch.long).to(device)
                optimizer.zero_grad()
                logits = model(inp)
                # use the last token's prediction (each sample is one prediction step)
                logits = logits[:, -1, :]
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

                global_step += 1
                if global_step % checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, t, global_step, resume_checkpoint)
                    log_file.write(f"Interval {t} Step {global_step} training loss: {loss.item()}\n")
                    log_file.flush()

                    # Evaluate on validation set and update best model in round if better loss is found
                    model.eval() # model weights frozen for validation set run
                    val_loss = 0.0
                    count = 0
                    with torch.no_grad():
                        for inp, target in val_loader:
                            inp = torch.tensor(inp, dtype=torch.long).to(device)
                            target = torch.tensor(target, dtype=torch.long).to(device)
                            logits = model(inp)
                            logits = logits[:, -1, :]
                            loss = criterion(logits, target)
                            val_loss += loss.item() * inp.size(0)
                            count += inp.size(0)
                    avg_val_loss = val_loss / count
                    log_file.write(f"Interval {t} Epoch {epoch + 1} validation loss: {avg_val_loss}\n")
                    log_file.flush()

                    model.train() # back to train mode

                    if avg_val_loss < best_val_loss:
                        # Save the best model out of current million interval
                        best_val_loss = avg_val_loss
                        save_checkpoint(model, optimizer, t + 1, global_step, best_checkpoint_path)
                        log_file.write(f"Updated best model for interval {t+1}, w/ news avg val loss={best_val_loss}\n")
                        log_file.flush()

    log_file.close()
    print("Training complete.")
