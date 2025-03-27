import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import RZZDataset, tokenize, CONTEXT_SIZE
from transformer import DecoderOnlyTransformer
import numpy as np

# Load configuration
with open("config.json") as f:
    config = json.load(f)

grid_config = config["grid_search"]

# Use a small dataset for tuning (first 10000 zeros)
tuning_dataset = RZZDataset(1000000, 100)
# For simplicity, we use 80% for training and 20% for validation
n = len(tuning_dataset)
train_set = torch.utils.data.Subset(tuning_dataset, list(range(0, int(0.8 * n))))
val_set = torch.utils.data.Subset(tuning_dataset, list(range(int(0.8 * n), n)))


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inp, target in dataloader:
            inp = inp.to(device)
            target = target.to(device)
            logits = model(inp)
            # We predict only the last token position in the input sequence.
            # Alternatively, you can compute loss for each position (here we assume one prediction per sample).
            # For simplicity, we pick the last non-pad position.
            # Here, we assume that the input length is variable.
            # In our design, each training example is one prediction step.
            logits = logits[:, -1, :]
            loss = criterion(logits, target)
            total_loss += loss.item() * inp.size(0)
    return total_loss / len(dataloader.dataset)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_config = None
best_loss = float('inf')

for num_epochs in grid_config["num_epochs"]:
    for lr in grid_config["learning_rate"]:
        for batch_size in grid_config["batch_size"]:
            for d_model in grid_config["d_model"]:
                for num_heads in grid_config["num_heads"]:
                    for num_layers in grid_config["num_layers"]:
                        for dropout in grid_config["dropout"]:
                            print(f"Testing config: epochs={num_epochs}, lr={lr}, batch_size={batch_size}, "
                                  f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, dropout={dropout}")
                            d_ff = d_model * 4  # typical FF size
                            model = DecoderOnlyTransformer(d_model, num_heads, num_layers, d_ff, dropout).to(device)
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            criterion = torch.nn.CrossEntropyLoss()

                            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
                            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

                            for epoch in range(num_epochs):
                                model.train()
                                epoch_loss = 0.0
                                for inp, target in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                                    inp = inp.to(device)
                                    target = target.to(device)
                                    optimizer.zero_grad()
                                    logits = model(inp)
                                    logits = logits[:, -1, :]
                                    loss = criterion(logits, target)
                                    loss.backward()
                                    optimizer.step()
                                    epoch_loss += loss.item() * inp.size(0)
                                print(f"Epoch {epoch + 1} training loss: {epoch_loss / len(train_set)}")

                            val_loss = evaluate(model, val_loader, criterion, device)
                            print(f"Validation loss: {val_loss}")
                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_config = {
                                    "num_epochs": num_epochs,
                                    "learning_rate": lr,
                                    "batch_size": batch_size,
                                    "d_model": d_model,
                                    "num_heads": num_heads,
                                    "num_layers": num_layers,
                                    "dropout": dropout
                                }
print("Best hyperparameters found:", best_config)
