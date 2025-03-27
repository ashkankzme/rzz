import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from data_utils import get_test_dataset, detokenize, tokenize, CONTEXT_SIZE, token2id
from transformer import DecoderOnlyTransformer

# Load config
with open("config.json") as f:
    config = json.load(f)
K = config.get("K", 10)
N = config.get("N", 1000)

# Hyperparameters: they should match the best configuration used during training.
best_hparams = {
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1
}
d_ff = best_hparams["d_model"] * 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoderOnlyTransformer(best_hparams["d_model"],
                               best_hparams["num_heads"],
                               best_hparams["num_layers"],
                               d_ff,
                               best_hparams["dropout"]).to(device)

# Load the latest checkpoint
checkpoint_path = os.path.join("checkpoints", "latest.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

test_dataset = get_test_dataset(K, N)

def nucleus_sample(logits, p=0.9):
    """
    Sample from the logits using nucleus (top-p) sampling.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1)
    return next_token

def generate_sequence(prefix, max_length=CONTEXT_SIZE, p=0.9):
    """
    Given a starting prefix (string), generate tokens until the 'e' token is produced.
    """
    model_input = torch.tensor([tokenize(prefix)], dtype=torch.long).to(device)
    generated = prefix
    while True:
        # Ensure input is of size (B, seq_length) with proper padding if needed.
        if model_input.size(1) < CONTEXT_SIZE:
            pad_length = CONTEXT_SIZE - model_input.size(1)
            pad_tensor = torch.full((model_input.size(0), pad_length), token2id["p"], dtype=torch.long, device=device)
            model_input = torch.cat([model_input, pad_tensor], dim=1)
        logits = model(model_input)[:, model_input.size(1)-1, :]  # get logits for last token position
        next_token = nucleus_sample(logits, p=p)
        next_token_id = next_token.item()
        generated += detokenize([next_token_id])
        if detokenize([next_token_id]) == 'e' or len(generated) >= max_length:
            break
        # Append next token to the input sequence
        new_input = tokenize(detokenize([next_token_id]))
        model_input = torch.cat([model_input, torch.tensor(new_input, device=device).unsqueeze(0)], dim=1)
    return generated

def parse_rzz(formatted_str):
    """
    Given a formatted string like "b10:49.7738eppp...", extract the numeric part (as float).
    """
    try:
        # Find indices between ':' and 'e'
        colon_idx = formatted_str.find(':')
        e_idx = formatted_str.find('e')
        num_str = formatted_str[colon_idx+1:e_idx]
        return float(num_str)
    except Exception as ex:
        print("Error parsing rzz:", ex)
        return None

# Evaluate MSE over test set
mse_loss = 0.0
count = 0
for sample_idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
    # Get the original formatted sample (without trajectories) using data_utils formatting.
    # Reconstruct the original formatted string using the tokenized input from the first training example in the trajectory.
    # For simplicity, assume we can reformat using the same logic as in data_utils.
    # Here we fetch the test sample (we assume the first trajectory for each zero contains the full sequence).
    tokens, _ = test_dataset[sample_idx]
    formatted_str = "".join([detokenize([token]) for token in tokens]).replace("p", "")  # remove pad tokens for parsing
    # Determine the prefix: "b<i>:" as starting point.
    colon_idx = formatted_str.find(':')
    prefix = formatted_str[:colon_idx+1]
    generated = generate_sequence(prefix)
    predicted_val = parse_rzz(generated)
    # To get the ground truth value, parse the original full string (up to 'e')
    ground_truth = parse_rzz(formatted_str)
    if predicted_val is not None and ground_truth is not None:
        mse_loss += (predicted_val - ground_truth) ** 2
        count += 1

if count > 0:
    mse_loss /= count
else:
    mse_loss = float('inf')
print("Test MSE Loss:", mse_loss)
