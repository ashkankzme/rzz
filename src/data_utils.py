import math
import json
import random
import numpy as np
from torch.utils.data import Dataset
from mpmath import mp

# Import the zeros fetching function from zeros_db
from zeros_db import zeros_starting_at_N

# Vocabulary definition
VOCABULARY = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ':', 'b', 'e', ' ', 'p']
token2id = {ch: i for i, ch in enumerate(VOCABULARY)}
id2token = {i: ch for ch, i in token2id.items()}

CONTEXT_SIZE = 32


def format_rzz_sample(index, rzz_val):
    """
    Format a single zero entry to the string form: "b<i>:<rzz>e" padded with 'p' tokens
    where rzz_val is converted to a float with 4 decimals.
    """
    # Keep only the first 4 decimal places (rounding as needed)
    rzz_str = f"{float(rzz_val):.4f}"
    # Create the formatted string; note that index can be multiple digits.
    base_str = f"b{index}:{rzz_str}e"
    # Pad with 'p' until context length is reached
    pad_length = CONTEXT_SIZE - len(base_str)
    if pad_length < 0:
        # If the formatted string is too long, truncate (should not happen for our data)
        base_str = base_str[:CONTEXT_SIZE]
    else:
        base_str += "p" * pad_length
    return base_str


def tokenize(text):
    """Tokenize the text character by character using the defined vocabulary."""
    return [token2id[c] for c in text]


def detokenize(token_ids):
    """Convert list of token ids back to string."""
    return ''.join(id2token[i] for i in token_ids)


class RZZDataset(Dataset):
    """
    PyTorch Dataset for the Riemann Zeta Zeros.
    The dataset fetches zeros from the zeros_db interface,
    formats them and then creates training trajectories for next-token prediction.
    """

    def __init__(self, start_index, count, trajectory=True):
        """
        :param start_index: starting zero index (global index)
        :param count: number of zeros to fetch
        :param trajectory: if True, generate training trajectories (each step is one token prediction).
        """
        self.samples = []
        zeros = zeros_starting_at_N(start_index, count)
        for idx, mp_val in zeros:
            formatted = format_rzz_sample(idx, mp_val)
            # Generate trajectory training examples if needed:
            # Only predict tokens after the "b<i>:" portion.
            # Find index of ':' to know where the numeric part begins.
            colon_idx = formatted.find(':')
            # Input prefix always includes "b<i>:".
            prefix = formatted[:colon_idx + 1]
            # The remaining string is prediction target tokens (e.g., "49.7738e..." etc.)
            remainder = formatted[colon_idx + 1:]
            # For next-token prediction, we generate multiple (input, target) pairs.
            # We start predicting from the first token of remainder until the 'e' token is predicted.
            # Each training example is padded to context size.
            for i in range(1, len(remainder)):
                # Only generate trajectory up until the token before 'e'
                if remainder[i - 1] == 'e':
                    break
                input_seq = prefix + remainder[:i]
                target = remainder[i]
                # Pad input_seq if needed
                padded_input = input_seq + "p" * (CONTEXT_SIZE - len(input_seq))
                self.samples.append((tokenize(padded_input), token2id[target]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return token ids tensor and target token id.
        inp, target = self.samples[idx]
        return np.array(inp, dtype=np.int64), target


def get_train_val_datasets(t, K, val_fraction=0.01):
    """
    For a given time-step t, returns:
      - training dataset: zeros from t*1,000,000 to (t+1)*1,000,000 - 1
      - validation dataset: a fixed random subset (val_fraction) of zeros from the next million interval.
    """
    train_start = t * 1_000_000
    train_count = 1_000_000
    val_start = (t + 1) * 1_000_000
    val_count = int(1_000_000 * val_fraction)

    train_dataset = RZZDataset(train_start, train_count)
    val_dataset = RZZDataset(val_start, val_count)
    return train_dataset, val_dataset


def get_test_dataset(K, N):
    """
    Returns the test dataset.
    Test zeros are from the (K+1)-th million and only the first N zeros are used.
    """
    test_start = (K + 1) * 1_000_000
    test_dataset = RZZDataset(test_start, N)
    return test_dataset
