import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 16  # as defined in the vocabulary
CONTEXT_SIZE = 32


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, d_model = x.size()
        # Linear projections
        q = self.q_linear(x)  # (B, seq, d_model)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Split into heads
        def split_heads(tensor):
            return tensor.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # (B, heads, seq, d_k)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, heads, seq, seq)
        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B, heads, seq, d_k)
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        output = self.out(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed forward with residual
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.position_embedding = nn.Embedding(CONTEXT_SIZE, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x):
        """
        :param x: (B, seq_len) tensor of token ids
        """
        batch_size, seq_len = x.size()
        token_emb = self.token_embedding(x)  # (B, seq_len, d_model)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, d_model)
        x = token_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        logits = self.fc_out(x)  # (B, seq_len, VOCAB_SIZE)
        return logits


if __name__ == '__main__':
    # Quick test
    model = DecoderOnlyTransformer(d_model=64, num_heads=4, num_layers=2, d_ff=128)
    sample_input = torch.randint(0, VOCAB_SIZE, (2, CONTEXT_SIZE))
    logits = model(sample_input)
    print("Output shape:", logits.shape)
