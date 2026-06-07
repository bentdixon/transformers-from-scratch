import torch.nn as nn
import torch
import torch.nn.functional as F

import warnings
import math


warnings.simplefilter("ignore")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: (batch, seq_len) of token IDs
        return self.embed(x) * math.sqrt(self.d_model)  # A scaling factor used to multiply embeddings so magnitude is comparable to positional encoding
        # returns (batch, seq_len, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        # position: (max_seq_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        pe = pe.unsqueeze(0)            # Adds a dimension at index 0 -> (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)  # Not a trainable parameter

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Size of each head
        # One big projection each, split into head afterwards
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # q, k v: (batch, seq_len, d_model)
        batch = q.size(0)

        # 1. Project, then reshape into heads -> (batch, n_heads, seq_len, d_k)
        q = self.w_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled dot-product scores -> (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        # What is being meaningfully multiplied is (seq_len, seq_len) @ (seq_len, d_k)
        context = torch.matmul(attn, v)  # (batch, n_heads, seq_len, d_k)

        # 3. Concat heads -> (batch, seq_len, d_model), then final projection
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        # This computes: output = x @ W^T + b
        return self.w_o(context)  # Output projection matrix
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
    

# An encoder layer is comprised of multi-head attention and a FFN, plus residual connection and layer normalization
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # The Q, K, and V begin identically (each are built from the same representation)
        attn_out = self.attn(x, x, x, mask)  
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(max_len, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        x = self.dropout(self.pos(self.embed(src)))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x  # (batch, src_len, d_model)
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 1. masked self-attention (tgt_mask = look-ahead mask)
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # 2. cross-attention: Q=x (decoder), K=V=enc_out (encoder)
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        # 3. feed-forward
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.pos   = PositionalEncoding(max_len, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.dropout(self.pos(self.embed(tgt)))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_layers=6, n_heads=8,
                 d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx  # Store PAD so the mask helpers can use it
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)  # The prediction about which word comes next

    def make_src_mask(self, src):
        # (batch, 1, 1, src_len): True where token is NOT padding
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        batch, tgt_len = tgt.shape
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # lower-triangular: position i may attend to j only if j <= i
        look_ahead = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).bool()
        return pad_mask & look_ahead
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.out(dec_out)  # (batch, tgt_len, tgt_vocab)


def noam_lambda(step, d_model, warmup):
    # lr ramps up linearly for `warmup` steps, then decays as 1/sqrt(step)
    step = max(step, 1)  # avoid step**-0.5 at step 0
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


def make_copy_batch(batch, seq_len, vocab, pad_idx, bos_idx, device):
    # source = random tokens; target = same sequence (copy task)
    data = torch.randint(3, vocab, (batch, seq_len - 1), device=device)
    src = data
    bos = torch.full((batch, 1), bos_idx, device=device)
    tgt = torch.cat([bos, data], dim=1)  # decoder input: <bos> then the sequence
    # label = what each position should predict (target shifted left by one)
    label = torch.cat([data, torch.full((batch, 1), pad_idx, device=device)], dim=1)
    return src, tgt, label


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    PAD, BOS = 0, 1                       # PAD = padding token ID, BOS = beginning-of-sequence
    VOCAB, SEQ, D_MODEL = 50, 10, 64      # small settings so this runs fast as a smoke test

    model = Transformer(VOCAB, VOCAB, d_model=D_MODEL, n_layers=6,
                        n_heads=4, d_ff=256, max_len=64, pad_idx=PAD).to(device)

    # Cross-entropy with label smoothing; ignore_index skips PAD positions in the loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    # base lr is 1.0 because LambdaLR multiplies it by the Noam schedule value
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: noam_lambda(s, D_MODEL, warmup=400)
    )

    model.train()
    for step in range(1, 4000):
        src, tgt, label = make_copy_batch(64, SEQ, VOCAB, PAD, BOS, device)
        logits = model(src, tgt)                          # (batch, tgt_len, vocab)
        # flatten so each position is one classification example
        loss = criterion(logits.reshape(-1, VOCAB), label.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip exploding grads
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    # greedy decoding: generate one token at a time, only the last position matters each step
    model.eval()
    src, _, _ = make_copy_batch(1, SEQ, VOCAB, PAD, BOS, device)
    with torch.no_grad():
        ys = torch.full((1, 1), BOS, device=device)
        for _ in range(SEQ - 1):
            out = model(src, ys)
            nxt = out[:, -1].argmax(-1, keepdim=True)  # next token = argmax of last position
            ys = torch.cat([ys, nxt], dim=1)
    print("source: ", src[0].tolist())
    print("decoded:", ys[0, 1:].tolist())  # drop the leading <bos> when printing


if __name__ == "__main__":
    train()