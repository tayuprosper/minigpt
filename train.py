import os
import time
import numpy as np
from llm.tokenizer import CharTokenizer
from llm.nn import manual_seed
from llm.model import GPT, GPTConfig, save_model
from llm.optim import AdamW

BLOCK_SIZE = 64
BATCH_SIZE = 32
STEPS      = 2000
MAX_LR     = 3e-3
EVAL_EVERY = 250


text = open("data/fables.txt", encoding="utf-8").read()
tok  = CharTokenizer(text)
data = np.array(tok.encode(text))
n    = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"{len(train_data):,} training / {len(val_data):,} validation tokens")


def get_batch(split, rng):
    starts = rng.integers(0, len(split) - BLOCK_SIZE - 1, size=BATCH_SIZE) 
    x = np.stack([split[s:s+BLOCK_SIZE] for s in starts])
    y = np.stack([split[s+1:s+BLOCK_SIZE+1] for s in starts])
    return x, y


def get_lr(step, warmup=100, min_ratio=0.1):
    if step < warmup:
        return MAX_LR * (step + 1) / warmup
    progress = (step - warmup) / (STEPS - warmup)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return MAX_LR * (min_ratio + (1 - min_ratio) * cosine)

manual_seed(0)
model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE))
optimizer = AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
print(f"Model has {model.num_params():,} parameters")

def estimate_loss(splits):
    model.eval()
    rng = np.random.default_rng(123)
    losses = []
    for _ in range(8):
        x, y = get_batch(splits, rng)
        losses.append(float(model(x, y)[1].data))
    model.train()
    return np.mean(losses)


os.makedirs("checkpoints", exist_ok=True)
rng = np.random.default_rng(0)
best_val = float("inf")
start = time.time()
model.train()

for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0:
        tr, va = estimate_loss(train_data), estimate_loss(val_data)
        note = ""
        if va < best_val:
            best_val = va
            save_model(model, tok.chars, "checkpoints/fables.npz")
            note = " saved"
        secs = time.time() - start
        print(f"step {step:4} | train {tr:.3f} | val {va:.3f} "
        f"| {secs:4.0f}s{note}", flush=True)
    if step == STEPS:
        break
    optimizer.lr = get_lr(step)
    x, y = get_batch(train_data, rng)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()