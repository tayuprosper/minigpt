import sys
import numpy as np
from llm.model import load_model
from llm.tokenizer import CharTokenizer
def next_token_probs(model, ids):
    """Probabilities for the token that follows the list `ids`."""
    context = np.array(ids[-model.config.block_size:])[None, :]
    logits, _ = model(context)
    z = logits.data[0, -1].astype(np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()

def generate(model, ids, n_new, rng):
    """Add n_new tokens to the list `ids`, one at a time."""
    ids = list(ids)
    for _ in range(n_new):
        p = next_token_probs(model, ids)
        ids.append(int(rng.choice(len(p), p=p)))
    return ids


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The "
    model, chars = load_model("checkpoints/fables.npz")
    tok = CharTokenizer(chars)
    rng = np.random.default_rng()
    ids = generate(model, tok.encode(prompt), 300, rng)
    print(tok.decode(ids))
    
    
if __name__ == "__main__":
    main()