from dataclasses import dataclass
import numpy as np
from llm.block import Block, LayerNorm
from llm.nn import Module, Embedding, Linear, Dropout, cross_entropy

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.2
    
class GPT(Module):
    def __init__(self, config):
        self.config = config
        C  = config.n_embd
        self.tok_emb = Embedding(config.vocab_size, C)
        self.pos_emb = Embedding(config.block_size, C)
        self.drop = Dropout(config.dropout)
        self.blocks = [Block(C, config.n_head, config.dropout) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(C)
        self.head = Linear(C, config.vocab_size, bias=False)
        
        
    def forward(self, idx, targets=None):
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence too long"
        x = self.tok_emb(idx) + self.pos_emb(np.arange(T))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        V = logits.shape[-1]
        loss = cross_entropy(logits.reshape(B*T, V), np.asarray(targets).reshape(B*T))
        
        return logits, loss
    


def save_model(model, chars, path):
    c = model.config
    arrays = {f"p{i}": p.data for i, p in enumerate(model.parameters())}
    np.savez(path, chars=np.array(chars),config=np.array([c.vocab_size, c.block_size, c.n_layer,c.n_head, c.n_embd]), **arrays)
    
def load_model(path):
    f = np.load(path)
    v, bs, nl, nh, ne = (int(n) for n in f["config"])
    model = GPT(GPTConfig(v, bs, nl, nh, ne, dropout=0.0))
    for i, p in enumerate(model.parameters()):
        p.data = f[f"p{i}"]     
    model.eval()
    return model, "".join(f["chars"])