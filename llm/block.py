import numpy as np

from llm.nn import Module, param, gelu, Linear, Dropout
from llm.attention import CausalSelfAttention


class LayerNorm(Module):
    
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = param(np.ones(dim))
        self.beta = param(np.zeros(dim))
        
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        x_hat = centered * (var + self.eps) ** -0.5
        return x_hat * self.gamma + self.beta
    
class MLP(Module):
    
    def __init__(self, n_embd):
        self.fc = Linear(n_embd, 4 * n_embd)
        self.proj = Linear(4 * n_embd, n_embd)
        
    def forward(self, x):
        return self.proj(gelu(self.fc(x)))
    

class Block(Module):
    """
    One transformer block: attention, then MLP, each
    with LayerNorm before it and a residual connection around it.
    """
    
    def __init__(self, n_embd, n_head, dropout):
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.drop = Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x
    