import numpy as np


from llm.tensor import Tensor
from llm.nn import Module, Linear, softmax

def causal_mask(T):
    """ (T, T): 0 where a token may look ( itself and the past )
    -1e9 where it may not (the future)"""
    
    allowed = np.tril(np.ones((T,T), dtype=bool))
    return Tensor(np.where(allowed, 0.0, -1e9).astype(np.float32))


class CausalSelfAttention(Module):
    """ Multi-head attention where tokens only see the past"""
    
    def __init__(self, n_embd, n_head):
        assert n_embd % n_head == 0, "n_embd must be divisble by n_head"
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.query = Linear(n_embd, n_embd)
        self.key = Linear(n_embd, n_embd)
        self.value = Linear(n_embd, n_embd)
        self.proj = Linear(n_embd, n_embd)
        self.weights = None
      
    def split_heads(self, x):
        B, T, C = x.shape
        x = x.reshape(B, T, self.n_head, self.head_size)
        return x.transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(0, 2, 1,3).reshape(B, T, H * D)
    
    def forward(self, x):
        T = x.shape[1]
        q = self.split_heads(self.query(x))
        # (B, H, T, D)
        
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        scores = q @ k.transpose(0, 1, 3, 2)
        # (B, H, T, T)
        
        scores = scores * (1.0 / np.sqrt(self.head_size))
        scores = scores + causal_mask(T)
        # hide the future
        
        weights = softmax(scores, axis=-1)
        # rows sum to 1
        self.weights = weights.data
        out = weights @ v
        # (B, H, T, D)
        return self.proj(self.merge_heads(out))