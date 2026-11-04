import numpy as np


from llm.tensor import Tensor
from llm.nn import Module, Linear, softmax

def casual_mask(T):
    """ (T, T): 0 where a token may look ( itself and the past )
    -1e9 where it may not (the future)"""
    
    allowed = np.tril(np.ones((T,T), dtype=bool))
    return Tensor(np.where(allowed, 0.0, -1e9).astype(np.float32))


