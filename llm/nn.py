import numpy as np
from llm.tensor import Tensor


RNG = np.random.default_rng(0)


def manual_seed(seed):
    global RNG
    RNG = np.random.default_rng(seed)


def param(array):
    # A Tensor that training is allowed to change
    t = Tensor(np.asarray(array, dtype=np.float32))
    t.is_param = True

    return t


class Module:
    def parameters(self):
        params = []
        for value in self.__dict__.values():
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, Module):
                    params += item.parameters()
                elif getattr(item, "is_param", False):
                    params.append(item)
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def num_params(self):
        return sum(p.data.size for p in self.parameters())
    
    def __call__(self, *args):
        return self.forward(*args)


class Linear(Module):
    # y = x @ W + b

    def __init__(self, n_in, n_out, bias=True, std=0.02):
        self.W = param(RNG.normal(0.0, std, size=(n_in, n_out)))
        self.b = param(np.zeros(n_out)) if bias else None

    def forward(self, x):
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y


def gelu(x):
    # Gaussian Error Linear Unit
    c = np.sqrt(2.0 / np.pi)
    inner  = ( x + 0.044715 * (x * x *x)) * c
    return 0.5 * x * (1.0 + inner.tanh())


def softmax(x, axis=-1):
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    y = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(y, (x,), "softmax")

    def _backward():
        g = out.grad
        x._add_grad(y * (g - (g * y).sum(axis=axis, keepdims=True)))

    out._backward = _backward
    return out



def cross_entropy(logits, targets):

    x = logits.data

    N = x.shape[0]
    shifted = x - x.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_sum

    loss = np.asarray(-log_probs[np.arange(N), targets].mean(), x.dtype)
    out = Tensor(loss, (logits, ), "cross_entropy")

    def _backward():
        g = np.exp(log_probs)
        g[np.arange(N), targets] -= 1.0
        logits._add_grad(g * (out.grad / N))

    out._backward = _backward
    return out


def embedding_lookup(weight, idx):
    """ out [...] = weight[idx[...]]: Pick one row per id"""
    out = Tensor(weight.data[idx], (weight, ), "embedding")

    def _backward():
        g = np.zeros_like(weight.data)
        np.add.at(g, idx, out.grad)
        weight._add_grad(g)

    out._backward = _backward
    return out

class Embedding(Module):

    def __init__(self, n_rows, dim, std=0.02):
        self.weight = param(RNG.normal(0.0, std, size=(n_rows, dim)))

    def forward(self, idx):
        return embedding_lookup(self.weight, np.asarray(idx))

