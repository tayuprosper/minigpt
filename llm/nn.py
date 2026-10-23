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


                       