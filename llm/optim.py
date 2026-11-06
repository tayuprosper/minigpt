import numpy as np


class AdamW:
    """Adam with decoupled weight decay."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0
        
    def zero_grad(self):
        for p in self.params:
            p.grad = None
            
    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue    
        m[:] = b1 * m + (1 - b1) * p.grad
        v[:] = b2 * v + (1 - b2) * p.grad ** 2
        m_hat = m / (1 - b1 ** self.t)
        v_hat = v / (1 - b2 ** self.t)
        if p.data.ndim >= 2:
            p.data -= self.lr * self.weight_decay * p.data
        p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)