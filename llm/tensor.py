import numpy as np


class Tensor:
    def __init__(self, data, parents=(), op=""):
        if not isinstance(data, (np.ndarray, np.generic)):
            data = np.array(data, dtype=np.float32)
        self.data = np.asarray(data)
        self.grad = None
        self._parents = parents
        self._backward = lambda: None
        self.op = op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self.op})"
    
    def _add_grad(self, g):

        g = unbroadcast(g, self.shape)

        if self.grad is None:
            self.grad = g
        else:
            self.grad = self.grad + g

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other, dtype=self.data.dtype))
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self._add_grad(out.grad)
            other._add_grad(out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other, dtype=self.data.dtype))
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self._add_grad(out.grad * other.data)
            other._add_grad(out.grad * self.data)

        out._backward = _backward
        return out
    def backward(self):
        order, visited = [], set()

        def visit(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            for parent in t._parents:
                visit(parent)
            order.append(t)
        visit(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            t._backward()

        # clear each object has passed its gradient backward
        # clean links so that it doesn't wast memory during training.
        for t in order:
            t._parents = ()
            t._backward = lambda: None
    def __pow__(self, k):
        out = Tensor(self.data ** k, (self, ), f"**{k}")

        def _backward():
            self._add_grad(out.grad * k * self.data ** (k - 1))

        out._backward = _backward
        return out
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self.data * other ** -1
        return self * (1.0 / other)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return (-self) + other

    def exp(self):
        out = Tensor(np.exp(self.data), (self, ), "exp")

        def _backward():
            self._add_grad(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self, ), "log")

        def _backward():
            self._add_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self, ), "tanh")

        def _backward():
            self._add_grad(out.grad * (1.0 - out.data ** 2))

        out._backward = _backward
        return out


    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._add_grad(np.broadcast_to(g, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            g = out.grad
            self._add_grad(g @ np.swapaxes(other.data, -1, -2))
            other._add_grad(np.swapaxes(self.data, -1, -2) @ g)
        out._backward = _backward
        return out
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            self._add_grad(out.grad.reshape(self.shape))
            out._backward = _backward
        return out

    
    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,), "transpose")
        def _backward():
            self._add_grad(out.grad.transpose(*np.argsort(axes)))
            out._backward = _backward
        return out

def numerical_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        up = f()
        x[idx] = old - eps
        down = f()
        x[idx] = old

        grad[idx] = (up - down) / ( 2* eps)
    return grad
def unbroadcast(grad, shape):

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

