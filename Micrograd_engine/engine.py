import math
import numpy as np

class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = np.asarray(data, dtype=float)
    self._prev = set(_children)
    self.label = label
    self._op = _op
    self.grad = np.zeros_like(self.data)
    self._backward = lambda : None

  def __repr__(self):
    return f"Data: {self.data} Grad: {self.grad}"

  def __radd__(self, other):
    return self + other

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad

    out._backward = _backward

    return out

  def __rsub__(self, other):
    return (self*-1)+(other)

  def __sub__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + (other.data*-1.0), (self, other), '-')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += -1.0 * out.grad

    out._backward = _backward

    return out

  def __rmul__(self, other):
    return self * other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward

    return out

  def __rpow__(self, other):
    return (self ** other)

  def __pow__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data ** other.data, (self, other), '**')

    def _backward():
      self.grad += (other.data * (self.data ** (other.data-1)))*out.grad

    out._backward = _backward

    return out

  def backward(self):
    topo = []
    verified = set()

    def build_topo(v):
      if(v not in verified):
        verified.add(v)
        for i in v._prev:
          build_topo(i)
        topo.append(v)

    build_topo(self)

    self.grad = 1.0

    for k in reversed(topo):
      k._backward()

  def tanh(self):
    x = max(-50.0, min(50.0, self.data))
    n1 = (math.exp(x) + math.exp(-1*x))
    n2 = (math.exp(x) - math.exp(-1*x))
    tanh = n2/n1
    out = Value(tanh, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - (tanh)**2)*out.grad

    out._backward = _backward

    return out

  def sigmoid(self):
    x = max(-50.0, min(50.0, self.data))
    n = (1) / (1 + math.exp(-1*x))
    out = Value(n, (self, ), 'sigmoid')

    def _backward():
      self.grad += (n * (1 - n))*out.grad

    out._backward = _backward

    return out