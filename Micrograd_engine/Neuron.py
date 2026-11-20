import random
import numpy as np

class Neuron:
  def __init__(self, num_input):
    self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_input)]
    self.bias = Value(random.uniform(-1, 1))

  def __call__(self, X, end):
    outs = sum(((w * x) for w, x in zip(self.weights, X)), self.bias)
    if(end):
      return outs.sigmoid()

    return outs.tanh()

  def parameters(self):
    out = self.weights + [self.bias]
    return out

class Layer:
  def __init__(self, num_input, num_neurons):
    self.neurons = [Neuron(num_input) for _ in range(num_neurons)]

  def __call__(self, X, end):
    outs = [n(X, end) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    out = [y for x in self.neurons for y in x.parameters()]
    return out

class MLP:
  def __init__(self, num_inputs, num_neurons):
    sz = [num_inputs] + num_neurons
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(num_neurons))]

  def __call__(self, X):
    for k in self.layers:
      if self.layers[-1] == k:
        X = k(X, True)
        return X
      X = k(X, False)
    return X

  def parameters(self):
    out = [y for x in self.layers for y in x.parameters()]
    return out

