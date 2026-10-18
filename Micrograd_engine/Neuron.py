from engine import Value
import random

class Neuron:
    def __init__(self, len_input):
        self.w = [Value(random.uniform(-0.5, 0.5)) for _ in range(len_input)]
        self.b = Value(random.uniform(-0.5, 0.5))
    def __call__(self, X, is_output):
        out = sum(((w_value * x_value) for w_value, x_value in zip(self.w, X)), self.b)

        if is_output:
            out.sigmoid()
        else:
            out.tanh()
        
        return out
    def parameters(self):
        return self.w + [self.b]
class Layer:
    def __init__(self, len_input, len_neuron):
        self.neurons = [Neuron(len_input) for _ in range(len_neuron)]
    def __call__(self, X, i_o):
        outs = [n(X, i_o) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
class MLP:
    def __init__(self, len_input, list_neuron):
        sz = [len_input] + list_neuron
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(list_neuron))]
    def __call__(self, X):
        for layer in self.layers:
            if self.layers[-1] == layer and self.layers.index(layer) == len(self.layers) - 1:
                X = layer(X, True)
                return X
            X = layer(X, False)
        return X
    def parameters(self):
        return [p for n in self.layers for p in n.parameters()]
        