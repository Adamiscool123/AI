import math

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._children = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __rsub__(self, other):
        return Value(other)-self

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += 1.0*out.grad
            other.grad += -1.0*out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self+other
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad

        out._backward = _backward
        
        return out

    def __rmul__(self, other):
        return self*other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad

        out._backward = _backward
        
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data**other.data, (self, other), '**')

        def _backward():
            self.grad += (other.data * (self.data**(other.data-1)))*out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        checked = set()
        def build_topo(v):
            if v not in checked:
                checked.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def sigmoid(self):
        n = (1) / (1+math.exp(-self.data))
        out = Value(n, (self, ), 'sigmoid')

        def _backward():
            self.grad += (math.exp(-self.data)) / ((1+math.exp(-self.data))**2) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        n = (math.exp(2*self.data)-1) / (math.exp(2*self.data)+1)
        out = Value(n, (self, ), 'tanh')

        def _backward():
            self.grad += (1-(out.data)**2)*out.grad

        out._backward = _backward

        return out