from engine import Value
from visual import draw_dot
from Neuron import MLP


X = [2.0, 3.0, 4.0, 5.0, -3.0]
n = MLP(5, [10, 10, 10, 10, 1])

dot = draw_dot(n(X))
dot.render('graph', view=True, cleanup=True)