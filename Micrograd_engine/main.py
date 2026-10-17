from engine import Value
from visual import draw_dot
from Neuron import MLP


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

n = MLP(4, [4, 4, 1])

y_pred = [n(x) for x in xs]

y_target = [1.0, 2.0, 3.0, 4.0]

loss = sum(((y_p - y_t)**2 for y_p, y_t in zip(y_pred, y_target)))

print(loss)