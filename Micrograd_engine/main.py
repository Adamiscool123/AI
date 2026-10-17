from engine import Value
from visual import draw_dot
from Neuron import MLP


x = [2.0, 3.0, -1.0]

n = MLP(3, [4, 4, 4, 4, 4, 4 ,1])

n(x)

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [0.0, 3.5, -2.0],
    [3.3, -1.5, 1.5],
    [0.2, 5.0, 1.0],
    [1.0, 1.0, -1.0],
]

y_target = [1.0, 0.5, 1.0, 0.5]

for k in range(1000):
    y_pred = [n(x) for x in xs]

    loss = sum((y_p-y_t)**2 for y_p, y_t in zip(y_pred, y_target))

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    for p in n.parameters():
        p.data += -0.08*p.grad

    print(k, loss.data)

print(y_pred)