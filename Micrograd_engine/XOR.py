from Neuron import MLP

X = [0.0, 1.0]
n = MLP(2, [4, 4, 1])
xs = [
    [0.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
]
y_target = [0.0, 0.0, 1.0, 1.0]

i = 1000

for k in range(i):
    y_pred = [n(x) for x in xs]
    loss = sum((y_p - y_t)**2 for y_p, y_t in zip(y_pred, y_target))
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    for p in n.parameters():
        p.data += -0.05*p.grad
    if(k%100 == 0 or k == i-1):
        print(k, loss.data)
        print(y_pred)