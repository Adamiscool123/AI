from engine import Value

x1 = Value(3.0)
x2 = Value(4.0)
w1 = Value(-1.0)
w2 = Value(2.0)
b = Value(3.0)
n = x1*w1 + x2*w2
o = n + b
p = o.sigmoid()
p.backward()
print(f"Data: {n.data}, Grad: {n.grad}")