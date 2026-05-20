from engine import Value
import random

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((xi*wi for xi, wi in zip(x, self.w)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        layer = [neuron(x) for neuron in self.neurons]
        return layer[0] if len(layer) == 1 else layer

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

xs = [
  [1.0, 1.0],
  [1.0, -1.0],
  [-1.0, 1.0],
  [-1.0, -1.0]
]
ys = [-1.0, 1.0, 1.0, -1.0]

# implementing neural net
mlp = MLP(2, [4, 4, 1])

for epoch in range(50):
    #forward pass
    y_pred = [mlp(x) for x in xs]

    # loss calculation
    loss = sum((y-yp)**2 for y,yp in zip(ys, y_pred))

    # initializing gradients to 0
    for p in mlp.parameters():
        p.grad = 0

    #backward pass / back propogation
    loss.backward()

    #update parameters
    for p in mlp.parameters():
        p.data += -0.05 * p.grad

    if epoch == 0 or epoch == 49:
        print(f'Loss at epoch : {epoch} is {loss}')

print('Final prdictions: ')
for i in y_pred:
    print(i.data)