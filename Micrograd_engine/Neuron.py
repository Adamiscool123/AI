import random
from engine import Value

#Created a Neuron
class Neuron:
    #Initialized weights and bias. Takes in amount of inputs
    def __init__(self, amount_input):
        #Creates a list of x amount of weights depending on the amount of inputs and creates a random value from -1 to 1 for each weight.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(amount_input)]
        #Creates a random bias value from -1 to 1
        self.b = Value(random.uniform(-1, 1))
    #So I can do x = [2.0, 3.0] n = Neuron(amount_input=2) n(x)
    #It is basically the Neurons body
    def __call__(self, X):
        #We do zip() so we can connect the weights to its corresponding X values.
        output = sum((wi*xi for wi,xi in zip(self.w, X)), self.b) #Equivalent to sum of (wi*xi) + bias 
        #Pass through a sigmoid function
        final = output.sigmoid()
        return final

class Layer:
    def __init__(self, amount_input, amount_neurons):
        self.neurons = [Neuron(amount_input) for _ in range(amount_neurons)]
    def __call__(self, X):
        outs = [n(X) for n in self.neurons]
        return outs

#The Mutli_Layer Perceptron
class MLP:
    def __init__(self, amount_input, layers):
        #Connect amount of neurons in input layer + neurons in other layers to a list
        sz = [amount_input] + layers
        #Create a layers list where it creates the neurons for each layer
        #This connects the first input to the 2nd layer then the 2nd input to the third layer and etc...
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(layers))]
    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X