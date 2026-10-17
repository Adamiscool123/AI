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

    
