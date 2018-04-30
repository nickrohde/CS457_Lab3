# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:46:22 2018

@author: nick
"""

import random 

def main():
    random.seed()    
    

    
def printResults(x,t,neuron):
    file = open("results", "a")
    correct = 0.0
    accuracy = 0.0
    
    for i in range(len(x)):
        y = neuron.evaluate(x[i])
        file.write("Neuron evaulated to ")
        file.write(str(y))
        file.write(" expected ")
        file.write(str(t[i]))
        file.write("\n")
        if y == t[i]:
            correct += 1.0

    accuracy = (correct / len(x)) * 100.0
    
    file.write("Accuracy: ")
    file.write(str(accuracy))
    file.write("%\n")
    file.write("Final weights of neuron:\n")
    file.write(" ".join('%0.2f' % item for item in neuron.w))
    file.write("\n\n")

    return accuracy


class Neuron:
    def __init__(self, w_initial, learnRate):
        self.LEARN_RATE = learnRate
        self.w = list(w_initial)
        self.changes = 0
        
    def evaluate(self, x):
        raise NotImplementedError()
        
    def trainNeuron(self, x, t):
        raise NotImplementedError()
        
    def adjustWeights(self, x, y, t):
        raise NotImplementedError()


class SigmoidNeuron(Neuron):
    b = 0
    
    def __init__(self, w_initial, learnRate, b_val):
        Neuron.__init__(self, w_initial, learnRate)
        b = b_val
        
    def evaluate(self, x):
        
        
    def trainNeuron(self, x, t):
        
        
    def adjustWeights(self, x, y, t):
        

class LinearNeuron(Neuron):

    def __init__(self, w_initial, learnRate):
        Neuron.__init__(self, w_initial, learnRate)
    
    def evaluate(self, x):
        y = 0
        for i in range(len(x)):
            y += self.w[i] * x[i]
        
        if y > 0: 
            return 1
        else:
            return -1
        
    def trainNeuron(self, x, t):
        successful = False
        
        # train until all training vectors were evaluated correctly
        while not successful:
            successful = True

            # shuffle the order of the training set
            order = random.sample(range(len(x)), len(x))

            for i in order:
                y = self.evaluate(x[i])

                # neuron evaluated correctly, do nothing
                if (y == t[i]):
                    continue
                    
                # neuron failed, adjust weights
                else:
                    successful = False
                    self.adjustWeights(x[i],y,t[i])
        
    def adjustWeights(self, x, y, t):
        self.changes += 1
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - self.LEARN_RATE * (y-t) * x[i]
                        
main()
