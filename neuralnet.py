from datagenerator import _DataGenerator
import numpy as np

class NeuralNet(object):
    def __init__(self, inputLayerSize, h1LayerSize, h2LayerSize, outputLayerSize):

        self.inputLayerSize = inputLayerSize
        self.h1LayerSize = h1LayerSize
        self.h2LayerSize = h2LayerSize
        self.outputLayerSize = outputLayerSize

        self.i_h1_weight = np.random.rand(inputLayerSize, h1LayerSize)
        self.h1_h2_weight = np.random.rand(h1LayerSize, h2LayerSize)
        self.h2_o_weight = np.random.rand(h2LayerSize, outputLayerSize)

        self.inputVal = np.zeros(inputLayerSize)
        self.h1Val = np.zeros(h1LayerSize)
        self.h2Val = np.zeros(h2LayerSize)
        self.outVal = np.zeros(outputLayerSize)

    def forwardProp(self, inValues):
        self.inputVal = inValues
        
        self.h1Val = self.sigmoid(np.dot(self.inputVal, self.i_h1_weight))
        self.h2Val = self.sigmoid(np.dot(self.h1Val, self.h1_h2_weight))
        
        self.outVal = self.sigmoid(np.dot(self.h2Val, self.h2_o_weight))
        
        return self.outVal

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))


dg = _DataGenerator()
inputs, outputs = dg.getInputsOutputs()

nn = NeuralNet(784, 15, 15, 10)
print(nn.forwardProp(inputs))
