from datagenerator import _DataGenerator
import numpy as np

class NeuralNet(object):
    def __init__(self, inputLayerSize, h1LayerSize, h2LayerSize, outputLayerSize):

        self.inputLayerSize = inputLayerSize
        self.h1LayerSize = h1LayerSize
        self.h2LayerSize = h2LayerSize
        self.outputLayerSize = outputLayerSize

        self.i_h1_weight = np.random.rand(inputLayerSize, h1LayerSize)*2 - 1
        self.h1_h2_weight = np.random.rand(h1LayerSize, h2LayerSize)*2 - 1
        self.h2_o_weight = np.random.rand(h2LayerSize, outputLayerSize)*2 - 1

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

    def costFunction(self, aOutValues, rOutValues):
        # aOutValues - [. . .] (outputLayerSize)
        # rOutValues - [. . .] (outputLayerSize)
        total = 0
        for i in range(self.outputLayerSize):
            total += 0.5*(rOutValues[i] - aOutValues[i])**2

        return total

    def backProp(self, inValues, rOutValues):
        aOutValues = self.forwardProp(inValues)

        #Iterate over h2_o_weights
        dC_dh2o_w = np.zeros((self.h2LayerSize, self.outputLayerSize))
        for o in range(self.outputLayerSize):
            for h2 in range(self.h2LayerSize):
                dC_dh2o_w[h2][o] = -(rOutValues[o] - aOutValues[o]) * aOutValues[o]*(1-aOutValues[o]) * self.h2Val[h2]

        #Iterate over h1_h2_weights
        dC_dh1h2_w = np.zeros((self.h1LayerSize, self.h2LayerSize))
        for h2 in range(self.h2LayerSize):
            for h1 in range(self.h1LayerSize):
                dC_dh1h2_w[h1][h2] = dC_dh2 * self.h2Val[h2]*(1-self.h2Val[h2]) * self.h1Val[h1]
                #TODO: calculate dC_dh2

        #Iterate over i_h1_weights
        dC_dih1_w = np.zeros((self.inputLayerSize, self.h1LayerSize))
        for h1 in range(self.h1LayerSize):
            for i in range(self.inputLayerSize):
                dC_dih1_w[i][h1] = dC_dh1 * self.h1Val[h1]*(1-self.h1Val[h1]) * self.inputVal[i]
                #TODO: calculate dC_dh1

        return {"i_to_h1": dC_dih1_w,
                "h1_to_h2": dC_dh1h2_w,
                "h2_to_o": dC_dh2o_w}


dg = _DataGenerator()
inputs, outputs = dg.getInputsOutputs()

nn = NeuralNet(784, 15, 15, 10)

nn.backProp(inputs[0], outputs[0])
