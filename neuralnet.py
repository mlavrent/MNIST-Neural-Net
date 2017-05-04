from datagenerator import DataGenerator
from imageconverter import ImageConverter
from weightsaver import WeightSaverLoader
import numpy as np
import datetime

class NeuralNet(object):
    LEARNING_RATE = 0.5
    
    def __init__(self, inputLayerSize, h1LayerSize, h2LayerSize, outputLayerSize, loadWeights=False):
        self.wsl = WeightSaverLoader()
        np.seterr(over='ignore')
        
        self.inputLayerSize = inputLayerSize
        self.h1LayerSize = h1LayerSize
        self.h2LayerSize = h2LayerSize
        self.outputLayerSize = outputLayerSize

        if not(loadWeights):
            self.i_h1_weight = np.random.rand(inputLayerSize, h1LayerSize)*2 - 1
            self.h1_h2_weight = np.random.rand(h1LayerSize, h2LayerSize)*2 - 1
            self.h2_o_weight = np.random.rand(h2LayerSize, outputLayerSize)*2 - 1
        else:
            weights = self.wsl.loadWeights("weightData/" + str(inputLayerSize) + "_" + str(h1LayerSize) + \
                            "_" + str(h2LayerSize) + "_" + str(outputLayerSize) + ".npz", \
                                           [inputLayerSize, h1LayerSize, h2LayerSize, outputLayerSize])

            self.i_h1_weight = weights[0]
            self.h1_h2_weight = weights[1]
            self.h2_o_weight = weights[2]
            

        self.inputVal = np.zeros(inputLayerSize)
        self.h1Val = np.zeros(h1LayerSize)
        self.h2Val = np.zeros(h2LayerSize)
        self.outVal = np.zeros(outputLayerSize)

    def saveWeights(self):
        self.wsl.saveWeights("weightData/", self.i_h1_weight, self.h1_h2_weight, self.h2_o_weight)

    def categorizeImage(self, imgArray):
        output = self.forwardProp(imgArray)

        return np.argmax(output)

    def forwardProp(self, inValues):
        self.inputVal = inValues
        
        self.h1Val = self.sigmoid(np.dot(self.inputVal, self.i_h1_weight))
        self.h2Val = self.sigmoid(np.dot(self.h1Val, self.h1_h2_weight))
        
        self.outVal = self.sigmoid(np.dot(self.h2Val, self.h2_o_weight))
        
        return self.outVal

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def costFunction(self, aOutValues, rOutValues):
        # aOutValues - [. . .] (outputLayerSize) - actual obtained values (after forward prop)
        # rOutValues - [. . .] (outputLayerSize) - the real values (from data set labels)
        total = 0
        for i in range(self.outputLayerSize):
            total += 0.5*(rOutValues[i] - aOutValues[i])**2

        return total

    def averageCost(self, aOutValues, rOutValues):
        # aOutValues - [[. . .]
        #               [. . .]
        #               [. . .]] (num values down, outputLayerSize across)
        total = 0
        for v in range(aOutValues.shape[0]):
            total += self.costFunction(aOutValues[v], rOutValues[v])

        return total/aOutValues.shape[0]

    def backProp(self, inValues, rOutValues):
        aOutValues = self.forwardProp(inValues)

        #Iterate over h2_o_weights
        dC_dh2o_w = np.zeros((self.h2LayerSize, self.outputLayerSize))
        delta_h2_o = np.zeros(self.outputLayerSize)
        
        for o in range(self.outputLayerSize):
            delta_h2_o[o] = -(rOutValues[o] - aOutValues[o]) * aOutValues[o]*(1-aOutValues[o])
            for h2 in range(self.h2LayerSize):
                dC_dh2o_w[h2][o] = delta_h2_o[o] * self.h2Val[h2]

        #Iterate over h1_h2_weights
        dC_dh1h2_w = np.zeros((self.h1LayerSize, self.h2LayerSize))
        delta_h1_h2 = np.zeros(self.h2LayerSize)
        
        for h2 in range(self.h2LayerSize):
            for o in range(self.outputLayerSize):
                delta_h1_h2[h2] += delta_h2_o[o]*self.h2_o_weight[h2][o] * \
                                   self.h2Val[h2]*(1-self.h2Val[h2])
                
            for h1 in range(self.h1LayerSize):
                dC_dh1h2_w[h1][h2] = delta_h1_h2[h2] * self.h1Val[h1]

        #Iterate over i_h1_weights
        dC_dih1_w = np.zeros((self.inputLayerSize, self.h1LayerSize))
        delta_i_h1 = np.zeros(self.h1LayerSize)
        
        for h1 in range(self.h1LayerSize):
            for h2 in range(self.h2LayerSize):
                delta_i_h1[h1] += delta_h1_h2[h2]*self.h1_h2_weight[h1][h2] * \
                                  self.h1Val[h1]*(1-self.h1Val[h1])
                
            for i in range(self.inputLayerSize):
                dC_dih1_w[i][h1] = delta_i_h1[h1] * self.inputVal[i]

        return {"i_to_h1": dC_dih1_w,
                "h1_to_h2": dC_dh1h2_w,
                "h2_to_o": dC_dh2o_w}
    
    def train(self, inputData, labelData):

        for d in range(inputData.shape[0]):
            changes = self.backProp(inputData[d], labelData[d])

            self.i_h1_weight -= self.LEARNING_RATE * changes["i_to_h1"]
            self.h1_h2_weight -= self.LEARNING_RATE * changes["h1_to_h2"]
            self.h2_o_weight -= self.LEARNING_RATE * changes["h2_to_o"]

            if d % (inputData.shape[0]/100) == 0:
                print(str(round(d/inputData.shape[0]*100, 1)) + "%")


start = datetime.datetime.now()

#Uncomment to do further training on the Neural Network
'''dg = DataGenerator()
trainInputs, trainOutputs, testInputs, testOutputs = dg.getInputsOutputs()'''
'''
end = datetime.datetime.now()


print("Data Loaded in " + str(round((end-start).total_seconds(),1)) + " seconds")
'''
nn = NeuralNet(784, 15, 15, 10, loadWeights=True)
'''
print(nn.averageCost(nn.forwardProp(testInputs), testOutputs))
nn.train(trainInputs, trainOutputs)
print(nn.averageCost(nn.forwardProp(testInputs), testOutputs))'''

imgC = ImageConverter()
arr = imgC.loadImageAsArray("../Images/roham.png")
print(nn.categorizeImage(arr))
#np.set_printoptions(suppress=True)

'''totalHits = 0
totalTries = 0

for i in range(testInputs.shape[0]):
    guess = nn.categorizeImage(testInputs[i])
    actual = np.argmax(testOutputs[i])
    
    if guess == actual:
        totalHits += 1
        
    totalTries += 1

print(totalHits/totalTries * 100)

#nn.saveWeights()'''

