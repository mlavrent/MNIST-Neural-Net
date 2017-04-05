import csv
import numpy as np

class _DataGenerator(object):
    def genTrainData(self):
        data = []
        with open('../train-data.csv', 'r') as f:
            data = [list(map(int,rec)) for rec in csv.reader(f, delimiter=',')]

        data = np.array(data)
        labels = data[:,0]
        data = np.delete(data, 0, 1)

        data = np.split(data, data.shape[0]*.75)[0]
        labels = np.split(labels, labels.shape[0]*.75)[0]

        testData = np.split(data, data.shape[0]*.75)[1]
        testLabels = np.split(labels, labels.shape[0]*.75)[1]

        return data, labels, testData, testLabels
    
    def scaleData(self, data):
        return data/255

    def labelsToOutputs(self, labels):
        outputs = np.zeros((labels.size, 10))

        for i in range(labels.size):
            outputs[i,labels[i]] = 1

        return outputs
        
    def getInputsOutputs(self):
        data, labels, testData, testLabels = self.genTrainData()
        
        trainInputs = self.scaleData(data)
        trainOutputs = self.labelsToOutputs(labels)

        testInputs = self.scaleData(testData)
        testOutputs = self.labelsToOutputs(testLabels)

        return trainInputs, trainOutputs, testInputs, testOutputs
