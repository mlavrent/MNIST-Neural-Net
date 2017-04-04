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

        return labels, data
    
    def scaleData(self, data):
        return data/255

    def labelsToOutputs(self, labels):
        outputs = np.zeros((labels.size, 10))

        for i in range(labels.size):
            outputs[i,labels[i]] = 1

        return outputs
        
    def getInputsOutputs(self):
        labels, data = self.genTrainData()
        inputs = self.scaleData(data)
        outputs = self.labelsToOutputs(labels)

        return inputs, outputs
