import csv
import numpy as np

class _DataGenerator(object):
    def genTrainData(self):
        data = []
        with open('train-data.csv', 'b') as f:
            data = [list(map(int,rec)) for rec in csv.reader(f, delimiter=',')]

        data = np.array(data)
        labels = data[:,0]
        np.delete(data, 0, 1)

        return labels, data
