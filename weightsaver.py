import numpy as np

class WeightSaverLoader(object):

    def saveWeights(self, pathToFolder, *args):
        # *args are the np array weights in order
        fileName = ""

        #Generate file name for npz archive
        for array in args:
            if fileName != "":
                fileName += "_"
                
            fileName += str(array.shape[0])
        fileName += "_" + str(args[len(args) - 1].shape[1])
        fileName = pathToFolder + fileName + ".npz"
 
        np.savez(fileName, *args)

    def loadWeights(self, file, nnArray):
        #weightArray is an array of the number of neurons in each layer
        loadedFile = np.load(file)

        #Error checking for equal number of layers
        if len(nnArray)-1 != len(loadedFile.items()):
            return -1

        allWeights = [loadedFile["arr_" + str(num)] for num in range(len(loadedFile.items()))]

        #Error checking for non-matching imported values and needed sizes
        for i in range(len(allWeights)):
            if allWeights[i].shape[0] != nnArray[i]:
                return -1
        if allWeights[len(allWeights) - 1].shape[1] != nnArray[len(nnArray) - 1]:
            return -1
            
        loadedFile.close()
        
        return allWeights
