from datagenerator import _DataGenerator

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
