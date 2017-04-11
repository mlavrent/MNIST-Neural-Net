from PIL import Image
import numpy as np

class ImageConverter(object):

    def resizeImage(self, image):
        return image.resize((28,28), PIL.Image.LANCZOS)
        
    def imageToNPArray(self, path):
        im = Image.open(path)

        if im.size != (28,28):
            im = self.resizeImage(im)
        
        rPix = np.array(list(im.getdata(band=0)))
        gPix = np.array(list(im.getdata(band=1)))
        bPix = np.array(list(im.getdata(band=2)))

        #Convert to one-color grayscale
        pixels = np.sqrt((rPix**2 + gPix**2 + bPix**2)/3)

        return pixels

    def invertImage(self, pixArray):
        return 255 - pixArray

    def convToBlackWhiteImage(self, pixArray):
        #pixArray should be 1 dimensional
        
        bwArray = [(255 if p > 127 else 0) for p in pixArray]

        return bwArray

    def loadImageAsArray(self, path):
        pixArray = self.imageToNPArray(path)
        pixArray = self.invertImage(pixArray)
        pixArray = self.convToBlackWhiteImage(pixArray)

        return pixArray/255


