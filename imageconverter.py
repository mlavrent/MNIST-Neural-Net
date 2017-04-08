from PIL import Image
import numpy as np

class ImageConverter(object):
    
    def imageToNPArray(self, path):
        im = Image.open(path)
        
        rPix = np.array(list(im.getdata(band=0)))
        gPix = np.array(list(im.getdata(band=1)))
        bPix = np.array(list(im.getdata(band=2)))

        #Convert to one-color grayscale
        pixels = np.sqrt((rPix**2 + gPix**2 + bPix**2)/3)

        return pixels

    def invertImage(self, pixArray):
        return 255 - pixArray

im = ImageConverter()
a = im.imageToNPArray("../Images/four.png")

a = im.invertImage(a)


