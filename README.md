# MNIST-Neural-Net
This is a neural network designed to train on the MNIST data set for recognizing handwritten digits. It is run with an input layer of 784 inputs (28x28 images) and 2 hidden layers, each with 15 neurons. The output layer has 10 neurons, each of which represents the probability that the image is the digit 0-9, in order. The sigmoid activation function is used. 

The data set used to train this is the MNIST data set, found here: http://yann.lecun.com/exdb/mnist/. The training data set (60 000 images) is used for everything; it is split 75/25% for training and testing, respectively.

# Other Classes
The program also provides several helper classes - one loads and prepares the training data from the MNIST set. The other uses the Pillow library to be able to import a user's own images for running through the program.

# File Structure
The MNIST set is too large to include here on Github; as a result, the file structure to use the images and to train the data is shown here.

Digit Recognizer    
|   train-data.csv
└───Images
|
└───NeuralNets

All of the code found in this repository should be located in the `NeuralNets` folder. The `Images` folder is for other images you may want to test yourself. The image loader is able to take images larger than 28x28 (it will scale them down), but it is suggested to provide 28x28 .png images (which can be made in MS Paint).

The `train_data.csv` can be found here https://pjreddie.com/media/files/mnist_train.csv.

