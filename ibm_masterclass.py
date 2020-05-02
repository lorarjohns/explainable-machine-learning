from keras import backend as K
from keras.models import Sequential
from keras.models.normalization import BatchNormalization
from keras.layers.convolutional import (Conv2D, MaxPooling2D)
from keras.layers.core import (Activation, Flatten, Dropout, Dense)

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        '''
        @param width: image width in pixels
        @param height: image height in pixels
        @param depth: number of channels: for color usually 3 (RGB),
            for grayscale usually 1
        @param classes: number of classes in the final fully-connected 
            output layer
        '''
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1

        if K.image_data_format() == 'channels_first':
            channel_dim = 1
            input_shape = (depth, height, width)

model = Sequential([Flatten(input_shape=(28,28)),
                    Dense(128, activation='relu'),
                    Dense(10, activation='softmax')])

fashion_mnist = keras.datasets.mnist
(train_images, test_images), (train_labels, test_labels) = fashion_mnist.load_data()

class_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

train_images = train_images/255
test_images = test_images/255

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\n Test Accuracy {test_acc}\nTest Loss{test_loss}')