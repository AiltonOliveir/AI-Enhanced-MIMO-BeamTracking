import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Conv2D
from keras.layers import Input, ReLU, Flatten
from keras.layers import Dense, Dropout
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model

import numpy as np

def Residual(nChannels, inputs):
    x = inputs
    y = Conv2D(nChannels, (3,3), padding='same')(x)
    y = BatchNormalization(axis=(1,2))(y)
    y = ReLU()(y)
    y = Conv2D(nChannels, (3,3), padding='same', activation=None)(y)
    y = BatchNormalization(axis=(1,2))(y)
    
    if nChannels != inputs.shape[-1]:
        x = Conv2D(nChannels, (1,1))(x)
    y = y + x
    y = ReLU()(y)
    return y

def selection_model(input_shape, n_beams):
    # Lidar Input Processing (Beam Selection)
    lidar_input = Input(input_shape)
    x = Conv2D(16, (5,5), padding='same', activation='relu')(lidar_input)
    x = BatchNormalization(axis=(1,2))(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Dropout(0.1)(x)
    x = Residual(32, x)
    x = BatchNormalization(axis=(1,2))(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Dropout(0.1)(x)
    x = Residual(64, x)
    x = BatchNormalization(axis=(1,2))(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Dropout(0.1)(x)
    x = Residual(128, x)
    x = BatchNormalization(axis=(1,2))(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(n_beams*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(n_beams, activation="softmax")(x)
    
    return Model(inputs=lidar_input, outputs=x)

if __name__ == "__main__":
    model = selection_model(input_shape=(192, 186, 10), n_beams=64)
    input = [np.ones((20,192, 186, 10))]
    print(model(input))
    model.summary()
