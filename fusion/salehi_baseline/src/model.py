import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Conv2D
from keras.layers import Input, ReLU, Flatten
from keras.layers import Dense, Dropout
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import regularizers

import numpy as np

def id_block(nChannels, inputs):
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
    x = Conv2D(32, (3,3), padding='same', activation='relu')(lidar_input)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = id_block(32, x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = id_block(32, x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = id_block(32, x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = id_block(32, x)
    x = Flatten()(x)
    x = Dense(n_beams*2, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.25)(x)
    x = Dense(n_beams, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.25)(x)
    x = Dense(n_beams, activation="softmax")(x)
    
    return Model(inputs=lidar_input, outputs=x)

if __name__ == "__main__":
    model = selection_model(input_shape=(348,320,10), n_beams=16)
    input = [np.ones((20,348,320,10))]
    print(model(input))
    model.summary()
