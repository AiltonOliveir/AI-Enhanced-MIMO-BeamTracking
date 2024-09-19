import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Conv2D
from keras.layers import Input, ReLU, Flatten, Reshape
from keras.layers import Dense, Dropout
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import CategoryEncoding, TimeDistributed
from keras.layers import Concatenate
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

def tracking_model(input_shape, window_length, n_beams):
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
    x = Reshape((1, n_beams))(x) # (time stemp, n_beams one hot)
    
    # Stack Beam Selection with the previous Beams
    beam_index_input = Input((window_length))
    y = Reshape((window_length, 1))(beam_index_input) #(time stemp, n_beams int)
    y = TimeDistributed(CategoryEncoding(num_tokens=n_beams, output_mode='one_hot'))(y)
    z = Concatenate(axis=1)([y,x])
    
    #Tracking of Beams
    z = LSTM(n_beams*4, return_sequences=True)(z)
    z = LSTM(n_beams)(z)
    z = Dense(n_beams, activation="softmax")(z)
    
    return Model(inputs=[lidar_input, beam_index_input], outputs=z)

if __name__ == "__main__":
    model = tracking_model(input_shape=(348,320,7), window_length=3, n_beams=16)
    input = [np.ones((20,348,320,7)), np.ones((20,3))]
    print(model(input))
    model.summary()
