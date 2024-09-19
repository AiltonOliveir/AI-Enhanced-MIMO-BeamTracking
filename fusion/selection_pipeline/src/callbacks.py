import os
import numpy as np
from keras.callbacks import Callback

class SaveTestOutputsCallback(Callback):
    def __init__(self, generator, output_dir, model, cfg):
        super(SaveTestOutputsCallback, self).__init__()
        self.generator = generator
        self.output_dir = output_dir
        self.model = model
        self.batchsize = cfg.batchsize
        self.n_scene = cfg.n_scene_in_episode
        if self.batchsize % self.n_scene:
            print("Invalid batch size")
            raise ValueError()
        self.n_ep_samples = self.batchsize // self.n_scene # number of scenes in the episode
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_test_batch_end(self, batch, logs=None):
        #indexs = list(range(batch * self.n_ep_samples, (batch + 1) * self.n_ep_samples))
        X, Y = self.generator[batch]
        predictions = self.model.predict(X)
        
        file_output = os.path.join(self.output_dir, f'{batch}.npz')
        np.savez_compressed(file_output, predictions=predictions, output=Y)
