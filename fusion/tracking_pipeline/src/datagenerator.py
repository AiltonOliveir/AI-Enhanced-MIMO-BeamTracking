import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import json
import argparse
from src.utils import Ailton_AttrDict
from termcolor import colored
import pandas as pd
from tensorflow import keras
from keras.utils import to_categorical

class DataGenerator(keras.utils.Sequence):
    def __init__(self, mode, cfg) -> None:
        self.datasetPath = cfg.datasetPath
        self.nTx = cfg.nTx
        self.nRx = cfg.nRx
        self.window_size = cfg.window_size
        self.n_scene = cfg.n_scene_in_episode
        self.batchsize = int(cfg.batchsize // cfg.n_scene_in_episode)
        np.random.seed(cfg.seed)
        self.samplesCSV = pd.read_csv(self.datasetPath.replace("mode", mode))
        if cfg.batchsize % cfg.n_scene_in_episode:
            print("Invalid ", colored("Batchsize", "red"), " value")
            print("Batchsize must be divided by the number of episodes: ", colored(f"{cfg.n_scene_in_episode}", "red"))
            print("Suggested values: ", colored(f"{np.array([1,2,3])*cfg.n_scene_in_episode}", "red"), '...')
            raise ValueError()

    def on_epoch_end(self) -> None:
        self.samplesCSV = self.samplesCSV.sample(frac=1)

    def __len__(self) -> int:
        return int(np.floor(len(self.samplesCSV) / (self.batchsize)))

    def __getitem__(self, index) -> tuple:        
        samples = self.samplesCSV.iloc[
            index * self.batchsize : (index + 1) * self.batchsize
        ]

        for i in range(len(samples)):
            cur_X = np.load(samples.iloc[i]["input_path"])["spatial_input"].astype(float)
            cur_Y = np.load(samples.iloc[i]["input_path"])["beam_index"]
            for scene in range(self.window_size, self.n_scene):
                cur_beam_in = cur_Y[scene-self.window_size:scene]
                if scene == self.window_size:
                    prev_beams = cur_beam_in[np.newaxis,:]
                else:
                    prev_beams = np.concatenate([prev_beams, cur_beam_in[np.newaxis,:]], axis=0)
            if i == 0:
                lidar_X = cur_X[self.window_size:]
                beam_X = prev_beams
                Y = cur_Y[self.window_size:]
            else:
                lidar_X = np.concatenate([lidar_X, cur_X[self.window_size:]], axis=0)
                beam_X = np.concatenate([beam_X, prev_beams], axis=0)
                Y = np.concatenate([Y, cur_Y[self.window_size:]], axis=0)
                
        return ([lidar_X.astype(float), beam_X.astype(float)], to_categorical(Y, self.nTx * self.nRx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", "-c", help="Path to config file", type=str, default='cfg/main_pipeline.json'
    )
    args = parser.parse_args()

    with open(args.cfg_file, "r") as cfg_file:
        cfg = Ailton_AttrDict.from_dict(json.load(cfg_file))

    datagen = DataGenerator(mode="test", cfg=cfg.DataGenerator)

    for i in range(len(datagen)):
        x, y = datagen[i]
        print(x[0].shape, x[1].shape, y.shape)