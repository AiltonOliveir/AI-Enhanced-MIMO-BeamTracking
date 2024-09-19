import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import argparse
from src.utils import Ailton_AttrDict
import json
from termcolor import colored
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam, RMSprop
from keras.metrics import TopKCategoricalAccuracy

from src.callbacks import SaveTestOutputsCallback
from src.model import selection_model
from src.datagenerator import DataGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", "-c", help="Path to config file", type=str, default='./cfg/main_pipeline.json'
    )
    args = parser.parse_args()

    with open(args.cfg_file, "r") as cfg_file:
        cfg = Ailton_AttrDict.from_dict(json.load(cfg_file))

    print(
        colored("# Declaring Data generators...", "green", attrs=["reverse", "blink"])
    )
    test_datagen = DataGenerator(mode="test", cfg=cfg.DataGenerator)

    print(colored("# Declaring sequence model...", "green", attrs=["reverse", "blink"]))
    model = selection_model(
        input_shape=cfg.DataGenerator.input_shape,
        n_beams=cfg.DataGenerator.nTx * cfg.DataGenerator.nRx
    )

    if cfg.Optimizers.choise.lower() == "adam":
        print(
            colored(
                "# Setting up Adam optimizer...", "green", attrs=["reverse", "blink"]
            )
        )
        opt = Adam(
            learning_rate=cfg.Optimizers.Adam.learning_rate,
            beta_1=cfg.Optimizers.Adam.beta_1,
            beta_2=cfg.Optimizers.Adam.beta_2,
            epsilon=cfg.Optimizers.Adam.epsilon,
            amsgrad=cfg.Optimizers.Adam.amsgrad,
            name=cfg.Optimizers.Adam.name,
        )
    elif cfg.Optimizers.choise.lower() == "rmsprop":
        print(
            colored(
                "# Setting up RMSprop optimizer...", "green", attrs=["reverse", "blink"]
            )
        )
        opt = RMSprop(
            learning_rate=cfg.Optimizers.RMSprop.learning_rate,
            rho=cfg.Optimizers.RMSprop.rho,
            momentum=cfg.Optimizers.RMSprop.momentum,
            epsilon=cfg.Optimizers.RMSprop.epsilon,
            centered=cfg.Optimizers.RMSprop.centered,
            name=cfg.Optimizers.RMSprop.name,
        )
    print(colored("# Compiling the model...", "green", attrs=["reverse", "blink"]))
    model.compile(
        optimizer=opt,
        loss=CategoricalCrossentropy(),
        metrics=[
            TopKCategoricalAccuracy(k=i, name="top{}".format(i)) for i in range(1, 11)
        ],
    )

    model.load_weights(f"models/{cfg.model_name}_epochs_{cfg.epochs}_salehi.weights.h5")

    print(
        colored(
            "# Starting the testing process...", "green", attrs=["reverse", "blink"]
        )
    )

    filename_outputs = f"logs/{cfg.model_name}_test_output"
    callback = SaveTestOutputsCallback(test_datagen, filename_outputs, model, cfg.DataGenerator)
    
    scores = model.evaluate(
        x=test_datagen,
        batch_size=cfg.DataGenerator.batchsize,
        steps=len(test_datagen),
        callbacks=[callback]
    )

    filename = f"logs/{cfg.model_name}_test_epochs_{cfg.epochs}.csv"
    summary = dict(zip(model.metrics_names, scores))
    summary = pd.DataFrame(summary, index=[0])
    summary.to_csv(filename, index=False)