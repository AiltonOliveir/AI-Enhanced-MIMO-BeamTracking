import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from src.utils import Ailton_AttrDict
import json
from termcolor import colored
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.metrics import TopKCategoricalAccuracy

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

    print(colored("# Declaring Data generators...", "green", attrs=["reverse", "blink"]))
    
    train_datagen = DataGenerator(mode="train", cfg=cfg.DataGenerator)
    val_datagen = DataGenerator(mode="val", cfg=cfg.DataGenerator)

    print(colored("# Declaring sequence model...", "green", attrs=["reverse", "blink"]))
    model = selection_model(
        input_shape=cfg.DataGenerator.input_shape,
        n_beams=cfg.DataGenerator.nTx * cfg.DataGenerator.nRx,
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
    model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=["accuracy", 
                                                                          TopKCategoricalAccuracy(2, name='top2'),
                                                                          TopKCategoricalAccuracy(3, name='top3'),
                                                                          TopKCategoricalAccuracy(4, name='top4'),
                                                                          TopKCategoricalAccuracy(5, name='top5')])

    print(
        colored(
            "# Starting the training process...", "green", attrs=["reverse", "blink"]
        )
    )
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    filename = f"logs/{cfg.model_name}_train_epochs_{cfg.epochs}.csv"
    history_logger = CSVLogger(filename, separator=",", append=True)

    if not os.path.isdir("models"):
        os.mkdir("models")
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f"models/{cfg.model_name}_epochs_{cfg.epochs}_salehi.weights.h5",
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )
    history = model.fit(
        train_datagen,
        validation_data=val_datagen,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_datagen),
        callbacks=[history_logger, model_checkpoint_callback],
    )
    history_dict = history.history
    with open('salehi_training_history.json', 'w') as f:
        json.dump(history_dict, f)





