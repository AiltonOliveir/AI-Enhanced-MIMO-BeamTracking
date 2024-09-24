import os
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

from src.utils import Ailton_AttrDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", "-c", help="Path to config file", type=str, default='cfg/main_pipeline.json'
    )
    args = parser.parse_args()

    with open(args.cfg_file, "r") as cfg_file:
        cfg = Ailton_AttrDict.from_dict(json.load(cfg_file))

    filename = f"logs/{cfg.model_name}_train_epochs_{cfg.epochs}.csv"
    plot_train_path = f"logs/{cfg.model_name}_train_epochs_{cfg.epochs}.png"

    if os.path.exists(filename):
        df_train = pd.read_csv(filename)
        plt.subplot(2,1,1)
        sns.lineplot(df_train['accuracy'], label='Train')
        sns.lineplot(df_train['val_accuracy'], label='Val')
        plt.grid()
        plt.subplot(2,1,2)
        sns.lineplot(df_train['loss'], label='Train')
        sns.lineplot(df_train['val_loss'], label='Val')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid()
        plt.savefig(plot_train_path, bbox_inches='tight')
        plt.close()
    else:
        print(f'File not found: {filename}')
        print('Skipping train plots')
    
    filename = f"logs/{cfg.model_name}_test_epochs_{cfg.epochs}.csv"
    plot_test_path = f"logs/{cfg.model_name}_test_epochs_{cfg.epochs}.png"

    if os.path.exists(filename):
        df_train = pd.read_csv(filename)
        topk_values = []
        for i in range(1,11):
            topk_values.append(df_train[f'top{i}'].values[0])
        plt.plot(range(1,11),topk_values)
        plt.xlabel('Top K\'s')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.savefig(plot_test_path, bbox_inches='tight')
        plt.close()
    else:
        print(f'File not found: {filename}')
        print('Skipping test plots')