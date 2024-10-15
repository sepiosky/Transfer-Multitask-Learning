import pickle
from socketserver import ThreadingUnixDatagramServer


import src
import argparse
from pathlib import Path
import os
from src.config import get_cfg_defaults
from src.utils import train


EXPERIMENTS_DIRECTORY = "src/config/experiments"
EXPERIMENTS_FILE_EXTENSION = ".yaml"
PIPELINE = ['build_model', 'train_model', 'eval_model']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--experiment','-e',help=f"Experiment config file name from '{EXPERIMENTS_DIRECTORY}' ")
    parser.add_argument('--debug', '-d', action='store_true', help="Enable debug mode")

    cfg = get_cfg_defaults()

    if parser.parse_args().debug:
        cfg.DEBUG = True

    experiment_config_path: str = (Path(os.path.abspath(EXPERIMENTS_DIRECTORY))/parser.parse_args().experiment).with_suffix(EXPERIMENTS_FILE_EXTENSION)
    cfg.merge_from_file(experiment_config_path)

    cfg.freeze()

    train(cfg)
