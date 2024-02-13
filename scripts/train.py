from delphi.train.training import DDP,TrainingConfig, model_initialization, train_loop
from delphi.train.utils import load_config
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument("--log_level", type=str, help="Log level to use.")
    args = parser.parse_args()

    config = load_config(args.config)
    TrainConf = TrainingConfig(config)
    model,model_args = model_initialization(config)
    train_loop(model, TrainConf)