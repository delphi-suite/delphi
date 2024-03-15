# set environment variabel HYDRA_FULL_ERROR=1
# to get full error messages
import os
from pprint import pprint

import hydra
from beartype import beartype

# from model_config import Config
from omegaconf import DictConfig, OmegaConf

from delphi.static.hydra_configs.model_config import instantiate_config


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert DictConfig to a plain dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Extract `llama2hf_config` as a dict and instantiate Llama2HfConfig,
    # then instantiate Config with type checking
    config = instantiate_config(cfg_dict)

    pprint(config, depth=4)
    # print(type(config.llama2hf_config.hidden_act))


if __name__ == "__main__":
    main()
