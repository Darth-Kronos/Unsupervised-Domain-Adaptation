import argparse
import os
from warnings import warn

import torch

import yaml


def parse_args(config=None, **kwargs):
    parser = argparse.ArgumentParser(description="MoCo")

    # parse config file first, then add arguments from config file
    config = "./utils/default_config.yaml" if config is None else config
    parser.add_argument("--config", default=config)
    args, unknown = parser.parse_known_args()
    config = yaml_config_hook(args.config)

    # add arguments from `config` dictionary into parser, handling boolean args too
    bool_configs = [
        "perturb",
    ]
    for k, v in config.items():
        if k == "config":  # already added config earlier, so skip
            continue
        v = kwargs.get(k, v)
        if k in bool_configs:
            parser.add_argument(f"--{k}", default=v, type=str)
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    for k, v in kwargs.items():
        if k not in config:
            parser.add_argument(f"--{k}", default=v, type=type(v))

    # parse added arguments
    args, _ = parser.parse_known_args()
    for k, v in vars(args).items():
        if k in bool_configs and isinstance(v, str):
            if v.lower() in ["yes", "no", "true", "false", "none"]:
                exec(f'args.{k} = v.lower() in ["yes", "true"]')

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)

    if not os.path.exists(args.tensorboard_log_dir):
        os.makedirs(args.tensorboard_log_dir, exist_ok=True)

    return args




def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            ext = ".yaml" if len(os.path.splitext(cf)) == 1 else ""
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ext)
            if not os.path.exists(cf):
                cf = os.path.basename(cf)
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                cf = os.path.join(repo_root, "config", config_dir, cf + ext)
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg = dict(l, **cfg)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg