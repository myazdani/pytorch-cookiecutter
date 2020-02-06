import argparse
import json
import importlib
from typing import Dict
import os
import sys
sys.path.append("..")


import torch


def run_experiment(experiment_config: Dict, save_weights: bool):
    """
    Run a training experiment.
    Parameters
    ----------
    experiment_config : (dict)
        Set of parameters for traninig model (for example, in training_models/base.py). 
        Keys needed are:
        "model" : eg,  "BaseModel"
        "dataset": eg, "{{cookiecutter.dataset_loading_class}}"
        "dataset_args": dict of args used to instantiate {{cookiecutter.dataset_loading_class}}
                        and specify validation split, eg {"validation_split": 0.2}
        "network": eg, "linear"
        "num_epochs": eg, 3
        "device": eg "cpu"
        "optimizer": "SGD"
        "optimizer_args": eg, {"lr":1e-3}
        
    """    
    
    models_module = importlib.import_module("{{cookiecutter.proj_name}}.training_models")
    model_class_ = getattr(models_module, experiment_config["model"])
    del experiment_config["model"]
    networks_module = importlib.import_module("{{cookiecutter.proj_name}}.networks")

    network_fn = getattr(networks_module, experiment_config["network"])
    del experiment_config["network"]
    datasets_module = importlib.import_module("{{cookiecutter.proj_name}}.datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    del experiment_config["dataset"]
    
    num_epochs = experiment_config["num_epochs"]
    del experiment_config["num_epochs"]
    
    model = model_class_(dataset_cls=dataset_class_, 
                         network_fn=network_fn,
                         **experiment_config)
    
    print(model)
    
    model.fit(num_epochs = num_epochs)
    
    
    if save_weights:
        model.save_weights()
    

    model.evaluate()
    


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final weights will be saved to canonical, version-controlled location"
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Experimenet JSON ('{\"dataset\": \"{{cookiecutter.dataset_loading_class}}\", \"model\": \"BaseModel\", \"network\": \"mlp\"}'"
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()

    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config, args.save)

if __name__ == "__main__":
    main()