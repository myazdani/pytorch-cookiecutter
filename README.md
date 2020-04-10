![PyTorch Cookiecutter](cookiecutter-pytorch-logo.png)

# PyTorch Cookiecutter

Developing a deep learning model requires extensive experimentation and evaluation. Throughout the life cyle of a project, it's very easy to lose track of the various models, data sets, and hyper-parameters. Having an organized repository makes this process much easier. 

I found the text recognizer project demoed at the [Full Stack Deep Learning bootcamp](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project) inspiring and excellent. I decided to try to generalize their repo as a template for future projects. [Cookiecutter](https://github.com/cookiecutter/cookiecutter) is a tool for creating project templates and I decided to use it to create template in the spirit of the FSDL text recognizer, but use PyTorch instead of Keras as the deep learning framework. 

This template is certainly not unique as many other similar projects exist. I suppose what makes this unique is it is specifically inspired by the work done at the bootcamp. This is still a work in progress (isn't everything?) and is meant mostly for my own work, but maybe others will also find it useful. 

## Starting a new project

If [cookiecutter](https://github.com/cookiecutter/cookiecutter) is not installed, first `pip install cookiecutter` 

After cookiecutter is installed, clone the PyTorch Cookiecutter template:
```cookiecutter https://github.com/myazdani/pytorch-cookiecutter```


You will then get prompted (leaving blank will use defaults):
```
repo_name [project-repo]: iris-classification
proj_name [deep_model]: iris_model
dataset_loading_class [DataSetLoading]: IrisDataLoader
dataset [dataset]:
readme_project_name [Project Name]: Iris Classifier
project_desc [Project description]: A project for building Iris classification with PyTorch
```

This will then create a repo in your working directory with the following structure:

```
iris-classification/
├── experiments
│   └── baseline_experiments.sh
├── iris_model
│   ├── datasets
│   │   ├── __init__.py
│   │   └── dataset_loading.py
│   ├── networks
│   │   ├── __init__.py
│   │   ├── linear.py
│   │   ├── mlp.py
│   │   └── mlp_fixed.py
│   └── training_models
│       ├── __init__.py
│       ├── base.py
│       └── xavier_init.py
├── readme.md
└── training_scripts
    ├── __init__.py
    └── run_experiment.py


```

The folder `iris_model` has all the components needed to build a model: data loading scripts in `datasets`, network architecture scripts in `networks`, and training scripts in `training_models`. You likeley will have to overwrite and fill in the details for dataset_loading (this is where the PyTorch data loaders are defined). A few basic sample architectures that serve as useful baselines are also placed in `networks`. `mlp_fixed.py` is an MLP architecture but with the weights of the hidden layer fixed at random (so only the output layer weights are updated). In my experience this architecture serves as a useful stepping stone when evaluating between the linear architecture and a full blown MLP. 

The specification of backprop (the loss function, specific optimizer, learning rates, batch size etc.) are all specified in `training_models`. `xavier_init.py` is inherits from `base.py` but changes how the network weights are initialized. If you want to try other initializations or compare other training strategies, you can inherit from `base.py` and overwrite as needed. 

In `training_scripts` we have the `run_experiment.py` that runs a specific experiment with the specific network architecture, optimizer, etc. `run_experiment.py` takes a single string argument as a JSON that it then parses as a python dict. For example, we can have:

```
python ./training_scripts/run_experiment.py '{"model": "BaseModel", 
                                              "dataset": "IrisDataLoader", 
                                              "network":"linear",  
                                              "network_args": {"input_shape": 4, "output_shape": 1},
                                              "num_epochs": 3, 
                                              "device": "cpu", 
                                              "dataset_args": {"validation_split": 0.2}, 
                                              "optimizer":"SGD", 
                                              "loss" : "BCEWithLogitsLoss",
                                              "optimizer_args": {"lr":1e-3}}'                                          
```                                          


Finally, since it is so easy to get lost with all the experiments we run, I find it to be good practice to document which experiments were run in executable shell scripts that we place in the `experiments` folder. 


### Full template:

The generic template is organized as follows:
```
├── {{cookiecutter.proj_name}}
│   ├── datasets
│   │   ├── {{cookiecutter.dataset}}_loading.py
│   │   └── __init__.py
│   ├── networks
│   │   ├── __init__.py
│   │   ├── linear.py
│   │   ├── mlp_fixed.py
│   │   └── mlp.py
│   └── training_models
│       ├── base.py
│       ├── __init__.py
│       └── xavier_init.py
├── experiments
│   └── baseline_experiments.sh
├── readme.md
└── training_scripts
    ├── __init__.py
    └── run_experiment.py
```    