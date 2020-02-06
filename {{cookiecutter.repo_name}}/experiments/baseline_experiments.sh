
python3 ../training_scripts/run_experiment.py   '{"model": "BaseModel", 
                                          "dataset": "{{cookiecutter.dataset_loading_class}}", 
                                          "network":"linear",  
                                          "num_epochs": 3, 
                                          "device": "cpu", 
                                          "dataset_args": {"validation_split": 0.2}, 
                                          "optimizer":"SGD", 
                                          "optimizer_args": {"lr":1e-3}}' > linear_results.txt
                                          
                                          
python3 ../training_scripts/run_experiment.py   '{"model": "BaseModel", 
                                          "dataset": "{{cookiecutter.dataset_loading_class}}", 
                                          "network":"mlp",  
                                          "num_epochs": 3, 
                                          "device": "cpu", 
                                          "dataset_args": {"validation_split": 0.2}, 
                                          "optimizer":"SGD", 
                                          "optimizer_args": {"lr":1e-3}}' > mlp_results.txt                                          