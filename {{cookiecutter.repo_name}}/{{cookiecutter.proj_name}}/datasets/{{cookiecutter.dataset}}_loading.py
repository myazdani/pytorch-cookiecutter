import argparse
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch
  
class {{cookiecutter.dataset_loading_class}}(Dataset):
    '''Loading data'''
    
    def __init__(self):
        pass        
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
        
    @property 
    def input_share(self):
        pass

    
def _parse_args():
    pass


def main():
    '''example loading data and passing through one batch'''
    pass
    

if __name__ == '__main__':
    main()