"""Model class, to be extended by specific types of models."""
from pathlib import Path
from typing import Callable, Dict, Optional
import numpy as np

import torch
import pdb

from .base import BaseModel


DIRNAME = Path(__file__).parents[1].resolve() / 'weights'


class XavierModel(BaseModel):
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict = None, 
                 network_args: Dict = None, optimizer: str = None, optimizer_args: Dict = None,
                 device = "cpu", xavier_gain = 1.0):
        
        super().__init__(dataset_cls, network_fn, dataset_args, network_args, 
                         optimizer, optimizer_args, device)
        
        
        for p in list(self.network.parameters()):
            if len(p.data.size()) > 1:
                torch.nn.init.xavier_normal_(p, gain = xavier_gain)
                
            
                
        
        
def main():
    pass
    
if __name__ == "__main__":
    main()