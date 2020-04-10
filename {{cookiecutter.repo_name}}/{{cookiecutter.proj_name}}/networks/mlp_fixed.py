from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import Sequential
from .mlp import MLP

class MLPfixed(MLP):
    
    def __init__(self, input_shape,
                 output_shape,
                 layer_size, 
                 dropout_amount,
                 num_layers):
        
        super().__init__(input_shape, output_shape, layer_size, 
                         dropout_amount,
                         num_layers)
        
        self.fix_all_layers_except_last()

            
    def fix_all_layers_except_last(self):
        # set grads for all layers to False
        for p in reversed(list(self.parameters())):
            p.requires_grad = False
        # set grads of last layersr to True
        for i, p in enumerate(reversed(list(self.parameters()))):
            if i > 1:
                break
            p.requires_grad = True        
    
def mlp_fixed(input_shape: Tuple[int, ...],
              output_shape: Tuple[int, ...],
              layer_size: int = 128,
              dropout_amount: float = 0.2,
              num_layers: int = 3):
    """
    Simple multi-layer perceptron: just fully-connected layers with 
    dropout between them, with softmax predictions. All hidden layers 
    except the last are fixed. Creates num_layers layers.
    """
    net = MLPfixed(**locals())
    return net 


def main():
    model = mlp_fixed(input_shape = (28,), output_shape = (10,))
    x = torch.randn(12,28)
    output = model(x.view(-1,28))
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output probs sum to:", output[0].sum())
    

if __name__ == '__main__':
    main()