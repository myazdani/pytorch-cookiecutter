from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import Sequential


class MLP(nn.Module):
    
    def __init__(self, input_shape,
                 output_shape,
                 layer_size, 
                 dropout_amount,
                 num_layers):
        
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
    
        num_classes = self.output_shape[0]
        num_inputs = self.input_shape[0]        

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, layer_size))    
        for _ in range(num_layers):
            self.layers.append(nn.Linear(layer_size, layer_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_amount))
        self.layers.append(nn.Linear(layer_size,num_classes))
        self.layers.append(nn.Softmax(dim = 1))

    
    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return z
    
def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int = 128,
        dropout_amount: float = 0.2,
        num_layers: int = 3):
    """
    Simple multi-layer perceptron: just fully-connected layers with 
    dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    net = MLP(**locals())
    return net 


def main():
    model = mlp(input_shape = (28,), output_shape = (10,))
    x = torch.randn(12,28)
    output = model(x.view(-1,28))
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output probs sum to:", output[0].sum())
    

if __name__ == '__main__':
    main()