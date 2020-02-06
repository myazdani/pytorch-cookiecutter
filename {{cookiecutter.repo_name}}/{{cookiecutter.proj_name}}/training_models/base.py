"""Model class, to be extended by specific types of models."""
from pathlib import Path
from typing import Callable, Dict, Optional
import numpy as np

import torch
from tensorboardX import SummaryWriter


DIRNAME = Path(__file__).parents[1].resolve() / 'weights'


class BaseModel:
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict = None, 
                 network_args: Dict = None, optimizer: str = None, optimizer_args: Dict = None,
                 device = "cpu"):
        
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'
        
        if dataset_args is None:
            dataset_args = {}            
        self.train_loader, self.validation_loader = self.train_validation_loader(dataset_cls, 
                                                                                 **dataset_args)               
        
        self.train_losses = []
        self.valid_losses = []
        
        self.network = network_fn(**network_args).to(device)        
        self.optimizer = getattr(torch.optim, optimizer)(self.network.parameters(), 
                                                         **optimizer_args)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.device = device
        
    def __repr__(self):
        strings = [f'Name: {self.name}']
        strings.append(f'Optimizer: {self.optimizer}')
        strings.append(f'Criterion: {self.criterion}')
        strings.append(f'Network: {self.network}')
        strings.append(f'Device: {self.device}')        
        return '\n----------------\n'.join(strings)          
        
        
        
    def fit(self, num_epochs = 1):
        writer = SummaryWriter(logdir = self.name)
        for epoch in range(num_epochs):
            train_loss = self.train(epoch, self.train_loader, self.network, self.criterion, 
                                    self.optimizer, self.device)

            with torch.no_grad():
                valid_loss = self.validate(epoch, self.validation_loader, self.network, 
                                           self.criterion, self.device)    
                
                
            writer.add_scalar('data/train_loss', train_loss, epoch)
            writer.add_scalar('data/valid_loss', valid_loss, epoch)
            writer.add_scalars('data/learning_curves', {"train": train_loss,
                                                        "valid": valid_loss}, epoch) 
            
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
                
                
    def train(self, epoch, dataloader, network, criterion, optimizer, device):
        network.train()
        batch_loss = []
        for i_batch, (x_batch, y_batch) in enumerate(dataloader):
            images = x_batch.to(device)
            labels = y_batch.to(device)

            output = network(images)

            loss = criterion(output, labels)
            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return np.mean(batch_loss)

    def validate(self, epoch, dataloader, network, criterion, device):
        network.eval()
        batch_loss = []
        for i_batch, (x_batch, y_batch) in enumerate(dataloader):
            images = x_batch.to(device)
            labels = y_batch.to(device)

            output = network(images)

            loss = criterion(output, labels)
            batch_loss.append(loss.item())
        mean_loss = np.mean(batch_loss)
        return mean_loss
                
    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.pt')    
    
    def save_weights(self):
        torch.save(self.network.state_dict(), self.weights_filename)

    def load_weights(self):
        self.network.load_state_dict(torch.load(self.weights_filename))
        self.network.eval()
    
    def _evaluate_data_loader(self, data_loader):
        self.network.eval()
        total_hits = []
        for i_batch, (x_batch, y_batch) in enumerate(data_loader):
            images = x_batch.to(self.device)
            labels = y_batch.to(self.device)

            preds = torch.argmax(self.network(images), 1)

            total_hits.append(preds == labels)   
        
        return torch.cat(total_hits).cpu().numpy().mean()
        
    
    def evaluate(self, data_loader = None):
        if data_loader is None:
            train_avg_accs = self._evaluate_data_loader(self.train_loader)
            print("Avg training accuracy:", train_avg_accs)
            valid_avg_accs = self._evaluate_data_loader(self.validation_loader)
            print("Avg validation accuracy:", valid_avg_accs)

            return train_avg_accs, valid_avg_accs
        else:
            data_loader_acc = self._evaluate_data_loader(data_loader)
            return data_loader_acc
        
    def train_validation_loader(self, dataset_cls, csv_file, batch_size = 32, validation_split = .2, 
                                shuffle_dataset = True, random_seed = None):
        '''https://stackoverflow.com/a/50544887'''
        dataset = dataset_cls(csv_file)
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                   sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                        sampler=valid_sampler)

        return train_loader, validation_loader   
    


        
def main():
    pass
    
    
if __name__ == "__main__":
    main()