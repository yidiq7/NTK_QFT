''' defining the model'''

import torch
import torch.nn as nn
from torch.optim import SGD
import pytorch_lightning as pl

class Net(pl.LightningModule):
    
    def __init__(self, lr=1e-3, width=8, activation='tanh'):
        super(Net, self).__init__()
        self.lr = lr # learning rate
        self.loss = nn.MSELoss() # using MSE Loss
        self.epoch = self.current_epoch # save current epoch
        self.activation = activation # nonlinearity
        
        if activation == 'tanh':
            self.nonlin = nn.Tanh() # Tanh activation
            
        if activation == 'relu':
            self.nonlin = nn.ReLU() # ReLU activation
        
        self.width = width
        self.input_dim, self.output_dim = 1, 1 # 1D vector input, output

        self.fc1 = nn.Linear(self.input_dim, self.width)
        self.fc2 = nn.Linear(self.width, self.output_dim)
        
        self.fc2.weight.data.fill_(0) # 0 output at initialization
        self.fc2.bias.data.fill_(0) # 0 output at initialization
        
    def forward(self, x):
        x = x.view(-1,1) # fix size mismatch
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.view(-1,1)) # compute loss
        logs={"train_loss:", loss} # for logging
        batch_dictionary={"loss": loss, "log": logs} # for logging
        return batch_dictionary
    
    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean() # report average loss
        self.logger.experiment.add_scalar("Train_Loss/Epoch", avg_loss, self.current_epoch) # for logging 
        epoch_dictionary = {'loss': avg_loss}
        
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr) # training using gradient descent
        return optimizer
        
if __name__ == '__main__':
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    ### model statistics ###
    model = Net()
    print(model.eval())
    print(f"Number of model parameters: {count_parameters(model)}")
    