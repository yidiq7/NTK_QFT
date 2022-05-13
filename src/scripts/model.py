''' defining the model'''

import torch
import torch.nn as nn
from torch.optim import SGD
import pytorch_lightning as pl

class DomainNet(pl.LightningModule):
    
    def __init__(self, lr, hidden_dim=8):
        super(DomainNet, self).__init__()
        self.lr = lr
        self.loss = nn.MSELoss() # using MSE Loss
        self.epoch = self.current_epoch # save current epoch
        self.tanh = nn.Tanh() # tanh activation
        
        self.hidden_dim = hidden_dim
        self.input_dim, self.output_dim = 1, 1 # 1D vector input, output

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.fc2.weight.data.fill_(0) # 0 output at initialization
        self.fc2.bias.data.fill_(0) # 0 output at initialization
        
    def forward(self, x):
        x = x.view(-1,1) # fix size mismatch
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.view(-1,1))
        logs={"train_loss:", loss} # for logging
        batch_dictionary={"loss": loss, "log": logs} # for logging
        return batch_dictionary
    
    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean() # report average loss
        self.logger.experiment.add_scalar("Train_Loss/Epoch", avg_loss, self.current_epoch) # for logging 
        epoch_dictionary = {'loss': avg_loss}
        
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr) # training using stochastic gradient descent
        return optimizer
        
if __name__ == '__main__':
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = DomainNet()
    print(model.eval())
    print(f"Number of model parameters: {count_parameters(model)}")
    