'''Training uses PyTorch Lightning and Tensorboard for logging. will need an existing 
tensorboard install to view the training results

parsers are: 

--epoch 
--n_epoch
--lr

feel free to add parsers as necessary
'''

import argparse
import pytorch_lightning as pl
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data import data
from model import DomainNet

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type = int, default = 10000)
    parser.add_argument("--n_epoch", type = int, default = 500)
    parser.add_argument('--lr', type = float, default = 1e-3)
    args = parser.parse_args()
    
    logger = TensorBoardLogger ('tb_logs', name = 'logs') # creates folder for logger
    
    model = DomainNet(lr=args.lr) # call the model
    checkpoint_callback = ModelCheckpoint('../models/runs', # where to save
                                          filename = 'domain_model_{epoch}', 
                                          every_n_epochs = args.n_epoch, # save model every n epoch
                                          save_top_k = -1) # keeps all models 
    
    trainer = pl.Trainer(max_epochs = args.epoch, # how long to train
                        logger = logger,
                        callbacks = [checkpoint_callback])

    trainer.fit(model, 
                data['train_dl'], # train on train set
                data['test_dl']) # evaluating on test set
    trainer.save_checkpoint( "../models/runs/final_model.pt")
    
    print('TRAINING FINISHED')
    
    ### Visualize ###
    
    sns.set()
    
    model.eval()
    x1 = torch.Tensor(np.linspace(-10,10,1000))
    preds = model(x1).detach().numpy()
    
    inf = torch.Tensor(np.array([100000000000])) # effective infinity
    _inf = torch.Tensor(np.array([-100000000000])) # effective negative infinity
    
    Q = (model(inf) - model(_inf)).item() # topological charge
    
    plt.scatter(x1, preds, s=1, color='red', label = 'Model Predictions')
    plt.plot(np.linspace(-10,0), np.ones(50), color = 'black', label = 'Ground Truth')
    plt.plot(np.linspace(0,10), -np.ones(50), color = 'black') 
    plt.title(f'Model: epochs: {args.epoch}, lr: {args.lr}, Q = {Q:.2f}')
    plt.legend()
    plt.show()
