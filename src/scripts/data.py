''' simulating the data'''

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1) # simulate same data every time

class Data(Dataset):
    
    def __init__(self, xdata, ydata):     
        self.inp = xdata[:]
        self.outp = ydata[:]

    def __len__(self):
        return len(self.inp)
    
    def __getitem__(self, idx):
        inpt  = torch.Tensor(self.inp[idx])
        oupt  = torch.Tensor(self.outp[idx])
        return inpt, oupt

def generate_data(train_samples = 10000, test_samples=1000):
    
    x_train, x_test, y_train, y_test = [], [], [], []
    
    x_train = [np.random.uniform(-10,10) for i in range(train_samples)] # drawn from uniform distribution
    x_test = [np.random.uniform(-10,10) for i in range(test_samples)] # drawn from uniform distribution
        
    y_test = [1 if x_test[i] < 0 else -1 for i in range(len(x_test))] # 1 if x < 0, -1 if x > 0
    y_train = [1 if x_train[i] < 0 else -1 for i in range(len(x_train))] # 1 if x < 0, -1 if x > 0
    
    return (torch.Tensor(x_train), torch.Tensor(x_test), torch.Tensor(y_train), torch.Tensor(y_test))

x_train, x_test, y_train, y_test = generate_data()
    
train_ds = Data(x_train, y_train) # turn arrays into Dataset
test_ds = Data(x_test, y_test) # turn arrays into Dataset

train_dl = DataLoader(train_ds, batch_size=train_ds.__len__()) # using full batch for training 
test_dl = DataLoader(test_ds, batch_size=64) # not used

data = {'train_dl': train_dl, 'test_dl': test_dl, 'test_data': y_test} # called in model

if __name__ == '__main__':
    
    sns.set() # cosmetics
    
    ### plotting ###
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(x_train, y_train, s=5, color='blue')
    ax1.set_title('Train Data')
    ax2.scatter(np.linspace(-10,10,10000), x_train, s=1, color='red')
    ax2.set_title('Train Data Distribution')
    plt.show()
    
