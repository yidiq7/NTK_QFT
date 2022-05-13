import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch

from model import DomainNet
from data import data

sns.set()

def load_models(PATH):
    paths = []

    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith('ckpt'):
                paths.append(os.path.join(PATH,file))
            
    models = [DomainNet.load_from_checkpoint(paths[i]) for i in range(len(paths))]

    preds = [models[i](data['test_data']).detach().numpy() for i in range(len(models))]
    
    inf = torch.Tensor(np.array([100000000000]))
    _inf = torch.Tensor(np.array([-100000000000]))
    
    Qs = [(models[i](inf) - models[i](_inf)).item() for i in range(len(models))]
    
    return models, preds, Qs 

models, preds, Qs = load_models(PATH = '/Users/snehpandya/Projects/Domain Net/src/models/test')

length, width = 10, 5
fig, axs = plt.subplots(length, width)

for i in range(0,int(length-1)):
    for j in range(0, length):
        axs[i,j].set_title('test', fontsize=30)
        
        





# def plot(PATH):
#     pass


    
    
    
        