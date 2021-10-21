import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def vae_loss(x, x_out, mu, logvar, weights, beta=1):
    """
    Binary Cross Entropy Loss + Kiebler-Lublach Divergence
    
    Adapted from https://github.com/oriondollar/TransVAE
    """
    x = x.long()[:,:-1]# - 1 #this has been changed from original implementation
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
#     print('loss', x.size())
#     print('loss', x_out.size())
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCE + KLD, BCE, KLD


class KLAnnealer:
    """
    Scales KL weight (beta) linearly according to the number of epochs
    
    Taken from https://github.com/oriondollar/TransVAE

    """
    def __init__(self, kl_low, kl_high, n_epochs, start_epoch):
        self.kl_low = kl_low
        self.kl_high = kl_high
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch

        self.kl = (self.kl_high - self.kl_low) / (self.n_epochs - self.start_epoch)

    def __call__(self, epoch):
        k = (epoch - self.start_epoch) if epoch >= self.start_epoch else 0
        beta = self.kl_low + k * self.kl
        if beta > self.kl_high:
            beta = self.kl_high
        else:
            pass
        return beta
    
    
def vae_prop_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, beta=1):
    """
    Binary Cross Entropy Loss + Kiebler-Lublach Divergence
    
    This version includes a MSE loss for property prediction

    Taken from https://github.com/oriondollar/TransVAE
    """
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCE + KLD + MSE, BCE, KLD, MSE
