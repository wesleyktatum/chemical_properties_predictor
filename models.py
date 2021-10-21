import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm.notebook import tqdm

import loss
from data_utils import make_std_mask
    
    
SMILES_CHARS = ['*', ' ', '#', '%', '(', ')', '+', '-', '.', '/', '=', '@', '[', '\\', ']',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'Br', 'Cl', 'F', 'H', 'C', 'I', 'K', 'N', 'O', 'P', 'S',
                'br', 'cl', 'f', 'h', 'c', 'i', 'k', 'n', 'o', 'p', 's']
    
###################################
###### Linear Model and Utils #####
###################################

class LinearLogP(nn.Module):
    """
    
    """
    
    def __init__(self, n_hidden, batchsize, embedding_dims):

        super(LinearLogP, self).__init__()
        
        self.SMILES_CHARS = SMILES_CHARS
        
        self.embedder = nn.Embedding(num_embeddings = len(self.SMILES_CHARS),
                                     embedding_dim = embedding_dims,
                                     padding_idx = 0)
        
        self.smiles_in = 500
        self.enc_dims = 100
        self.pred_in = 25100 # = (250 padded SMILES length + 1 Weight) * enc_dims
        self.out_dim = 1
        
        self.hidden_dim = n_hidden
        self.batchsize = batchsize
        
        self.smiles_encoder = nn.Sequential(
            nn.Linear(embedding_dims, self.smiles_in),
#             nn.ReLU(),
            nn.Linear(self.smiles_in, self.enc_dims),
            nn.ReLU()
        )
        
        self.weight_encoder = nn.Sequential(
            #expand the weight scalar to an array that matches SMILES encoding length
            nn.Linear(1, self.enc_dims),
            nn.ReLU()
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(self.pred_in, self.hidden_dim),
#             nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.ReLU()
        )
        
        
    def forward(self, smiles, weight):
        
        embedded_smiles = self.embedder(smiles)
                                     
        # print('SMILES input: \t', smiles.size())
        # print('Embedded SMILES: \t', embedded_smiles.size())
        # print('Weight input: \t', weight.size())
        smiles_enc = self.smiles_encoder(embedded_smiles)
        weight_enc = self.weight_encoder(weight)
        # print('SMILES Encoded: \t', smiles_enc.size())
        # print('Weight Encoded: \t', weight_enc.size())
        
        mol_tensor = torch.cat((smiles_enc, weight_enc), dim = 1)
        mol_tensor = mol_tensor.view(self.batchsize, 1, -1)
        # print('Encoded molecular info: \t', mol_tensor.size())
        
        logP = self.predictor(mol_tensor)
        # print('Prediction tensor: \t', logP.size())
        # print('++++++++++++++++++')
        
        return logP
    
    

def train_logP(model, training_dataset, optimizer):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     epoch_losses = []
    batch_losses = []
    train_count = 0
    train_targets = []
    train_out = []
    
    model.train()
    
    criterion = nn.SmoothL1Loss()
    
    for smiles, weight, logP in tqdm(training_dataset, total = len(training_dataset)):
        smiles = smiles.to(device)
        weight = weight.to(device)
        logP = logP.to(device)
        
        #reset gradients
        for param in model.parameters():
            param.grad = None
        
        #make predictions
        logP_out = model(smiles, weight)
#         print(logP_out.size())
        
        train_targets.extend(logP.cpu().tolist())
        train_out.extend(logP_out.cpu().tolist())
        
        #calculate loss
        loss = criterion(logP_out, logP)
        
        #backpropagate
        torch.autograd.backward(loss)
        optimizer.step()
        
        batch_losses.append(loss.item())
        train_count += 1
        
    epoch_loss = sum(batch_losses)/train_count
    train_parity = {'targets':train_targets,
                    'predictions':train_out}
    
    return model, epoch_loss, train_parity


def test_logP(model, test_dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    criterion = nn.SmoothL1Loss()
    
    with torch.no_grad():
        
        test_losses = []
        test_count = 0
        test_target = []
        test_out = []
        
        for smiles, weight, logP in test_dataset:
            smiles = smiles.to(device)
            weight = weight.to(device)
            logP = logP.to(device)
            
            logP_out = model(smiles, weight)
            
            test_target.extend(logP.cpu().tolist())
            test_out.extend(logP_out.cpu().tolist())
            
            loss = criterion(logP_out, logP)
            test_count += 1
            
            test_losses.append(loss)
            
        test_loss = sum(test_losses)/test_count
        test_parity = {'targets':test_target,
                       'predictions':test_out}
        
        return test_loss, test_parity

    
def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
        
    return
            
    
########################################
##### VAE Models and Utils #####
########################################  


class VAEShell():
    """
    Class for handling, training, and testing of the encoder and decoder halves of
    the VAE. Training both halves allows the trained encoder and latent space to then be used
    for property prediction
    
    Adapted from https://github.com/oriondollar/TransVAE
    """
    
    def __init__(self, params):
        
        self.params = params
        
        assert 'SAVE_PATH' in self.params.keys(), ('Please supply params["SAVE_PATH"]')
        self.save_path = self.params['SAVE_PATH']
        
        assert 'MODEL_TYPE' in self.params.keys(), ('Please supply params["MODEL_TYPE"] as "RNNAttn" or "TransVAE"')
        
        assert 'CHAR_DICT' in self.params.keys(), ('Please supply params["CHAR_DICT"]')
        
        assert 'CHAR_WEIGHTS' in self.params.keys(), ('Please supply params["CHAR_WEIGHTS"]')
        
        
        #identify missing params and use default values
        if 'N_EPOCHS' not in self.params.keys():
            self.params['N_EPOCHS'] = 10
        if 'SAVE_FREQ' not in self.params.keys():
            self.params['SAVE_FREQ'] = self.params['N_EPOCHS']
        if 'SAVE' not in self.params.keys():
            self.params['SAVE'] = True
        if 'LOAD' not in self.params.keys():
            self.params['LOAD'] = False
        if 'EMBEDDING_DIMS' not in self.params.keys():
            self.params['EMBEDDING_DIMS'] = 50
        if 'LR' not in self.params.keys():
            self.params['LR'] = 1e-5
        if 'BETA_INIT' not in self.params.keys():
            self.params['BETA_INIT'] = 1e-8
        if 'BETA' not in self.params.keys():
            self.params['BETA'] = 0.05
        if 'ANNEAL_START' not in self.params.keys():
            self.params['ANNEAL_START'] = 0
        if 'MAX_LENGTH' not in self.params.keys():
            self.params['MAX_LENGTH'] = 250
        
        self.vocab_size = len(self.params['CHAR_DICT'])
        self.pad_idx = 0
        
        self.loss_func = loss.vae_loss
        
        self.src_len = self.params['MAX_LENGTH'] #length of padded SMILES strings + <EOS>
        self.trg_len = self.params['MAX_LENGTH'] - 1 #length of padded SMILES strings
            
        self.n_epochs = self.params['N_EPOCHS']
        self.best_loss = np.inf
        self.best_loss_epoch = 0
        self.current_epoch = 0
        
        self.current_state = {'current_epoch': 0,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'best_loss_epoch': self.best_loss_epoch,
                              'params': self.params}
        
        if self.params['LOAD'] == True:
            self.current_state = self.load(self.save_path)
            
            self.best_loss = self.current_state['best_loss']
            self.best_loss_epoch = self.current_state['best_loss_epoch']
            self.current_epoch = self.current_state['current_epoch']
                
        return
    
    
    def save(self, current_state, filename = None):
        
        if filename == None:
            filename = self.save_path
                
        torch.save(current_state, self.save_path)
            
        return
    
        
    def load(self, filename = None):
        if filename == None:
            filename = self.save_path
            
        ckpt = torch.load(filename)
        
        for k, v in ckpt.items():
            if k == 'params':
                self.params = v
            if k == 'model_state_dict':
                self.current_state['model_state_dict'] = v
            if k == 'optimizer_state_dict':
                self.current_state['optimizer_state_dict'] = v
                
            else:
                self.current_state[k] = v
                
        self.build_model()
        
        self.model.load_state_dict(self.current_state['model_state_dict'])
        self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])

        return self.current_state
    
    
    def train(self, train_dataloader, optimizer, epoch, track_tokens = True, plot_confusion_matrix = False):

        self.model.train()
        losses = []
        accuracies = []
        
        if plot_confusion_matrix:
            tgts = []
            x_outs = []
        
        for mols in tqdm(train_dataloader, desc = 'Training',
                         total = len(train_dataloader),
                         leave = False, position = 1):
            avg_losses = []
            avg_bce_losses = []
            avg_kld_losses = []
            
            avg_smile_acc = []
            avg_token_acc = []
            
            if self.use_gpu:
                mols = mols.cuda()
                
            src = Variable(mols).long()
            tgt = Variable(mols[:,:-1]).long()
            
            src_mask = (src != self.pad_idx).unsqueeze(-2)
            tgt_mask = make_std_mask(tgt, self.pad_idx)
            
            if self.params['MODEL_TYPE'] == 'RNNAttn':
                x_out, mu, logvar = self.model(src, tgt, src_mask, tgt_mask)
                loss, bce, kld = self.loss_func(src, x_out, mu, logvar, self.params['CHAR_WEIGHTS'], self.beta)
                smile_acc, token_acc, position_acc = self.calc_reconstruction_accuracies(tgt, x_out)
                
                avg_losses.append(loss.item())
                avg_bce_losses.append(bce.item())
                avg_kld_losses.append(kld.item())
                
                avg_smile_acc.append(smile_acc)
                avg_token_acc.append(token_acc)
                
                loss.backward()
                
            if track_tokens:
                self.track_token_preds(epoch, tgt, x_out)
            if plot_confusion_matrix:
                tgts.append(tgt)
                x_outs.append(x_out)
                
            loss = sum(avg_losses)/len(avg_losses)
            bce_loss = sum(avg_bce_losses)/len(avg_bce_losses)
            kld_loss = sum(avg_kld_losses)/len(avg_kld_losses)
            
            smile_acc = sum(avg_smile_acc)/len(avg_smile_acc)
            token_acc = sum(avg_token_acc)/len(avg_token_acc)
            #position_acc is a 1x250 array that needs to be preserved and averaged separately
                
            
            losses.append((loss, bce_loss, kld_loss))
            accuracies.append((smile_acc, token_acc, position_acc))
                
            self.optimizer.step()
            self.model.zero_grad()
            
        if plot_confusion_matrix:
            targs = torch.cat(tgts, dim = 0)
            preds = torch.cat(x_outs, dim = 0)
            self.plot_confusion_matrix(targs, preds, train = True)
        
        return losses, accuracies
    
        
    def test(self, test_dataloader, epoch, track_tokens = True, plot_confusion_matrix = False):
        self.model.eval()
        losses = []
        accuracies = []
        
        if plot_confusion_matrix:
            tgts = []
            x_outs = []
        
        for mols in tqdm(test_dataloader, desc = 'Testing',
                         total = len(test_dataloader),
                         leave = False, position = 1):
            avg_losses = []
            avg_bce_losses = []
            avg_kld_losses = []
            
            avg_smile_acc = []
            avg_token_acc = []
            
            if self.use_gpu:
                mols = mols.cuda()
                
            src = Variable(mols).long()
            tgt = Variable(mols[:,:-1]).long()
            
            src_mask = (src != self.pad_idx).unsqueeze(-2)
            tgt_mask = make_std_mask(tgt, self.pad_idx)
            
            if self.params['MODEL_TYPE'] == 'RNNAttn':
                x_out, mu, logvar = self.model(src, tgt, src_mask, tgt_mask)
                loss, bce, kld = self.loss_func(src, x_out, mu, logvar, self.params['CHAR_WEIGHTS'], self.beta)
                smile_acc, token_acc, position_acc = self.calc_reconstruction_accuracies(src, x_out)
                
                avg_losses.append(loss.item())
                avg_bce_losses.append(bce.item())
                avg_kld_losses.append(kld.item())
                
                avg_smile_acc.append(smile_acc)
                avg_token_acc.append(token_acc)
                
            if track_tokens:
                self.track_token_preds(epoch, tgt, x_out)
            if plot_confusion_matrix:
                tgts.append(tgt)
                x_outs.append(x_out)
                
            loss = sum(avg_losses)/len(avg_losses)
            bce_loss = sum(avg_bce_losses)/len(avg_bce_losses)
            kld_loss = sum(avg_kld_losses)/len(avg_kld_losses)
            
            smile_acc = sum(avg_smile_acc)/len(avg_smile_acc)
            token_acc = sum(avg_token_acc)/len(avg_token_acc)
            #position_acc is a 1x250 array that needs to be preserved and averaged separately
            
            losses.append((loss, bce_loss, kld_loss))
            accuracies.append((smile_acc, token_acc, position_acc))
            
        if plot_confusion_matrix:
            targs = torch.cat(tgts, dim = 0)
            preds = torch.cat(x_outs, dim = 0)
            self.plot_confusion_matrix(targs, preds, train = False)
        
        return losses, accuracies
    
    
    def train_test_epochs(self, train_dataloader, test_dataloader, track_tokens = True):
                
        kl_annealer = loss.KLAnnealer(self.params['BETA_INIT'], self.params['BETA'],
                                      self.n_epochs, self.params['ANNEAL_START'])
        
        train_epoch_losses = []
        test_epoch_losses = []
        train_epoch_accuracies = []
        test_epoch_accuracies = []
        
        if track_tokens:
            self.token_tracker = {}
        
        for epoch in tqdm(range(self.current_epoch, self.n_epochs),
                          desc = 'Epochs', position = 0, leave = False):
            
            if track_tokens:
                self.token_tracker[epoch] = {}
                for tok in list(self.params['CHAR_DICT'].values()):
                    self.token_tracker[epoch][tok] = []
            
            self.beta = kl_annealer(epoch)
            
            if epoch == self.n_epochs-1:
                train_losses, train_accs = self.train(train_dataloader,
                                                      self.optimizer,
                                                      epoch = epoch,
                                                      track_tokens = track_tokens,
                                                      plot_confusion_matrix = True)
                test_losses, test_accs = self.test(test_dataloader,
                                                   epoch = epoch,
                                                   track_tokens = track_tokens,
                                                   plot_confusion_matrix = True)
            else:
                #losses and accuracies are returned as a tuple of each type:
                #(total_loss, bce_loss, kld_loss), (smile_acc, token_acc, position_acc)
                train_losses, train_accs = self.train(train_dataloader,
                                                      self.optimizer,
                                                      epoch = epoch,
                                                      track_tokens = track_tokens,
                                                      plot_confusion_matrix = False)
                test_losses, test_accs = self.test(test_dataloader,
                                                   epoch = epoch,
                                                   track_tokens = track_tokens,
                                                   plot_confusion_matrix = False)
            
            train_epoch_loss = self.avg_epoch_performance(train_losses)
            test_epoch_loss= self.avg_epoch_performance(test_losses)
            
            train_epoch_acc = self.avg_epoch_performance(train_accs, accuracy = True)
            test_epoch_acc = self.avg_epoch_performance(test_accs, accuracy = True)
            
            train_epoch_losses.append(train_epoch_loss)
            test_epoch_losses.append(test_epoch_loss)
            
            train_epoch_accuracies.append(train_epoch_acc)
            test_epoch_accuracies.append(test_epoch_acc)
            
            self.current_state['current_epoch'] = epoch
            self.current_state['model_state_dict'] = self.model.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict
            
            if test_epoch_losses[-1][0] < self.best_loss:
                self.best_loss = test_epoch_losses[-1][0]
                self.best_loss_epoch = epoch
                
                self.current_state['best_loss'] = self.best_loss
                self.current_state['best_loss_epoch'] = self.best_loss_epoch
            
            if epoch % int(self.params['SAVE_FREQ']) == 0:
                self.save(self.current_state)
        
        if track_tokens:
            return train_epoch_losses, test_epoch_losses, train_epoch_accuracies, test_epoch_accuracies, self.token_tracker
        else:
            return train_epoch_losses, test_epoch_losses, train_epoch_accuracies, test_epoch_accuracies
    
    
    def avg_epoch_performance(self, metric_list, accuracy = False):
        metric1 = []
        metric2 = []
        metric3 = []
        
        for metric_tup in metric_list:
            metric1.append(metric_tup[0])
            metric2.append(metric_tup[1])
            metric3.append(metric_tup[2])
            
        avg_metric1 = sum(metric1)/len(metric1)
        avg_metric2 = sum(metric2)/len(metric2)
        
        if accuracy:
            metric3 = np.array(metric3)
#             print(metric3.shape)
            avg_metric3 = np.mean(metric3, axis = 0)
#             print(avg_metric3.shape)
                
        else:
            avg_metric3 = sum(metric3)/len(metric3)
        
        return (avg_metric1, avg_metric2, avg_metric3)
    
            
            
    def calc_reconstruction_accuracies(self, input_smiles, output_smiles):
        """
        Calculates SMILE, token and positional accuracies for a set of
        input and reconstructed SMILES strings
        
        adapted from from https://github.com/oriondollar/TransVAE
        """
        max_len = self.params['MAX_LENGTH']
        smile_accs = []
        hits = 0
        misses = 0
        position_accs = np.zeros((2, max_len))
        
        for in_tensor, out_tensor in zip(input_smiles, output_smiles):
            #extract predicted token indices from output tensor
            out_probs = F.softmax(out_tensor, dim = -1)
            out_preds = torch.argmax(out_probs, dim = -1)
            
#             if in_tensor == out_preds:
#                 smile_accs.append(1)
#             else:
#                 smile_accs.append(0)

            misses += abs(len(in_tensor) - len(out_preds))
            for j, (token_in, token_out) in enumerate(zip(in_tensor, out_preds)):
                if token_in == token_out:
                    hits += 1
                    position_accs[0,j] += 1
                else:
                    misses += 1
                position_accs[1,j] += 1
                
            if misses == 0:
                smile_accs.append(1)
            else:
                smile_accs.append(0)
                

        smile_acc = np.mean(smile_accs)
        token_acc = hits / (hits + misses)
        position_acc = []
        
        for i in range(max_len):
            position_acc.append(position_accs[0,i] / position_accs[1,i])
            
        return smile_acc, token_acc, position_acc
    
    
    def track_token_preds(self, epoch, tgt, x_out):
        for in_smiles, out_smiles in zip(tgt, x_out):
            out_smiles = torch.argmax(F.softmax(out_smiles, dim = -1), dim = -1)
            
            for in_tok, out_tok in zip(in_smiles, out_smiles):
                self.token_tracker[epoch][in_tok.item()].append(out_tok.item())
                    
        return
    
    
    def reconstruct_smiles(self, index_tensor):
        smiles_tokens = []
        for ind in index_tensor:
            #get token (key) from char_dict, using token index (value)
            tok = list(self.params['CHAR_DICT'].keys())[list(self.params['CHAR_DICT'].values()).index(ind)]
            smiles_tokens.append(tok)
        
        smiles = ''.join(smiles_tokens) 
        
        return smiles
    
    
    def plot_confusion_matrix(self, tgt, x_out, train = True):
        chars = list(self.params['CHAR_DICT'].keys())
        indices = list(self.params['CHAR_DICT'].values())
                
        targs = []
        preds = []
        
        for in_smiles, out_smiles in zip(tgt, x_out):
            out_smiles = torch.argmax(F.softmax(out_smiles, dim = -1), dim = -1)
            
            for in_tok, out_tok in zip(in_smiles, out_smiles):
                in_tok = chars[indices.index(in_tok.item())]
                out_tok = chars[indices.index(out_tok.item())]
                
                targs.append(in_tok)
                preds.append(out_tok)
        
        print(classification_report(targs, preds, zero_division = 0))
        con_mat = confusion_matrix(targs, preds, labels = chars, normalize = 'true')
        plot = ConfusionMatrixDisplay(confusion_matrix = con_mat, display_labels = chars)
        
        plot.plot(include_values = False, xticks_rotation = "vertical")
        if train:
            plt.title('Training Confusion Matrix', fontsize = 18)
            fig = plt.gcf()
            fig.set_size_inches((20, 20))
        else:
            plt.title('Testing Confusion Matrix', fontsize = 18)
            fig = plt.gcf()
            fig.set_size_inches((20, 20))
        plt.show()
        return
        
    
    

class Embeddings(nn.Module):
    """
    This class stores the dictionaries of embeddings for tensor representations of
    the tokens used in the molecular structures.
    
    adapted from https://github.com/oriondollar/TransVAE
    """
    def __init__(self, embedding_dims, vocab_size):
        super().__init__()
        self.embed_dict = nn.Embedding(vocab_size, embedding_dims)
        self.embedding_dims = embedding_dims
        return

    def forward(self, x):
        return self.embed_dict(x) * math.sqrt(self.embedding_dims)
    
    

class Generator(nn.Module):
    """
    This class takes the reconstructed tensor representation and projects it into
    token-space for translating back into text representations
    
    adapted from https://github.com/oriondollar/TransVAE
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size-1)
        return
    
    
    def forward(self, x):
        return self.projection(x)

    
class AdamWOpt():
    """
    Wrapper class for AdamW optimizer
    
    adapted from https://github.com/oriondollar/TransVAE
    """
    def __init__(self, params, lr):
        self.optimizer = torch.optim.AdamW(params = params, lr = lr)
        self.state_dict = self.optimizer.state_dict()
        return
    
    def step(self):
        self.optimizer.step()
        self.state_dict = self.optimizer.state_dict()
        return
    
    def load_state_dict(self, state_dict):
        self.state_dict = state_dict
        self.optimizer.load_state_dict(state_dict)
        return
    
    
class ConvBottleneck(nn.Module):
    """
    Set of convolutional layers to reduce memory matrix to single
    latent vector
    
    Taken from https://github.com/oriondollar/TransVAE
    """
    def __init__(self, size):
        super().__init__()
        conv_layers = []
        in_d = size
        first = True
        for i in range(3):
            out_d = int((in_d - 64) // 2 + 64)
            if first:
                kernel_size = 9
                first = False
            else:
                kernel_size = 8
            if i == 2:
                out_d = 64
            conv_layers.append(nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size), nn.MaxPool1d(2)))
            in_d = out_d
        self.conv_layers = ListModule(*conv_layers)
        return

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return x
    

class DeconvBottleneck(nn.Module):
    """
    Set of deconvolutional layers to reshape latent vector
    back into memory matrix
    
    adapted from https://github.com/oriondollar/TransVAE
    """
    def __init__(self, size):
        super().__init__()
        deconv_layers = []
        in_d = 64
        for i in range(3):
            out_d = (size - in_d) // 4 + in_d
            stride = 3
            kernel_size = 11
            padding = 1
            if i == 2:
                kernal_size = 11
                out_d = size
                stride = 1
                padding = 4
                
            deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernel_size,
                                                                  stride=stride, padding=padding)))
            in_d = out_d
        self.deconv_layers = ListModule(*deconv_layers)
        return

    def forward(self, x):
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        return x
    
    
class LayerNorm(nn.Module):
    """
    Construct a layernorm module (manual)
    
    Taken from https://github.com/oriondollar/TransVAE
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        return

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class ListModule(nn.Module):
    """
    Create single pytorch module from list of modules
    
    Taken from https://github.com/oriondollar/TransVAE
    """
    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
    
##################################################
################ RNN-Based VAE ###################
##################################################

class RNNAttnVAE(VAEShell):
    """
    Class to build and handle both halves of an RNN with attention-based VAE.
    To utilize the trained encoder for property prediction, use the class
    RNNAttnPropPred
    
    Adapted from https://github.com/oriondollar/TransVAE
    """
    
    def __init__(self, params = {}, N = 3, d_model = 256, d_latent = 128,
                 dropout = 0.1, teacher_force = True):
        
        super().__init__(params)
        
        if 'MODEL_TYPE' not in self.params.keys():
            self.params['MODEL_TYPE'] = 'RNNAttn'
        
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_latent'] = d_latent
        self.params['dropout'] = dropout
        self.params['teacher_force'] = teacher_force
            
        if self.params['LOAD'] == False:
            self.build_model()
        else:
            self.load()
            
        return
    
    
    def build_model(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        encoder = RNNAttnEncoder(self.params['d_model'], self.params['d_latent'],
                                 self.params['N'], self.params['dropout'], self.src_len, self.device)
        decoder = RNNAttnDecoder(self.params['d_model'], self.params['d_latent'],
                                 self.params['N'], self.params['dropout'], self.params['teacher_force'],
                                 self.device)
        generator = Generator(self.params['d_model'], self.vocab_size)
        src_embed = Embeddings(self.params['EMBEDDING_DIMS'], self.vocab_size)
        tgt_embed = Embeddings(self.params['EMBEDDING_DIMS'], self.vocab_size)
        
        self.model = RNNEncoderDecoder(encoder, decoder, generator, src_embed, tgt_embed, self.params)
        
        #initialize model weights
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()
        
        #initiate optimizer
        self.optimizer = AdamWOpt(params = [p for p in self.model.parameters() if p.requires_grad],
                                  lr = self.params['LR'])
        
        return
    

class RNNEncoderDecoder(nn.Module):
    """
    Class to house the recurrent encoder-decoder model
    
    Adapted from https://github.com/oriondollar/TransVAE 
    """
    def __init__(self, encoder, decoder, generator, src_embed, trg_embed, params):
        super().__init__()
        
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        return
    
    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        mem, mu, logvar = self.encode(src)
        x, h = self.decode(tgt, mem)
        x = self.generator(x)
        return x, mu, logvar

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, tgt, mem):
        return self.decoder(self.trg_embed(tgt), mem)

            
class RNNAttnEncoder(nn.Module):
    """
    Class to encode the embedded molecular structures into the latent space
    
    Adapted from https://github.com/oriondollar/TransVAE
    """
    def __init__(self, d_model, d_latent, N, dropout, src_length, device):
        super().__init__()
        
        self.size = d_model
        self.n_layers = N
        self.max_length = src_length + 1
        self.device = device
        
        self.gru = nn.GRU(self.size, self.size, num_layers = N, dropout = dropout)
        self.attn = nn.Linear(self.size*2, self.max_length)
        self.conv_bottleneck = ConvBottleneck(self.size)
        self.z_means = nn.Linear(1600, d_latent) #changes with max sequence length
        self.z_var = nn.Linear(1600, d_latent) #changes with max sequence length
        self.dropout = nn.Dropout(p = dropout)
        self.norm = LayerNorm(self.size)
        
        return
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, return_attn=False):
        h = self.initH(x.shape[0])
        x = x.permute(1, 0, 2)
        x_out, h = self.gru(x, h)
        x = x.permute(1, 0, 2)
        x_out = x_out.permute(1, 0, 2)
        mem = self.norm(x_out)
#         print('x:\t', x.size())
#         print('x_out:\t', x_out.size())
#         print('h:\t', h.size())
        
        attn_weights = F.softmax(self.attn(torch.cat((x, mem), 2)), dim=2)
#         print('attn_weights:\t', attn_weights.size())
#         print('mem:\t', mem.size())
        attn_applied = torch.bmm(attn_weights, mem)
        mem = F.relu(attn_applied)
        
        mem = mem.permute(0, 2, 1)
        mem = self.conv_bottleneck(mem)
#         print('mem size, post-conv:\t', mem.size())
        mem = mem.contiguous().view(mem.size(0), -1)
#         print(mem.size())
        mu, logvar = self.z_means(mem), self.z_var(mem)
        mem = self.reparameterize(mu, logvar)
        
        if return_attn:
            return mem, mu, logvar, attn_weights.detach().cpu()
        else:
            return mem, mu, logvar

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device)



class RNNAttnDecoder(nn.Module):
    """
    Class to decode the latent-space molecular structures into the token space
    
    Adapted from https://github.com/oriondollar/TransVAE
    """
    def __init__(self, d_model, d_latent, N, dropout, teacher_force, device):
        super().__init__()
        
        self.size = d_model
        self.n_layers = N
        self.teacher_force = teacher_force
        self.device = device
        
        if self.teacher_force:
            self.gru_size = self.size * 2
        else:
            self.gru_size = self.size
        
        self.linear = nn.Linear(d_latent, 1600) #changes with max sequence length
        self.deconv_bottleneck = DeconvBottleneck(self.size)
        self.dropout = nn.Dropout(p = dropout)
        self.gru = nn.GRU(self.gru_size, self.size, num_layers = N, dropout = dropout)
        self.norm = LayerNorm(self.size)
        
        return
    
    def forward(self, tgt, mem):
#         print('tgt size:\t', tgt.size())
        embedded = self.dropout(tgt)
#         print('embedded size:\t\t', embedded.size())
        h = self.initH(mem.shape[0])
        
        mem = F.relu(self.linear(mem))
        mem = mem.contiguous().view(-1, 64, 25) #changes with max sequence length
#         print('mem size, pre-deconv:\t', mem.size())
        mem = self.deconv_bottleneck(mem)
#         print('mem size, deconv:\t', mem.size())
        mem = mem.permute(0, 2, 1)
#         print('mem size, permute:\t', mem.size())
        mem = self.norm(mem)
#         print('mem size, norm:\t\t', mem.size())
        mem = mem[:,:-1,:]
        
#         print('mem size, pre-concat:\t', mem.size())
        if self.teacher_force:
            mem = torch.cat((embedded, mem), dim=2)
        
        mem = mem.permute(1, 0, 2)
        mem = mem.contiguous()
        x, h = self.gru(mem, h)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return x, h

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device).float()


# class RNNAttnPropPred():
#     """
#     Class to load and use the trained encoder and latent-space of the VAE
    
#     MAY NOT BE NECESSARY
#     """
    
    