import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm.notebook import tqdm


SMILES_CHARS = ['*', ' ', '#', '%', '(', ')', '+', '-', '.', '/', '=', '@', '[', '\\', ']',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'Br', 'Cl', 'F', 'H', 'C', 'I', 'K', 'N', 'O', 'P', 'S',
                'br', 'cl', 'f', 'h', 'c', 'i', 'k', 'n', 'o', 'p', 's']

    
class LogPChemDataset(torch.utils.data.Dataset):
    """
    This iteration of the dataset includes molecular weight in the training data and
    supplies logP as the target property for prediction
    """
    def __init__(self, X, Y):
        super(ChemDataset).__init__()
        
        self.encoder = SmilesEncoder()
        
        self.smiles = X['Smiles']
        
        self.weights = X['Molecular Weight'].replace('None', np.nan)
        self.weights = self.weights.astype('float')
        
        self.logP = Y
        
        return
        
        
    def __len__(self):
        return len(self.logP)
    
    
    def __getitem__(self, key):
        smiles_tensor = self.encoder.encode(self.smiles.iloc[key])
#         smiles_tensor.double()
        # print(smiles_tensor.shape)
        # smiles_tensor = smiles_tensor.reshape((-1, 10))
        
        weight_array = np.array(self.weights.iloc[key])
        weight_tensor = torch.from_numpy(weight_array)
#         weight_tensor.double()
        weight_tensor = weight_tensor.reshape(-1, 1)
        
        logP_array = np.array(self.logP.iloc[key])
        logP_tensor = torch.from_numpy(logP_array)
#         logP_tensor.double()
        logP_tensor = logP_tensor.reshape(-1, 1)
        
        return smiles_tensor, weight_tensor, logP_tensor
    
    
class VAEChemDataset(torch.utils.data.Dataset):
    """
    This iteration of ChemDataset only supplies SMILES strings for VAE training
    """
    def __init__(self, X, CHAR_DICT = None, max_length = 250):
        super(VAEChemDataset).__init__()
        
        self.encoder = SmilesEncoder(CHAR_DICT, max_length = max_length)
        self.smiles = X
        return
        
        
    def __len__(self):
        return len(self.smiles)
    
    
    def __getitem__(self, key):
        try:
            smiles_tensor = self.encoder.encode(self.smiles.iloc[key])
        except:
#             smiles_tensor = self.encoder.encode(self.smiles.iloc[key])
#             print(smiles_tensor)
            print(key, self.smiles.iloc[key])
        
        return smiles_tensor
    
    
    
class SmilesEncoder():
    """
    Uses either a dictionary mapping or the sklearn implementation of one-hot-encoding
    to convert SMILES tokens into int indexes or one-hot-encoded tensors
    """
    
    def __init__(self, CHAR_DICT = None, max_length = 250, simple = False, one_hot = False):
        super(SmilesEncoder).__init__()
        
        self.SMILES_CHARS = SMILES_CHARS
        self.CHAR_DICT = CHAR_DICT
        self.one_hot = one_hot
        self.simple = simple
        self.max_length = max_length
        
        if self.one_hot:
            self.hot_encoder = OneHotEncoder(sparse = False)
            self.hot_encoder.fit(self.SMILES_CHARS)
            
        elif simple:
            self.smiles_indx = [self.SMILES_CHARS.index(x) for x in self.SMILES_CHARS]

            self.smiles2ind = {}
            self.ind2smiles = {}

            for sm, ix in zip(self.SMILES_CHARS, self.smiles_indx):
                self.smiles2ind[sm] = ix
                self.ind2smiles[ix] = sm

#             print(self.smiles2ind)
        else:
            assert self.CHAR_DICT != None, ('Supply CHAR_DICT')
            self.tokenizer = SmilesTokenizer(self.CHAR_DICT)

        return
    
    
    def encode(self, text):
        padded = self.pad(text, self.max_length)
        tokenized, index_list = self.tokenize(padded)
        
        if len(index_list) < self.max_length:
            for i in range(self.max_length-len(index_list)):
                tokenized.append('*')
                index_list.append(0)
                
        if len(index_list) > self.max_length:
            #drop extras, but keep the '<EOS>' token
            tokenized = tokenized[:self.max_length]
            index_list = index_list[:self.max_length]
            
        tokenized.append('<EOS>')
        index_list.append(self.CHAR_DICT['<EOS>'])
        
        if self.one_hot:
            token_array = np.array(tokenized)
            encoded_array = self.hot_encoder.transform(token_array)
            smiles_tensor = torch.tensor(encoded_array, dtype = torch.double)
            
            return smiles_tensor
            
        elif self.simple:
            index_list = [int(self.smiles2ind[x]) for x in tokenized]
            token_array = np.array(index_list)
            
            return token_array
            
        else:
            index_array = np.array(index_list)
            return index_array
    
    
    def decode(self, encoded_smiles):
        #TODO: add compatibility with new dictionary-based self.encode option
        
        encoded_array = encoded_smiles.numpy()
        decoded_array = self.hot_encoder.inverse_transform(encoded_array)
        decoded_array = decoded_array.reshape(1, -1)
        decoded_list = decoded_array[0].tolist()
        smiles_string = ''.join(decoded_list)
        
        return smiles_string
    
    
#     def tokenize(self, smile_str):
# #         pattern =  "(\[[^\]]+]|Br?|Cl?|Ca?|Na?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
# #         pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"
#         pattern = "(\[[^\]]+]|Br?|Cl?|Ca?|Na?|Li?|Mg?|Si?|Se?|B|K|N|O|S|P|F|I|H|b|c|n|o|s|p|\\[|\\]|\[[|\]]|\[|\]|\(|\)|\.|=|#|-|\+|\+|\\\\|\/|_|:|~|@|@|@@|@@|\?|>|\*|\$|\%|\d{3}|\d{2}|\d{1})"
#         reg = re.compile(pattern)
#         tokens = [token for token in reg.findall(smile_str)]
        
#         assert smile_str == ''.join(tokens), ("{} could not be joined".format(smile_str))
#         return tokens

    def tokenize(self, smiles_str):
        tokenized, index_list = self.tokenizer.tokenize(smiles_str)
        return tokenized, index_list
        
    
    
    def pad(self, smiles_str, max_length):
        if len(smiles_str) < max_length:
            pad_len = max_length - len(smiles_str)
            pad_str = ''
            
            for i in range(pad_len):
                pad_str = pad_str + '*'
            
            padded_str = smiles_str + pad_str
            
        else:
            padded_str = smiles_str
            
        return padded_str
    
    
    def get_char_weights(self, train_smiles, params, freq_penalty=0.5):
        """
        Calculates token weights for a set of input data
        """
        char_dist = {}
        char_counts = np.zeros((params['NUM_CHAR'],))
        char_weights = np.zeros((params['NUM_CHAR'],))
        
        for k in params['CHAR_DICT'].keys():
            char_dist[k] = 0
        for smile in train_smiles:
#             print(train_smiles[0])
            for i, char in enumerate(smile):
#                 print(char)
                char_dist[char] += 1
            for j in range(i, params['MAX_LENGTH']):
                char_dist['*'] += 1
        
        for i, v in enumerate(char_dist.values()):
            char_counts[i] = v
        
        top = np.sum(np.log(char_counts))
        
        for i in range(char_counts.shape[0]):
            if char_counts[i] == 1:
                #need to avoid log(1) = 0
                char_counts[i] = 2
            char_weights[i] = top / np.log(char_counts[i])
        
        min_weight = char_weights.min()
        
        for i, w in enumerate(char_weights):
            if w > 2*min_weight:
                char_weights[i] = 2*min_weight
        
        scaler = MinMaxScaler([freq_penalty,1.0])
        char_weights = scaler.fit_transform(char_weights.reshape(-1, 1))
        return char_weights[:,0]
    
    
class SmilesTokenizer():
    def __init__(self, char_dict):
        super().__init__()
        ELEMENT_SYMBOLS = [#uncommon atoms that can be construed as two atoms (e.g. CS, NO) are excluded
            'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'Ba', 'Be', 'Bh', 'Bi', 'Bk', 'Br', 'B', 'Ca', 'Cd', 'Ce',
            'Cl', 'Cm', 'Cr', 'Cu', 'C', 'Db', 'Ds', 'Dy', 'Er', 'Es', 'Eu', 'Fe', 'Fl', 'Fm', 'Fr', 'F', 'Ga', 'Gd',
            'Ge', 'He', 'Hg', 'H', 'In', 'Ir', 'I', 'Kr', 'K', 'La', 'Li', 'Lr', 'Lu', 'Lv', 'Md', 'Mg', 'Mn', 'Mo',
            'Mt', 'Na', 'Nb', 'Nd', 'Ne', 'Ni', 'N', 'O', 'Pa', 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'P', 'Ra',
            'Rb', 'Re', 'Rf', 'Rg', 'Rh', 'Rn', 'Ru', 'Sb', 'Se', 'Sg', 'Si', 'Sm', 'Sr', 'S', 'Ta', 'Tb', 'Tc', 'Te',
            'Th', 'Ti', 'Tl', 'Tm', 'Uuo', 'Uup', 'Uus', 'Uut', 'U', 'V', 'W', 'Xe', 'Yb', 'Y', 'Zn', 'Zr'
            ]

        BRACKET_SYMBOLS = [
            '\(', '\)', '\[', '\]', '\{', '\}'
        ]

        NUMBERS = '(\d{3}|\d{2}|\d{1})'

        SMILES_SYMBOLS = [
            '\.', '=', '#', '-', '\+', '\+', '\\\\', '\/', '_', ':', '~', '@@', '@@', '@', '@', '\?', '>', '\*', '\$', '\%'
        ]

        self.element_re = re.compile('|'.join(ELEMENT_SYMBOLS), flags = re.I)
        self.number_re = re.compile(NUMBERS)
        self.smiles_re = re.compile('|'.join(SMILES_SYMBOLS))
        self.bracket_re = re.compile('|'.join(BRACKET_SYMBOLS))
        
        self.char_dict = char_dict
        
        return
    
    
    def match_brackets(self, string):
        matches = []
        for m in self.bracket_re.finditer(string):
            match_span = (m.start(), m.group())
            matches.append(match_span)
        return matches
    

    def match_atoms(self, string):
        matches = []
        for m in self.element_re.finditer(string):
            match_span = (m.start(), m.group())
            matches.append(match_span)
        return matches
    

    def match_smiles_symbols(self, string):
        matches = []
        for m in self.smiles_re.finditer(string):
            match_span = (m.start(), m.group())
            matches.append(match_span)
        return matches
    

    def match_numbers(self, string):
        matches = []
        for m in self.number_re.finditer(string):
            match_span = (m.start(), m.group())
            matches.append(match_span)
        return matches
    
    
    def build_char_dict(self, smiles_strings):
        char_dict = {}
        char_set = []
        for smiles in tqdm(smiles_strings, total = len(smiles_strings)):
            brackets = self.match_brackets(smiles)
            atoms = self.match_atoms(smiles)
            symbols = self.match_smiles_symbols(smiles)
            numbers = self.match_numbers(smiles)

            span_list = brackets+atoms+symbols+numbers
            span_list = [tup[1] for tup in span_list]
#             span_set = list(set(span_list))

            char_set.extend(span_list)
        char_set = list(set(char_set))
        char_set.sort()

        try:
            #need to ensure that the padding token is at index 0
            pad_idx = char_set.index('*')
            del char_set[pad_idx]
            char_set.insert(0, '*')
        except:
            char_set.insert(0, '*')
            
#         char_set.append('<EOS>')

        for i, char in enumerate(char_set):
            char_dict[char] = i

        return char_dict
    
    
    def reconstruct_smiles_from_re(self, original, brackets, atoms, symbols, numbers):
        smiles_len = len(original)
        reconstructed = ''

        for i in range(smiles_len):
            reconstructed += 'z'

        for br in brackets:
            ind = br[0]
            tok = br[1]
            if len(tok) > 1:
                tok_num = len(tok)
            else:
                tok_num = 1
            for i in range(tok_num):
                reconstructed = reconstructed[:ind+i] + tok[i]  + reconstructed[ind+i+1:]

        for at in atoms:
            ind = at[0]
            tok = at[1]
            if len(tok) > 1:
                tok_num = len(tok)
            else:
                tok_num = 1
            for i in range(tok_num):
                reconstructed = reconstructed[:ind+i] + tok[i]  + reconstructed[ind+i+1:]

        for sy in symbols:
            ind = sy[0]
            tok = sy[1]
            if len(tok) > 1:
                tok_num = len(tok)
            else:
                tok_num = 1
            for i in range(tok_num):
                reconstructed = reconstructed[:ind+i] + tok[i]  + reconstructed[ind+i+1:]

        for num in numbers:
            ind = num[0]
            tok = num[1]
            if len(tok) > 1:
                tok_num = len(tok)
            else:
                tok_num = 1
            for i in range(tok_num):
                reconstructed = reconstructed[:ind+i] + tok[i]  + reconstructed[ind+i+1:]

        assert len(reconstructed) == len(original), ('smiles error: length mismatch between original and reconstructed')
        
        return reconstructed
    
    
    def spans_to_index_list(self, brackets, atoms, symbols, numbers):
        spans = brackets + atoms + symbols + numbers

        sorted_spans = sorted(spans, key = lambda spn: spn[0])
#         print(sorted_spans)

        index_list = []
        for span in sorted_spans:
            tok = span[1]
            ind = self.char_dict[tok]
            index_list.append(ind)

        return index_list
    
    
    def spans_to_tokenized(self, brackets, atoms, symbols, numbers):
        spans = brackets + atoms + symbols + numbers

        sorted_spans = sorted(spans, key = lambda spn: spn[0])
#         print(sorted_spans)

        token_list = []
        for span in sorted_spans:
            tok = span[1]
            token_list.append(tok)

        return token_list
    
    
    def tokenize(self, smiles):
        brackets = self.match_brackets(smiles)
        atoms = self.match_atoms(smiles)
        symbols = self.match_smiles_symbols(smiles)
        numbers = self.match_numbers(smiles)
        
        #assert statement in this func can catch some tokinzation errors
        reconstructed = self.reconstruct_smiles_from_re(smiles, brackets, atoms, symbols, numbers)
        
        token_list = self.spans_to_tokenized(brackets, atoms, symbols, numbers)
        index_list = self.spans_to_index_list(brackets, atoms, symbols, numbers)
        
        return token_list, index_list
    

#############################################
######## Data Cleaning & Preparation ########
#############################################

    
def df_Gaussian_normalize(dataframe):
    
    df = dataframe
    normed_df = pd.DataFrame()
    norm_key = {}

    shape = dataframe.shape

    if len(shape) == 1:
        colname = dataframe.name
        coldata = dataframe.iloc[:]
        stdev = coldata.std()
        mean = coldata.mean()

        normed_col = (coldata - mean) / stdev
        norm_key[colname] = [mean, stdev]

        return normed_col, norm_key

    if len(shape) > 1:
        for colname, coldata in df.iteritems():
            if colname == 'Smiles':
                normed_col = coldata
                stdev = 'string data column'
                mean = 'string data column'

            else:
                coldata = np.array(coldata)
                stdev = coldata.std()
                mean = coldata.mean()
            
                normed_col = (coldata - mean) / stdev

            normed_df[colname] = normed_col
            norm_key[colname] = [mean, stdev]
        
        return normed_df, norm_key


def df_Gaussian_denormalize(normed_df, norm_key):
  #THIS NEEDS TO BE UPDATED TO MATCH NORMALIZER FUNCTION
    
    denormed_df = pd.DataFrame()
    
    for colname, coldata in normed_df.iteritems():
        mean = norm_key[colname][0]
        std = norm_key[colname][1]
        
        denormed_col = (coldata * mean) + std
        
        denormed_df[colname] = denormed_col
        
    return denormed_df


def df_MinMax_normalize(dataframe):
    
    df = dataframe
    
    normed_df = pd.DataFrame()

    df_norm_key = {}

    if len(df.shape) == 1:
        colname = dataframe.name
        # print(colname)
        coldata = dataframe.iloc[:]
        min_val = coldata.min()
        max_val = coldata.max()

        normed_col = (coldata - min_val) / (max_val - min_val)
        df_norm_key[colname] = [min_val, max_val]

        return normed_col, df_norm_key

    if len(df.shape) > 1:
        for colname, coldata in df.iteritems():
            if colname == 'Smiles':
                normed_col = coldata
                min_val = 'string data column'
                max_val = 'string data column'

            else:
                max_val = coldata.max()
                min_val = coldata.min()

                normed_col = (coldata - min_val) / (max_val - min_val)

            df_norm_key[colname] = [min_val, max_val]
            normed_df[colname] = normed_col

        return normed_df, df_norm_key 


def df_MinMax_denormalize(normed_df, norm_key):
    
    denormed_df = pd.DataFrame()
    
    for colname, coldata in normed_df.iteritems():
        mn = norm_key[colname][0]
        mx = norm_key[colname][1]
        
        denormed_col = (coldata * (mx - mn)) + mn
        
        denormed_df[colname] = denormed_col
        
    return denormed_df


def subsequent_mask(size):
    """Mask out subsequent positions (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
        
    taken from https://github.com/oriondollar/TransVAE
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


##################################################
############## Plotting Utils ####################
##################################################

def plot_pairwise_correlation(token_tracker, mask = True):
    d = pd.DataFrame.from_dict(token_tracker, orient = 'index')
    d = d.transpose()
    # Compute the correlation matrix
    corr = d.corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    
    if mask:
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, cmap=cmap, mask = mask, square=True, cbar_kws={"shrink": .5})
    else:
        sns.heatmap(corr, cmap=cmap, square=True, cbar_kws={"shrink": .5})
    return


def plot_token_type_accuracy(token_tracker):
    #define plt.rcparams
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.serif'] = 'Arial'
    plt.rc('axes', labelsize = 18)
    plt.rc('xtick', labelsize = 12)
    plt.rc('ytick', labelsize = 16)
    plt.rc('legend', fontsize = 12)

    with open('../data/CHAR_DICT.json', 'r') as f:
    # with open('./CHAR_DICT.json', 'r') as f:
        char_dict = json.load(f)
        f.close()
        
    char_dict['<EOS>'] = 144
    chars = list(char_dict.keys())
    indices = list(char_dict.values())
    
    acc_dict = {}
    for k, v in token_tracker.items():
        acc_dict[k] = []
        for el in v:
            if el == k:
                acc_dict[k].append(1)
            else:
                acc_dict[k].append(0)
    
    accs = {}            
    for k, v in acc_dict.items():
        char = chars[indices.index(k)]
        if len(v) > 0:
            accs[char] = sum(v)/len(v)
        else:
            accs[char] = 0
    
    fig, ax = plt.subplots(figsize = (20, 10))
    plt.bar(range(len(accs)), list(accs.values()), align='center')
    plt.xticks(range(len(accs)), list(accs.keys()), rotation = 90)
    ax.set_xlabel('Token')
    ax.set_ylabel('Accuracy (decimal)')
    plt.title('Accuracy per Token', fontsize = 18)
    plt.show()
    
    return

def plot_token_type_accuracy_horizontal(token_tracker, last_ep_only = True):
    #define plt.rcparams
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.serif'] = 'Arial'
    plt.rc('axes', labelsize = 18)
    plt.rc('xtick', labelsize = 12)
    plt.rc('ytick', labelsize = 12)
    plt.rc('legend', fontsize = 12)

    with open('../data/CHAR_DICT.json', 'r') as f:
    # with open('./CHAR_DICT.json', 'r') as f:
        char_dict = json.load(f)
        f.close()
        
    char_dict['<EOS>'] = 144
    chars = list(char_dict.keys())
    indices = list(char_dict.values())
    
    acc_dict = {}
    if last_ep_only:
        last_epoch = list(token_tracker.keys())[-1]
        for k, v in token_tracker[last_epoch].items():
            acc_dict[k] = []
            for el in v:
                if el == k:
                    acc_dict[k].append(1)
                else:
                    acc_dict[k].append(0)
    else:
        for ep, ep_dict in token_tracker.items():
            for k, v in ep_dict.items():
                acc_dict[k] = []
                for el in v:
                    if el == k:
                        acc_dict[k].append(1)
                    else:
                        acc_dict[k].append(0)
    
    accs = {}            
    for k, v in acc_dict.items():
        char = chars[indices.index(k)]
        if len(v) > 0:
            accs[char] = sum(v)/len(v)
        else:
            accs[char] = 0
            
    y_pos = np.arange(len(accs.keys()))
    
    fig, ax = plt.subplots(figsize = (15, 35))
    plt.barh(list(accs.keys()), list(accs.values()), align = 'center')
    ax.set_yticks(y_pos)
#     ax.set_yticklabels(accs.keys())
    ax.set_ylabel('Token')
    ax.set_xlabel('Accuracy (decimal)')
    ax.set_xlim(0, 1.0)
    ax.axvline(x = 0.25, linestyle = '-.', c = 'k')
    ax.axvline(x = 0.50, linestyle = '-.', c = 'k')
    ax.axvline(x = 0.75, linestyle = '-.', c = 'k')
    plt.title('Accuracy per Token', fontsize = 18)
    plt.ylim(min(y_pos)-1, max(y_pos)+1)
    plt.show()
    
    return