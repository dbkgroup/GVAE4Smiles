import numpy as np
import nltk
import h5py

import smilesG as G
import os

MAX_LEN=277
NCHARS = len(G.GCFG.productions())


#%%
def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(G.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = G.get_zinc_tokenizer()
    tokens = list(map(tokenize, smiles))
    parser = nltk.ChartParser(G.GCFG)
    parse_trees = [next(parser.parse(t)) for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot

#%%
def _createData(fn):
    f = open(fn+'.smi','r')
    L = []
    for line in f:
        line = line.strip()
        L.append(line)
    f.close()
    data = np.zeros((len(L),MAX_LEN,NCHARS))
    for i in range(0, len(L), 100):
        print('Processing: i=[' + str(i) + ':' + str(i+100) + ']')
        onehot = to_one_hot(L[i:i+100])
        data[i:i+100,:,:] = onehot
    h5f = h5py.File(fn+'.h5','w')
    h5f.create_dataset('data', data=data)
    h5f.close()
    return data

#%%
def _readData(fn):
    h5f = h5py.File(fn+'.h5', 'r')
    data = h5f['data'][:]
    h5f.close()
    return data

#%%
def getData(fn='data/250k_rndm_zinc_drugs_clean',clr=False):
    if clr:
        if os.path.isfile(fn+'.h5'):
            os.remove(fn+'.h5')
    if os.path.isfile(fn+'.h5'):
        data = _readData(fn)
    else:
        data = _createData(fn)
    return data, fn
