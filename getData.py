#%%
import numpy as np
import pandas as pd
import nltk
import h5py
import gc
import sys
import smilesG as G
import os
import time
#import multiprocessing
import concurrent.futures

from functools import partial

from pathlib import Path

stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
from keras.utils import Sequence
sys.stdout=stdout

MAX_LEN=300 #was 277
NCHARS = len(G.GCFG.productions())

productions = G.GCFG.productions()

#%%
prod_map = {}
for ix, prod in enumerate(productions):
    prod_map[prod] = ix

tokenize = G.get_zinc_tokenizer()
parser = nltk.ChartParser(G.GCFG)

lhs_map = {}
for ix, lhs in enumerate(G.lhs_list):
    lhs_map[lhs] = ix

def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''

#%%
def checkCompat(smi):
    t = tokenize(smi)
    ok = 0
    try:
        pt = next(parser.parse(t))
        ml = len(pt.productions())
        if ml > MAX_LEN:
            ok = ml
    except ValueError:
        ok = 1
    return ok

def filterOK(smiles):
    good=[]
    bad=[]
    i = 0
    for smi in smiles:
        i+=1
        chk = checkCompat(smi)
        if chk==0:
            good.append(smi)
        else:
            bad.append(smi)
            if chk==1:
                print(i,'Bad: ',smi,flush=True)
            else:
                print(i,'L: ',chk,smi,flush=True)
    return good,bad

#%%

def getML(smiles):
    """ Check MAX_LEN """
    assert type(smiles) == list
    ML=0
    for i in range(0, len(smiles), 100):
        L=smiles[i:i+100]
        tokens = list(map(tokenize, L))
        parse_trees=[]
        for t in tokens:
            try:
                pt = next(parser.parse(t))
                parse_trees.append(pt)
            except:
                print('Ignore Bad: ',t)
        tmp = max([len(tree.productions()) for tree in parse_trees])
        ML = max(tmp,ML)
    return ML

#%%

def OneHot2Smiles(OH):
    prod_seq = [[productions[OH[index,t].argmax()]
                 for t in range(OH.shape[1])]
                for index in range(OH.shape[0])]
    smiles = [prods_to_eq(prods) for prods in prod_seq]
    return smiles

def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
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
def grammarFilter(smiles):
    assert type(smiles) == list
    good = []
    bad = []
    for smi in smiles:
        try:
            OH = to_one_hot([smi])
            good.append(smi)
        except:
            OH = None
            bad.append(smi)
        del OH
    return good, bad

def getSmi(fn):
    f = open(fn+'.smi','r')
    L = []
    for line in f:
        line = line.strip()
        L.append(line)
    f.close()
    return L

def saveSmi(L,fn):
    f = open(fn+'.smi','w')
    for line in L:
        f.write(line)
    f.close()


def _createData(fn):
    L = getSmi(fn)
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
def getData(fn,clr=False):
    if clr:
        if os.path.isfile(fn+'.h5'):
            os.remove(fn+'.h5')
    if os.path.isfile(fn+'.h5'):
        data = _readData(fn)
    else:
        data = _createData(fn)
    return data

#%%
def getZnSubset(fn,k):
    gc.collect()
    if os.path.isfile(fn+'.h5'):
        data = _readData(fn)
    else:
        data = _readData('data/250k_rndm_zinc_drugs_clean')
        data = data[0:k]
        gc.collect()
        h5f = h5py.File(fn+'.h5','w')
        h5f.create_dataset('data', data=data)
        h5f.close()
    return data

def clearCache(pth):
    for p in Path(pth).glob("*.npy"):
        p.unlink()


def saveOne(idx,pth):
    (rid,s) = idx
    outp={}
    print(rid,s)
    try:
        oh = to_one_hot([s])
        outp[rid]=s
        fn=pth+rid
        np.save(fn,oh)
    except:
        print('Failed:',rid,s)
    return outp


def makeCache(smifile,pth,count=None,verbose=True):
    smi = getSmi(pth+smifile)
    if count is None:
        count = len(smi)
    else:
        assert count>0 and count<=len(smi)
    smi = smi[0:count]
    ids = ['ID'+str(i) for i in range(count)]
    idx = dict(zip(ids,smi))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        outp = executor.map(partial(saveOne,pth=pth),idx.items())
    result={}
    for d in outp:
        result.update(d)
    outp = result
    np.save(pth+'idx',outp)
    return outp

def getDatAtID(pth,ID):
        X = np.load(pth + ID + '.npy')
        return X

def makeZnCache(pth = 'C:/DatCache/250kZinc/',verbose=True):
    fn = 'data/250k_rndm_zinc_drugs_clean'
    idx = makeCache(fn,pth,count=None,verbose=verbose)
    return idx

def getCacheIDX(pth = 'C:/DatCache/250kZinc/'):
    idx = np.load(pth+'idx.npy')
    return idx.item()

def getPlatform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    return platforms[sys.platform]


#%%
class ZincDataGen(Sequence):

    def __init__(self,idList,batchSz,dim,shuffle=True, pth = 'C:/DatCache/250kZinc/'):
        self.dim = dim
        self.batchSz = batchSz
        self.idList = idList
        self.shuffle = shuffle
        self.pth = pth
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.idList))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
       return int(np.floor(len(self.idList) / self.batchSz))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batchSz:(index+1)*self.batchSz]
        # Find list of IDs
        tmpIDs = [self.idList[k] for k in indices]
        # Generate data
        X = self.__data_generation(tmpIDs)
        return X, X

    def __data_generation(self, tmpIDs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batchSz, *self.dim))

        # Generate data
        for i, ID in enumerate(tmpIDs):
            # Store sample
            X[i,] = np.load(self.pth + ID + '.npy')
        return X

    def getDatAtID(self,ID):
        X = np.load(self.pth + ID + '.npy')
        return X

#%%
if __name__ == "__main__":
#%%
    if getPlatform() == 'Windows':
        #pth = 'D:/DatCache/500kZinc/' #Windows
        pth = 'C:/DatCache/test/'
    else:
        pth = '/DATA/SGODATA/Dat4GVAE/test/' #Linux
        #pth = "/DATA/SGODATA/Dat4GVAE/1MZinc/"
        #pth = "/DATA/SGODATA/Dat4GVAE/test/"

     
#%%
    clearCache(pth)

    fn = "Zn500k"
    #fn = '1MZinc'

    t0 = time.time()

    #idx = makeCache(fn,pth,count=None)
    idx = makeCache(fn,pth,count=1000)
    
    t1 = time.time() - t0
    
    print("Time: ", t1)
#%%
    del idx

    idx = getCacheIDX(pth)
#%%
    #for x in  idx.items():
    #    print(x[0],x[1])





