import numpy as np
import nltk
import h5py
import gc
import sys
import smilesG as G
import os
stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
import keras
sys.stdout=stdout

MAX_LEN=300 #was 277
NCHARS = len(G.GCFG.productions())

#%%
prod_map = {}
for ix, prod in enumerate(G.GCFG.productions()):
    prod_map[prod] = ix
tokenize = G.get_zinc_tokenizer()
parser = nltk.ChartParser(G.GCFG)

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

def makeZnCache(pth = 'C:/DatCache/250kZinc/',verbose=True):
    fn = 'data/250k_rndm_zinc_drugs_clean'
    smi = getSmi(fn)
    idx = {}
    for i,s in enumerate(smi):
        rid = 'ID'+str(i)
        if verbose:
            print(rid,s)
        idx[rid]=s
        oh = to_one_hot([s])
        fn=pth+rid
        np.save(fn,oh)
    np.save(pth+'idx',idx)
    return idx

#%%
def getZnIDX(pth = 'C:/DatCache/250kZinc/'):
    idx = np.load(pth+'idx.npy')
    return idx.item()

#%%
class ZincDataGen(keras.utils.Sequence):

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

#%%
if __name__ == "__main__":

    idx = getZnIDX()

    idList = list(idx.keys())

    tid = idList[0]

    s = 'C:/DatCache/250kZinc/' + tid + '.npy'

    tmp = np.load(s)

    generator = ZincDataGen(idList,128,(MAX_LEN,NCHARS))

    tmp,t1 = generator.__getitem__(0)

    pass




