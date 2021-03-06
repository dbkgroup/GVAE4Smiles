#%%
import numpy as np
import nltk
import h5py
import gc
import sys
import smilesG as G
import os
import concurrent.futures as cf

MAX_LEN=300 #was 277
NCHARS = len(G.GCFG.productions())

productions = G.GCFG.productions()

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
            print(i,"; ",smi)
        else:
            bad.append(smi)
            if chk==1:
                print(i,'Bad: ',smi,flush=True)
            else:
                print(i,'L: ',chk,smi,flush=True)
    return good,bad

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

def OneHot2Smiles(OH):
    prod_seq = [[productions[OH[index,t].argmax()]
                 for t in range(OH.shape[1])]
                for index in range(OH.shape[0])]
    smiles = [prods_to_eq(prods) for prods in prod_seq]
    return smiles

def doSmi2OH(smi):
    token = tokenize(smi)
    parser = nltk.ChartParser(G.GCFG)
    tree = next(parser.parse(token))
    prod_seq = tree.productions()
    indices = np.array([prod_map[prod] for prod in prod_seq], dtype=int)
    oh = np.zeros((1, MAX_LEN, NCHARS), dtype=np.float32)
    num_productions = len(indices)
    oh[0][np.arange(num_productions),indices] = 1.
    oh[0][np.arange(num_productions, MAX_LEN),-1] = 1.
    return oh

def s2oh(smi):
    n = len(smi)
    oh = np.zeros((n, MAX_LEN, NCHARS), dtype=np.float32)
    with cf.ProcessPoolExecutor() as e:
        result = e.map(doSmi2OH,smi)
    for i,x in enumerate(result):
        oh[i,:,:] = x
    return oh

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

def _readData(fn):
    h5f = h5py.File(fn+'.h5', 'r')
    data = h5f['data'][:]
    h5f.close()
    return data

def getData(fn,clr=False):
    if clr:
        if os.path.isfile(fn+'.h5'):
            os.remove(fn+'.h5')
    if os.path.isfile(fn+'.h5'):
        data = _readData(fn)
    else:
        data = _createData(fn)
    return data

def getZnSubset(fn,k):
    gc.collect()
    if os.path.isfile(fn+'.h5'):
        data = _readData(fn)
    else:
        data = _readData('data/250kZinc')
        data = data[0:k]
        gc.collect()
        h5f = h5py.File(fn+'.h5','w')
        h5f.create_dataset('data', data=data)
        h5f.close()
    return data


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


def getFileList(pth):
    fn = []
    for root, _directories, files in os.walk(pth):
        for filename in files:
            if filename.endswith(".npy") or filename.endswith(".smi"):
                fp = os.path.join(root,filename)
                fn.append(fp)
    return fn

#%%
if __name__ == "__main__":
    from time import time

    smi = getSmi('data/test/500kZinc')
    smi = smi[0:1000]

    t0 = time()
    oh = to_one_hot(smi)
    t1 = time() - t0
    print('Time:',t1)

    t0 = time()
    oh = s2oh(smi)
    t1 = time() - t0
    print('Time:',t1)














