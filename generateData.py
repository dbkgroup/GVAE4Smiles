import h5py
import numpy as np
from dataTools import getSmi, to_one_hot

from keras.utils import Sequence

from time import time

MAX_LEN = 300 #was 277
DIM = 80

def print_name(name, obj):
    if isinstance(obj, h5py.Dataset):
        print('Dataset:', name)
    elif isinstance(obj, h5py.Group):
        print('Group:', name)

def makeCacheHV(smifile,pth,cnt=None,verbose=True):
    t0 = time()
    smi = getSmi(pth+smifile)
    if cnt is None:
        cnt = len(smi)
    else:
        assert cnt>0 and cnt<=len(smi)
    smi = smi[0:cnt]
    outp = {}
    with h5py.File(pth+'cache.h5', 'w') as hf:
        dset = hf.create_dataset("default",(cnt,MAX_LEN,DIM),chunks=(1,MAX_LEN,DIM),compression="lzf",dtype='f4')
        for i,s in enumerate(smi):
            try:
                oh = to_one_hot([s])
                dset[i,:,:] = oh
                outp[i]=s
                if verbose:
                    print(i,s)
            except Exception as e:
                print('Failed:',i,s)
                print(e)
    np.savez_compressed(pth+'idx',outp)
    t1 = time() - t0
    print("Time:",t1)
    return outp

class H5DataGenV2(Sequence):
    """Generate data batches from cached h5py file."""

    def __init__(self,ind,batchSz, pth, shuffle=True):
        self.dim = (MAX_LEN, DIM)
        self.batchSz = batchSz
        self.ind = ind
        self.shuffle = shuffle
        self.pth = pth
        self.hfn = self.pth+'cache.h5'
        self.on_epoch_end()
        self.idx = self.getIDX()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.ind)

    def __len__(self):
       return int(np.floor(len(self.ind) / self.batchSz))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.ind[index*self.batchSz:(index+1)*self.batchSz]
        # Generate data
        X = self.__data_generation(indices)
        return X, X

    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batchSz, *self.dim),dtype='float32')

        with h5py.File(self.hfn,'r') as hf:
            dset = hf['default']
            # Generate data
            for i, ID in enumerate(indices):
                x = dset[ID,:,:]
                x = np.array(x)
                # Store sample
                X[i,] = x
        return X

    def getSmiBatch(self,index):
        indices = self.ind[index*self.batchSz:(index+1)*self.batchSz]
        return [self.idx[k] for k in indices]


    def getIDX(self):
        with np.load(self.pth+'idx.npz') as npl:
            idx = npl['arr_0'].item()
        return idx

    def getDatAtID(self,ID):
        with h5py.File(self.hfn,'r') as hf:
            dset = hf['default']
            x = dset[ID,:,:]
            x = np.array(x)
        return x


def testHVGen(cnt=5000):
    pth = 'data/6MZincHV2/'
    with h5py.File(pth+'cache.h5','r') as hf:
        dset=hf['default']
        n = dset.shape[0]
    print('Cache:',n)
    indices = np.arange(n)
    np.random.shuffle(indices)
    ind = indices[0:cnt]
    batch = 200
    genr = H5DataGenV2(ind,batch,pth=pth)
    nc = len(genr)

    t0 = time()
    # 1k 65; 5k 341; 10k 675; 25k 1673
    for i in range(nc):
        x = genr.__getitem__(i)[0]
        print(i)
    t1 = time() - t0
    print('Time:',t1)
    print(np.shape(x),x.dtype)



#%%
if __name__ == '__main__':
#%%

    pth = 'data/6MZincHV2/'

    # time: 351186 = 4.06 days
    #makeCacheHV('6MZinc',pth)

    with h5py.File(pth+'cache.h5', 'r') as hf:
        hf.visititems(print_name)

    # time: 4.79
    testHVGen(10000)
