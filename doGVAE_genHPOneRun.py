# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:06:24 2018

@author: Steve O'Hagan

Based off: https://github.com/kanojikajino/grammarVAE
Paper: https://arxiv.org/abs/1703.01925

"""

from generateData import H5DataGenV2

from GVAE import smilesGVAE, plotm

import numpy as np
import h5py

#%%
if __name__ == "__main__":

    pth = 'data/6MZincHV2/' #for data cache
    k = 500000

    with h5py.File(pth+'cache.h5','r') as hf:
        dset = hf['default']
        n = np.shape(dset)[0]
    indices = np.arange(n)
    #take training set from front of data
    idTrain = indices[0:k]
    #take validation & test set from end of data
    idValid = indices[-6000:-2000]
    idTest = indices[-2000:]

    gpus = 4
    batch = 312 * gpus

    genr = H5DataGenV2(idTrain,batch,pth=pth)
    vgenr = H5DataGenV2(idValid,batch,pth=pth)
    tstgen = H5DataGenV2(idTest,2000,pth=pth)
    
    XTE, _tmp =  tstgen.__getitem__(0)
    del _tmp, tstgen

    params  = {
        'LATENT':56, 
        'nC':1, 
        'nD':1, 
        'beta':1.0, 
        'gruf':256,
        'ngpu':gpus,
        'opt':'adam',
        'wFile':None
    }
        
    EPO = 1
    
    fn = f'testA'

    sgv = smilesGVAE(**params)

    sgv.doFitG(genr,vgenr,fn, EPOCHS = EPO)

    sgv.loadWeights(fn+'.hdf5')

    sgv.jaccScore(XTE)

    x1 = XTE[0:1,:,:]

    sgv.listSamples(x1)

    sgv.listMeans(x1)

    sgv.testPerformance(XTE)

    

