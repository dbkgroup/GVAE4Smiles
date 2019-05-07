# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:06:24 2018

@author: Steve O'Hagan

Based off: https://github.com/kanojikajino/grammarVAE
Paper: https://arxiv.org/abs/1703.01925

"""

from time import time
import numpy as np
import pandas as pd
import h5py

from generateData import H5DataGenV2

from buildGVAE import buildALL, plotm, testPerformance, doFitG, loadWeights, \
    encodeFilter, to_one_hot, doFit, MAX_LEN, DIM

#%%
if __name__ == "__main__":

    pth = 'data/6MZincHV2/' #for data cache
    k = 20000
    gpus = 1
    LATENT = 56
    nCNV = 3
    nDNS = 3

    #for model save
    fn = f'data/{k//1000}k6MZincL{LATENT}nCNV{nCNV}nDNS{nDNS}_{int(time())}'
    print('Path:',pth)
    print('Model save:',fn)

    with h5py.File(pth+'cache.h5','r') as hf:
        dset = hf['default']
        n = np.shape(dset)[0]
    indices = np.arange(n)
    #take training set from front of data
    idTrain = indices[0:k]
    #take validation & test set from end of data
    idValid = indices[-6000:-2000]
    idTest = indices[-2000:]
    batch = 325 * gpus
    genr = H5DataGenV2(idTrain,batch,(MAX_LEN,DIM),pth=pth)
    vgenr = H5DataGenV2(idValid,batch,(MAX_LEN,DIM),pth=pth)
    tstgen = H5DataGenV2(idTest,2000,(MAX_LEN,DIM),pth=pth)
    XTE, _tmp =  tstgen.__getitem__(0)
    del _tmp, tstgen

    wf = 'data/3000k6MZinc_L56_E60_val.hdf5'

    enc, dec, aen, emv, mgm = buildALL(LATENT,nCNV,nDNS,gpus,wFile=wf)

    plotm(aen)

    p,g = testPerformance(XTE,dec,emv)
    print(f'Zinc: {p:.2f} {g:.2f}')

    doFitG(aen,genr,vgenr,fn,EPOCHS = 2)
    loadWeights(enc,dec,aen,emv,wFile=fn+'.hdf5')

    p,g = testPerformance(XTE,dec,emv)
    print(f'Zinc2: {p:.2f} {g:.2f}')

    SMDat = pd.read_excel('data/Smiles.xlsx')

    SMList = list(SMDat['SMILES'])

    encoded, good, bad = encodeFilter(SMList,emv)

    np.random.shuffle(good)
    n = len(good)

    good = to_one_hot(good)

    k = n * 9 // 10

    trn = good[0:k,:,:]
    tst = good[k:,:,:]

    doFit(aen,trn,fn,bat=batch, EPOCHS = 60)

    #load best weights
    loadWeights(enc,dec,aen,emv,wFile=fn+'.hdf5')

    print(f'Zinc: {p:.2f} {g:.2f}')

    p,g = testPerformance(tst,dec,emv)
    print(f'Test: {p:.2f} {g:.2f}')

    p,g = testPerformance(trn,dec,emv)
    print(f'Train: {p:.2f} {g:.2f}')

    p,g = testPerformance(XTE,dec,emv)
    print(f'Zinc after: {p:.2f} {g:.2f}')


