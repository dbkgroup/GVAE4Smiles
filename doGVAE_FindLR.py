# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:06:24 2018

@author: Steve O'Hagan

Based off: https://github.com/kanojikajino/grammarVAE
Paper: https://arxiv.org/abs/1703.01925

"""

from time import time

import numpy as np
import h5py

from generateData import H5DataGenV2

from GVAE import smilesGVAE

from findLR_CLR import LRFinder

if __name__ == "__main__":

    k = 25000
    gpus = 1
    pth = 'data/6MZincHV2/' #for data
    fn = f'data/FindLR_{int(time())}' #for model save
    print('Path:',pth)
    print('Model save:',fn)
    with h5py.File(pth+'cache.h5','r') as hf:
        dset = hf['default']
        n = np.shape(dset)[0]
    indices = np.arange(n)
    idTrain = indices[0:k]
    idValid = indices[k:k+4000]
    idTest = indices[k+4000:k+6000]
    batch = 325 * gpus
    genr = H5DataGenV2(idTrain,batch,pth=pth)
    vgenr = H5DataGenV2(idValid,batch,pth=pth)
    tstgen = H5DataGenV2(idTest,2000,pth=pth)
    XTE, _tmp =  tstgen.__getitem__(0)
    del _tmp, tstgen

    params  = {
        'LATENT':56,
        'nC':3,
        'nD':3,
        'beta':1.0,
        'gruf':501,
        'ngpu':gpus,
        'opt':'adam',
        'wFile':None
    }

    EPO = 2

    lr_finder = LRFinder(min_lr=2.0e-4, max_lr=1.0e-2,
                     steps_per_epoch=np.ceil(k/batch), epochs=EPO)

    sgv = smilesGVAE(**params)

    print('Training autoencoder.')
    if sgv.mgm is None:
        sgv.aen.fit_generator(generator=genr, validation_data=vgenr,
                    use_multiprocessing=False, callbacks = [lr_finder],epochs=EPO)
    else:
        sgv.mgm.fit_generator(generator=genr, validation_data=vgenr,
                    use_multiprocessing=False, callbacks = [lr_finder],epochs=EPO)

    lr_finder.plot_loss()
    lr_finder.plot_lr()


