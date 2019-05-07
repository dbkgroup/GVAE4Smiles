# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:06:24 2018

@author: Steve O'Hagan

Based off: https://github.com/kanojikajino/grammarVAE
Paper: https://arxiv.org/abs/1703.01925

"""

from generateData import H5DataGenV2

from GVAE import smilesGVAE

import numpy as np
import h5py
import matplotlib.pyplot as plt

import optuna

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from os import listdir
from os.path import isfile, join

def getFileList(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


#%%
if __name__ == "__main__":

    pth = 'data/6MZincHV2/' #for data cache
    k = 100000

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
    batch = 325 * gpus

    genr = H5DataGenV2(idTrain,batch,pth=pth)
    vgenr = H5DataGenV2(idValid,batch,pth=pth)
    tstgen = H5DataGenV2(idTest,2000,pth=pth)
    XTE, _tmp =  tstgen.__getitem__(0)
    del _tmp, tstgen

    pfix = 'HPOtest100kGVAE_'

    def netHO(trial):
        pp = {       
            'LATENT' : trial.suggest_categorical('LATENT', [40,70,100]),
            'nC' : trial.suggest_int('nC', 1,3),
            'nD' : trial.suggest_int('nD', 1,3),
            'gruf' : trial.suggest_categorical('gruf', [64,128,256,501]),
            'beta' : trial.suggest_categorical('beta', [0.6,1.0,2.0]),
            'EPOCH' : trial.suggest_categorical('EPOCH', [30]),
            'ngpu' : trial.suggest_categorical('ngpu', [4]),
            'opt' : trial.suggest_categorical('opt', ['adam'])
        }
        
        fn = f'{pfix}{trial.number}'

        sgv = smilesGVAE(**pp)

        sgv.doFitG(genr,vgenr,fn)

        sc = sgv.evaluate(XTE)
        print(f'Score: {sc}')
    
        sc = 1.0 - sgv.jaccScore(XTE)
        #print(f'JaccLoss: {sc}')
        return sc

    study = optuna.create_study()

    study.optimize(netHO,n_trials=1)

    df = study.trials_dataframe()

    df2 = df['params'].copy()

    df2['bxe'] = df['value']

    df2.to_csv(f'data/{pfix}.csv')

    bp = study.best_params

    bt = study.best_trial.number

    bcn = smilesGVAE(**bp)

    fn = f'data/{pfix}{bt}.hdf5'

    bcn.loadWeights(fn)

    xp = bcn.aen.predict(XTE)

    bcn.evaluate(XTE)

    logdir = f'logs/{pfix}{bt}/'

    fn = getFileList(logdir)

    fn = fn[0]

    eacc = EventAccumulator(logdir+fn)

    eacc.Reload()

    tj = eacc.Scalars('loss')

    vj = eacc.Scalars('val_loss')

    steps = len(tj)

    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = tj[i][2] # value
        y[i, 1] = vj[i][2]

    plt.plot(x, y[:,0], label='training loss')
    plt.plot(x, y[:,1], label='validation loss')

    plt.xlabel("Steps")
    plt.ylabel("Jaccard Loss")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()

    df.to_csv('data/HPO4PTest100k.csv')

    #optuna.visualization.plot_intermediate_values(study)







