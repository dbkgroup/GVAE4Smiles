# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:33:31 2018

@author: Dr. Steve O'Hagan
"""
import pandas as pd

from GVAE import getData
from GVAE.doGVAE_gen import loadModel, encodeFilter, plotm, decode

pth = 'D:/Python3Scripts/GVAE/data/'

fn = '10kChEMBL23' #no need .smi suffix

smiList = getData.getSmi(pth + fn)

model_save = pth + '250k_rndm_zinc_drugs_clean_L56_E50_val.hdf5'

autoencoder,encoder,decoder,encoderMV = loadModel(model_save)

del autoencoder,encoder

encoded, good, bad = encodeFilter(smiList,encoderMV)

sz1 = decode(encoded,decoder)

plotm(encoderMV)

dat = pd.DataFrame(encoded)

dat['SMILES'] = good

dat['decoded'] = sz1



