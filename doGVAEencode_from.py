# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:06:24 2018

@author: Steve O'Hagan

Based off: https://github.com/kanojikajino/grammarVAE
Paper: https://arxiv.org/abs/1703.01925

"""

from GVAE import smilesGVAE
from dataTools import s2oh

import pandas as pd

if __name__ == "__main__":

    gpus = 1
    batch = 312 * gpus
    kk = 3000000

    # for data
    xl='data/Smiles.xlsx'

    #for weights
    wf = f'data/{kk//1000}k6MZinc_L56_E60_val.hdf5'

    #for result
    csv = f'data/{kk//1000}k6MZinc_L56_E60_val.csv'

    params  = {
        'LATENT':56,
        'nC':3,
        'nD':3,
        'beta':1.0,
        'gruf':501,
        'ngpu':gpus,
        'opt':'adam',
        'wFile':wf
    }

    print('XLS:',xl)
    print('Model save:',wf)

    dta = pd.read_excel(xl)

    smi = list(dta['SMILES'])

    sgv = smilesGVAE(**params)

    rslt, okList, badList = sgv.encodeFilter(smi)

    pd.DataFrame(rslt).to_csv(csv)

    XTE = s2oh(okList)

    perfect,good = sgv.testPerformance(XTE)

    print(f'Perf:{perfect:.2f} Good:{good:.2f}')






