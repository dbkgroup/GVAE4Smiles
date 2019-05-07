import pandas as pd
import nltk

from time import time
import concurrent.futures as cf

from dataTools import checkCompat, filterOK

import smilesG as G

def makeNP():
    pth = 'data/Smiles.xlsx'

    dat = pd.read_excel(pth)  

    dat = dat['SMILES']

    t0 = time()
    
    good = filterOK(dat)

    t1 = time() - t0

    print('time:',t1)

    print('Good len:',len(good))

    # 5k map: 239 s; pool.map: 47s; all 77k -> 1519s

    pth = 'data/NPsNotInTrainGood.smi'

    with open(pth,'w') as f:
        for smi in good:
            f.write(smi+'\n')
    print('Done.')


if __name__ == "__main__":

    pass

    
