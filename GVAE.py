# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:06:24 2018

@author: Steve O'Hagan

Based off: https://github.com/kanojikajino/grammarVAE
Paper: https://arxiv.org/abs/1703.01925

"""

from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers import Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import model_to_dot

import tensorflow as tf
import numpy as np
import nltk

from time import time

import smilesG as G

from dataTools import lhs_map, prods_to_eq, OneHot2Smiles, to_one_hot

from IPython.display import SVG, display

from rdkit import Chem

MAX_LEN = G.MAX_LEN
DIM = G.DIM

epsilon_std = 1.0

masks_K      = K.variable(G.masks)
ind_of_ind_K = K.variable(G.ind_of_ind)

productions = G.GCFG.productions()

def jac(yt,yp,sm=1e-6):
    y1 = np.reshape(yt,-1)
    y2 = np.reshape(yp,-1)
    intr = np.sum(y1*y2)
    sum_ = np.sum(y1+y2)
    return (intr+sm)/(sum_ - intr + sm)

def plotm(model):
    display(SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg')))

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def makeCNV(x,layers=1):
    for layer in range(layers):
        # filters, kernel_size
        if layer<2:
            x = Convolution1D(9, 9, activation = 'relu', name='conv_'+str(layer+1))(x)
        else:
            x = Convolution1D(10, 11, activation = 'relu', name='conv_'+str(layer+1))(x)
    return x

def encoderMeanVar(x, latent_rep_size, layers=3):
    h = makeCNV(x,layers)
    h = Flatten(name='flatten_1')(h)
    h = Dense(435, activation = 'relu', name='dense_1')(h)

    z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
    z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

    return (z_mean, z_log_var)

def buildEncoder(x, latent_rep_size, layers=3,beta=2.0):
    h = makeCNV(x,layers)
    h = Flatten(name='flatten_1')(h)
    h = Dense(435, activation = 'relu', name='dense_1')(h)

    def sampling(args):
        z_mean_, z_log_var_ = args
        batch_size = K.shape(z_mean_)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
        return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

    z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
    z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

    # this function is the main change.
    # essentially we mask the training data so that we are only allowed to apply
    #   future rules based on the current non-terminal
    def conditional(x_true, x_pred):
        most_likely = K.argmax(x_true)
        most_likely = tf.reshape(most_likely,[-1]) # flatten most_likely
        ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely),1) # index ind_of_ind with res
        ix2 = tf.cast(ix2, tf.int32) # cast indices as ints
        M2 = tf.gather_nd(masks_K, ix2) # get slices of masks_K with indices
        M3 = tf.reshape(M2, [-1,MAX_LEN,DIM]) # reshape them
        P2 = tf.multiply(K.exp(x_pred),M3) # apply them to the exp-predictions
        P2 = tf.div(P2,K.sum(P2,axis=-1,keepdims=True)) # normalize predictions
        return P2

    def vae_loss(x, x_decoded_mean):
        x_decoded_mean = conditional(x, x_decoded_mean) # we add this new function to the loss
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = MAX_LEN * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
        return xent_loss + beta * kl_loss

    return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

def buildDecoder(z, *, latent_rep_size,grL=3,grf=501):
    h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
    h = RepeatVector(MAX_LEN, name='repeat_vector')(h)
    for L in range(grL):
        h = GRU(grf, return_sequences = True, name=f'gru_{L+1}')(h)
    return TimeDistributed(Dense(DIM), name='decoded_mean')(h) # don't do softmax, we do this in the loss now

def pop_or_nothing(S):
    try: return S.pop()
    except: return 'Nothing'

def sample_using_masks(unmasked):
    """ Samples a one-hot vector, masking at each timestep.
        This is an implementation of Algorithm ? in the paper. """
    eps = 1e-100
    X_hat = np.zeros_like(unmasked)

    # Create a stack for each input in the batch
    S = np.empty((unmasked.shape[0],), dtype=object)

    for ix in range(S.shape[0]):
        S[ix] = [str(G.start_index)]

    # Loop over time axis, sampling values and updating masks
    for t in range(unmasked.shape[1]):
        next_nonterminal = [lhs_map[pop_or_nothing(a)] for a in S]
        mask = G.masks[next_nonterminal]
        masked_output = np.exp(unmasked[:,t,:])*mask + eps
        sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
        X_hat[np.arange(unmasked.shape[0]),t,sampled_output] = 1.0

        # Identify non-terminals in RHS of selected production, and
        # push them onto the stack in reverse order
        rhs = [[a for a in productions[i].rhs() if (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None')]
                   for i in sampled_output]

        for ix in range(S.shape[0]):
            S[ix].extend(list(map(str, rhs[ix]))[::-1])

    return X_hat # , ln_p

def isGood(smi):
    if smi == '':
        return False
    try:
        m = Chem.MolFromSmiles(smi)
    except:
        m = None
    return m is not None

def cmpSmiles(s1,s2):
    s1=s1.strip()
    s2=s2.strip()
    mx=max(len(s1),len(s2))
    s1=s1.ljust(mx)
    s2=s2.ljust(mx)
    hit=sum([x==y for x,y in zip(s1,s2)])
    return hit/mx

class smilesGVAE:

    def __init__(self, *, LATENT=56, nC=3, nD=3,ngpu=1, gruf=501, wFile=None, opt='adam', beta=2.0,EPOCH=1):
        self.LT = LATENT
        self.nC = nC
        self.nD = nD
        self.ngpu = ngpu
        self.gruF = gruf
        self.wFile = wFile
        self.opt = opt
        self.beta = beta
        self.doEStop = True
        self.esp = 5
        self.EPO = EPOCH
        self.mgm = None

        x = Input(shape=(MAX_LEN, DIM))
        _, z = buildEncoder(x, self.LT, self.nC, self.beta)
        self.enc = Model(x, z)

        encoded_input = Input(shape=(self.LT,))
        self.dec = Model(encoded_input, buildDecoder(encoded_input, latent_rep_size=self.LT, grL=self.nD,grf=self.gruF))

        x1 = Input(shape=(MAX_LEN, DIM))
        vae_loss, z1 = buildEncoder(x1, self.LT, self.nC)
        self.aen = Model(x1,buildDecoder(z1, latent_rep_size=self.LT,grL=nD,grf=self.gruF))

        # for obtaining mean and log variance of encoding distribution
        x2 = Input(shape=(MAX_LEN, DIM))
        (z_m, z_l_v) = encoderMeanVar(x2, self.LT, self.nC)
        self.emv = Model(inputs=x2, outputs=[z_m, z_l_v])

        if wFile:
            self.loadWeights(wFile)

        if self.ngpu>1:
            self.mgm=ModelMGPU(self.aen, gpus=self.ngpu)
            self.mgm.compile(optimizer = self.opt, loss = vae_loss, metrics = ['accuracy'])
        else:
            self.aen.compile(optimizer = self.opt, loss = vae_loss, metrics = ['accuracy'])
            self.mgm=None
        return

    def loadWeights(self,wFile):
        if self.mgm is not None:
            self.mgm.load_weights(wFile)
        else:
            self.aen.load_weights(wFile)
        self.enc.load_weights(wFile, by_name = True)
        self.dec.load_weights(wFile, by_name = True)
        self.emv.load_weights(wFile, by_name = True)
        return

    def evaluate(self,X):
        if self.mgm is None:
            return self.aen.evaluate(X,X)
        else:
            return self.mgm.evaluate(X,X)

    def decode(self,z):
        """ Sample from the grammar decoder """
        assert z.ndim == 2
        unmasked = self.dec.predict(z)
        X_hat = sample_using_masks(unmasked)
        # Convert from one-hot to sequence of production rules
        prod_seq = [[productions[X_hat[index,t].argmax()]
                     for t in range(X_hat.shape[1])]
                    for index in range(X_hat.shape[0])]
        return [prods_to_eq(prods) for prods in prod_seq]

    def encode(self,smiles):
        """ Encode a list of smiles strings into the latent space """
        assert type(smiles) == list
        one_hot = to_one_hot(smiles)
        return self.emv.predict(one_hot)[0]

    def testPerformance(self,XTE):
        smiles = OneHot2Smiles(XTE)
        z1 = self.encode(smiles)
        sz1 = self.decode(z1)
        perfect=0
        good=0
        nr=len(smiles)
        for mol,real in zip(sz1,smiles):
            m = isGood(mol)
            if m:
                good+=1
            s = cmpSmiles(real,mol)
            if s>=1.0:
                perfect+=1
            #print(real + '\n' + mol + ':', m,s,flush=True)
        perfect = 100*perfect/nr
        good = 100 * good/nr
        return (perfect,good)

    def doFitG(self,genr,vgenr,fn):
        EPOCHS = self.EPO
        modsave = fn+'.hdf5'
        print('Training autoencoder.')
        t = int(time())
        chkptr = ModelCheckpoint(filepath = modsave, verbose = 0, save_best_only = True)
        rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, min_lr = 0.0001)
        tb = TensorBoard(log_dir=f'logs/{fn}/{t}')
        cbList = [chkptr,rlr,tb]
        if self.doEStop:
            estop = EarlyStopping(patience=self.esp)
            cbList.append(estop)
        if self.mgm is None:
            self.aen.fit_generator(generator=genr, validation_data=vgenr,
                epochs = EPOCHS, callbacks = cbList)
        else:
            self.mgm.fit_generator(generator=genr, validation_data=vgenr,
                epochs = EPOCHS, callbacks = cbList)
        self.loadWeights(modsave)
        return

    def doFit(self,trn,fn, bat):
        EPOCHS = self.EPO
        modsave = fn+'.hdf5'
        print('Training autoencoder.')
        t = int(time())
        chkptr = ModelCheckpoint(filepath = modsave, verbose = 0, save_best_only = True)
        rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, min_lr = 0.0001)
        tb = TensorBoard(log_dir=f'logs/{fn}/{t}')
        cbList = [chkptr,rlr,tb]
        if self.doEStop:
            estop = EarlyStopping(patience=self.esp)
            cbList.append(estop)
        self.aen.fit(x=trn,y=trn, validation_split=0.2, epochs = EPOCHS,
               batch_size=bat, callbacks = cbList)
        if self.mgm is None:
            self.aen.fit(x=trn,y=trn, validation_split=0.2, epochs = EPOCHS,
               batch_size=bat, callbacks = cbList)
        else:
            self.mgm.fit(x=trn,y=trn, validation_split=0.2, epochs = EPOCHS,
               batch_size=bat, callbacks = cbList)
        self.loadWeights(modsave)
        return

    def jaccScore(self,Xt):
        z = self.emv.predict(Xt)[0]
        yp = self.dec.predict(z)
        yo = sample_using_masks(yp)
        sc = jac(Xt,yo)
        print(f'Score: {sc}')
        return sc

    def listSamples(self,x1,sam=10):
        s0 = OneHot2Smiles(x1)
        z = self.enc.predict(x1)
        yp = self.dec.predict(z)
        print(s0,'::')
        for i in range(sam):
            yo = sample_using_masks(yp)
            s1 = OneHot2Smiles(yo)
            print(f'{i}: {s1}')
        print(jac(x1,yo))

    def listMeans(self,x1,sam=10):
        s0 = OneHot2Smiles(x1)
        z = self.emv.predict(x1)[0]
        yp = self.dec.predict(z)
        print(s0,'::')
        for i in range(sam):
            yo = sample_using_masks(yp)
            s1 = OneHot2Smiles(yo)
            print(f'{i}: {s1}')
        print(jac(x1,yo))

    def encodeFilter(self,smiles):
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
        OH = to_one_hot(good)
        return self.emv.predict(OH)[0], good, bad


#%%
if __name__ == "__main__":

    pass

