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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import model_to_dot
from keras.utils import multi_gpu_model
from keras import optimizers

from IPython.display import SVG, display
import matplotlib.pyplot as plt

import tensorflow as tf
import os
import numpy as np
import nltk

import smilesG as G
import getData

from rdkit import Chem, rdBase

rdBase.DisableLog('rdApp.error')


#%%
MAX_LEN = getData.MAX_LEN
NCHARS = getData.NCHARS

rules = G.gram.split('\n')

DIM = len(rules)

masks_K      = K.variable(G.masks)
ind_of_ind_K = K.variable(G.ind_of_ind)

productions = G.GCFG.productions()
prod_map = {}
for ix, prod in enumerate(productions):
    prod_map[prod] = ix
lhs_map = {}
for ix, lhs in enumerate(G.lhs_list):
    lhs_map[lhs] = ix

#%%
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

#%%
def plotm(model):
    display(SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg')))

def encoderMeanVar(x, latent_rep_size, max_length, epsilon_std=0.01):
    h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
    h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
    h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
    h = Flatten(name='flatten_1')(h)
    h = Dense(435, activation = 'relu', name='dense_1')(h)

    z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
    z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

    return (z_mean, z_log_var)

#%%
def buildEncoder(x, latent_rep_size, max_length, epsilon_std = 0.01):
    h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
    h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
    h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
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
        xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
        return xent_loss + kl_loss

    return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

#%%
def buildDecoder(z, latent_rep_size, max_length, charset_length,grL=2,grf=128):
    h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
    h = RepeatVector(max_length, name='repeat_vector')(h)
    if grL>0:
        h = GRU(grf, return_sequences = True, name='gru_1')(h)
    if grL>1:
        h = GRU(grf, return_sequences = True, name='gru_2')(h)
    if grL>2:
        h = GRU(grf, return_sequences = True, name='gru_3')(h)
    return TimeDistributed(Dense(charset_length), name='decoded_mean')(h) # don't do softmax, we do this in the loss now

#%%
def create(charset, max_length = 277, latent_rep_size = 2, weights_file = None, mgpu=1):

    charset_length = len(charset)

    x = Input(shape=(max_length, charset_length))
    _, z = buildEncoder(x, latent_rep_size, max_length)
    encoder = Model(x, z)

    encoded_input = Input(shape=(latent_rep_size,))
    decoder = Model(encoded_input,
        buildDecoder(encoded_input, latent_rep_size, max_length, charset_length))

    x1 = Input(shape=(max_length, charset_length))
    vae_loss, z1 = buildEncoder(x1, latent_rep_size, max_length)
    autoencoder = Model(x1,buildDecoder(z1, latent_rep_size, max_length, charset_length))

    # for obtaining mean and log variance of encoding distribution
    x2 = Input(shape=(max_length, charset_length))
    (z_m, z_l_v) = encoderMeanVar(x2, latent_rep_size, max_length)
    encoderMV = Model(inputs=x2, outputs=[z_m, z_l_v])

    if weights_file:
        autoencoder.load_weights(weights_file)
        encoder.load_weights(weights_file, by_name = True)
        decoder.load_weights(weights_file, by_name = True)
        encoderMV.load_weights(weights_file, by_name = True)
    #opt = optimizers.RMSprop()
    #opt = optimizers.Adagrad()
    #opt = optimizers.Adadelta()
    #opt = optimizers.Adam()
    opt = optimizers.SGD(momentum=0.8,nesterov=True)
    if mgpu>1:
        mgm=ModelMGPU(autoencoder, gpus=mgpu)
        mgm.compile(optimizer = opt, loss = vae_loss, metrics = ['accuracy'])
    else:
        autoencoder.compile(optimizer = opt, loss = vae_loss, metrics = ['accuracy'])
        mgm=None
    return (autoencoder,encoder,decoder,encoderMV,mgm)

#%%
def pop_or_nothing(S):
    try: return S.pop()
    except: return 'Nothing'

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
#        # python 2
#        rhs = [filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
#                          productions[i].rhs()) for i in sampled_output]

        for ix in range(S.shape[0]):
            S[ix].extend(list(map(str, rhs[ix]))[::-1])

    return X_hat # , ln_p

#%%
def encode(smiles,encoderMV):
    """ Encode a list of smiles strings into the latent space """
    assert type(smiles) == list
    one_hot = getData.to_one_hot(smiles)
    return encoderMV.predict(one_hot)[0]

#%%
def decode(z,decoder):
    """ Sample from the grammar decoder """
    assert z.ndim == 2
    unmasked = decoder.predict(z)
    X_hat = sample_using_masks(unmasked)
    # Convert from one-hot to sequence of production rules
    prod_seq = [[productions[X_hat[index,t].argmax()]
                 for t in range(X_hat.shape[1])]
                for index in range(X_hat.shape[0])]
    return [prods_to_eq(prods) for prods in prod_seq]

def isGood(smi):
    if smi == '':
        return False
    m = Chem.MolFromSmiles(smi)
    return m is not None

def cmpSmiles(s1,s2):
    s1=s1.strip()
    s2=s2.strip()
    mx=max(len(s1),len(s2))
    s1=s1.ljust(mx)
    s2=s2.ljust(mx)
    hit=sum([x==y for x,y in zip(s1,s2)])
    return hit/mx

#%%
def OneHot2Smiles(OH):
    prod_seq = [[productions[OH[index,t].argmax()]
                 for t in range(OH.shape[1])]
                for index in range(OH.shape[0])]
    smiles = [prods_to_eq(prods) for prods in prod_seq]
    return smiles
#%%
def main(genr,vgenr,XTE,fn, LATENT = 56, EPOCHS = 100,refit=False,mgpu=1):
    domp=False
    autoencoder,encoder,decoder,encoderMV,mgm = create(rules, max_length=MAX_LEN, latent_rep_size = LATENT,mgpu=mgpu)
    model_save = fn+'_L' + str(LATENT) + '_E' + str(EPOCHS) + '_val.hdf5'
    if (not os.path.isfile(model_save)) or refit:
        print('Training autoencoder.')
        checkpointer = ModelCheckpoint(filepath = model_save, verbose = 1, save_best_only = True)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, min_lr = 0.0001)
        if mgm is None:
            autoencoder.fit_generator(generator=genr, validation_data=vgenr, epochs = EPOCHS,
                            use_multiprocessing=domp, workers=6, callbacks = [checkpointer, reduce_lr])
        else:
            mgm.fit_generator(generator=genr, validation_data=vgenr, epochs = EPOCHS,
                              use_multiprocessing=domp, workers=6, callbacks = [checkpointer, reduce_lr])
    print('Loading weights')
    autoencoder.load_weights(model_save)
    encoder.load_weights(model_save, by_name = True)
    decoder.load_weights(model_save, by_name = True)
    encoderMV.load_weights(model_save, by_name = True)
    smiles = OneHot2Smiles(XTE)
    z1 = encode(smiles,encoderMV)
    sz1 = decode(z1,decoder)
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
        print(real + '\n' + mol + ':', m,s,flush=True)
    perfect = 100*perfect/nr
    good = 100 * good/nr
    return autoencoder,encoder,decoder,encoderMV, perfect, good

#%%
if __name__ == "__main__":

    fn = 'data/250k_rndm_zinc_drugs_clean'

    idx = getData.getZnIDX()

    idList = list(idx.keys())

    idTrain = idList[0:240000]

    idValid = idList[240000:245000]

    idTest = idList[245000:]

    genr = getData.ZincDataGen(idTrain,128,(MAX_LEN,NCHARS))

    vgenr = getData.ZincDataGen(idValid,128,(MAX_LEN,NCHARS))

    tstgen = getData.ZincDataGen(idTest,2000,(MAX_LEN,NCHARS))

    XTE, _tmp =  tstgen.__getitem__(0)

    del _tmp

    autoencoder,encoder,decoder,encoderMV,perfect,good=main(genr,vgenr,XTE,fn,EPOCHS=20,refit=True)

    print(perfect,good)
    # got 44%,63% - with 250k Zinc, varies a lot between runs
    #  2%, 17% 100k/50
    #  0%, 4% 50k/50

    #plot assumes running in Ipython
    #plotm(autoencoder)
    #plotm(encoder)
    #plotm(decoder)
    #plotm(encoderMV)
#%%
    try:
        h = autoencoder.history

        plt.plot(h.history['acc'])
        plt.plot(h.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    except:
        pass


