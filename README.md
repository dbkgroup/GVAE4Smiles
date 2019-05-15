# GVAE4Smiles

Grammar Variational Autoencoder for Smiles.

Based off: https://github.com/kanojikajino/grammarVAE
Paper: https://arxiv.org/abs/1703.01925

Now includes code to cache one-hot-encodings to disk via h5py and the corresponding data generator. This is needed to cope with training on large datasets e.g. six million SMILES strings sampled from ZincDB. [Not included here as too big for github].

Any SMILES strings need to be filtered to conform to the MAX_LENGTH criterion and the SMILES-Grammar used.
