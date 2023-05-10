import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas
import os

from sklearn.datasets import fetch_openml
# from sklearn.manifold import TSNE
from sklearn.manifold._t_sne_bhmpi import tsne
from mpi4py import MPI


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get data
if rank == 0:
    if not os.path.exists('out/X.npy'):
        X = fetch_openml('mnist_784', as_frame=False)['data'].astype('float32')
        np.save('out/X.npy', X)
    else:
        X = np.load('out/X.npy')
else:
    X = np.empty((70000, 784), dtype='float32')

comm.Bcast(X, root=0)

# Fit TSNE
# tsne = TSNE(
#     n_components=2,
#     learning_rate='auto',
#     init='random',
#     perplexity=30,
#     method='barnes_hut',
#     verbose=2
# )

# Z = tsne.fit_transform(X[:1000])

N = 1000

Z = np.zeros((N, 2), dtype='float32')

tsne(
    X[:N],
    Z,
    n_components=2,
    perplexity=30,
    angle=0.5,
    verbose=2,
    random_state=0,
    method='barnes_hut',
    init='random',
    comm=comm
)

# plot Z
if rank == 0:
    plt.scatter(*Z.T, s=2)
    plt.savefig('out/tsne.png')
