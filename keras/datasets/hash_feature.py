import numpy as np
import os
import plyvel as ply
from PIL import Image, ImageOps
import StringIO
import gist
from sklearn.decomposition import PCA

def eigs_descend(X):
    eig_vals, eig_vecs = np.linalg.eig(X)
    # To get descending order, negate and sort in ascending order.
    sequence = np.argsort(-eig_vals)
    return eig_vecs[:, sequence]



def isoHash_lp(Lambda, iter, vector_size):
    a = np.trace(Lambda)/vector_size
    # Getting random orthogonal matrix U.
    R = np.random.random((vector_size, vector_size))
    U, _, _ = np.linalg.svd(R, compute_uv=1)
    Z = (U.dot(Lambda)).dot(U.T)

    for i in range(iter):
        # find T
        T = Z
        for j in range(vector_size):
            T[j, j] = a
        # find Z
        Q = eigs_descend(T)
        Z = (Q.dot(Lambda)).dot(Q.T)
    Q = Q.T
    return Q

def deep_hash(model,img):
    hash_code = model.predict(img)
    return hash_code
