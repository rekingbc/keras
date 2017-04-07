import numpy as np
import os
import plyvel as ply
from PIL import Image, ImageOps



def write_leveldb(db_path, converted_db_path,count, method='isohashlp',
    vector_size = 64, iter = 200):
    # A very primitive implementation. Need to refactor the code if
    # want to make it more versatile to a diversity of methods.
    db = ply.DB(db_path)
    converted_db = ply.DB(converted_db_path, create_if_missing=True,
     write_buffer_size=268435456)



    feat_dim = 960
    feats = np.zeros((feat_dim, count))

        with converted_db.write_batch() as converted_wb:
            if method.lower() == 'isohashlp':

                for key, value in db:
                    img = Image.frombytes('RGB', (224, 224), value)
                    img_data = np.asarray(img)

                    gistfeatures = gist.extract(img_data)
                    # The 960-dimensional gists, serving as inputs for isoHash.
                    # TODO: could have better implementation
                    feats[:, i] = gistfeatures
                    i += 1
                    if i%100 == 0:
                        print i

            else if method.lower() == 'deephash':
                deepfeats = np.zeros((deep_dim, count))
                    for key, value in db:
                        img = Image.frombytes('RGB', (224, 224), value)
                        img_data = np.asarray(img, dtype="float32")
                        img_data /= 255
                        deepfeat = deephash(model,img_data)



            # zero center the data
            avg = np.sum(gists, axis=1)/count
            avg = avg.reshape((gist_dim, 1))
            gists = gists - avg
            print "before pca"

            # This PCA method is said to be inefficient. Can switch.
            pca = PCA(n_components = vector_size)
            gist_reduced = (pca.fit_transform(gists.T)).T
            print "after pca"
            # Each row is a Principal Component.
            # gist_reduced is W^T*X
            Lambda = np.dot(gist_reduced, gist_reduced.T)

            Q = isoHash_lp(Lambda, iter, vector_size)
            # TODO: Use sign function to get Y.
            Y = (Q.T).dot(gist_reduced)
            Y = (Y >= 0)*1



          #write feature as key, image bytes as value
           i = 0
           for key, value in db:
               converted_key = Y[:, i].tobytes()
               converted_wb.put(converted_key, value)
               if i%100 == 0:
                   print i
                   i = i + 1


def write_lmdb(db_path, converted_db_path,count, method='isohashlp',
    vector_size = 64, iter = 200):
