import numpy as np
import os
import lmdb
from PIL import Image, ImageOps
import random
import sys

# Load the dataset to database, build the leveldb
# from the dataset:
# input: train.txt Path to the images
#
def build_db(db_path, rootpath, trainfile, imgsize=(224,224)):

    env = lmdb.open(db_path, map_size=2684354560)
    i = 0
    with env.begin(write=True) as tdb:
        with open(train_file) as f:
            for fileName in f.read().splitlines():
                print fileName
                i = i + 1
                img = Image.open(rootpath+fileName, 'r')
                img.thumbnail(imgsize, Image.ANTIALIAS)
                img = ImageOps.fit(img,imgsize,Image.ANTIALIAS)
                str_id = '{:08}'.format(i)
                tdb.put(str_id.encode('ascii'), img.tobytes())

def load_db(db_path,imgsize=(224,224)):
    env = lmdb.open(db_path, readonly=True)
    imglist=[]
    with env.begin() as tdb:
        cursor = tdb.cursor()
        for key, value in cursor:
            imgitem = value
            img = Image.frombytes('RGB',imgsize,imgitem)
            imglist.append(img)
