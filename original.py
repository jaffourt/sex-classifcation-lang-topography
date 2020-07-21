import nibabel as nib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import json
import os
import time
import sys
from json import JSONEncoder
import random
from collections import OrderedDict

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def iterate_dir(p):
    print('Finding nifti files...',end='')
    d=[]
    for f in os.listdir(p):
        fp=os.path.join(p,f)
        d.append(fp)
    print('Done.')
    return d

def load(l, subjs):
    print('Loading nifti files...',end='')
    v=OrderedDict()
    for d in l:
        for s in subjs:
            if s in d:
                img=nib.load(d)
                dat=img.get_fdata()
                dat[np.isnan(dat)]=0
                dat_2d = dat.reshape((dat.shape[0]*dat.shape[1]), dat.shape[2])
                dat_2d = dat_2d.transpose()
                v[d.split('/')[-1]]=dat_2d
    print('Done.')
    return v

def get_labels(f):
    print('Loading labels...',end='')
    df=pd.read_csv(f)
    subjs=df.subject
    cnames=df.label.unique()
    mapping = dict(zip(cnames, range(len(cnames))))
    labels=df[['label']]
    labels=labels.replace({'label':mapping})
    #coerce dtype obj->int
    labels=labels.astype(str).astype(int)
    labels=labels['label'].values
    print('Done.')
    return [labels, mapping, subjs]

def format_data(d):
    print('Formatting numpy array...',end='')
    #default MNI mapped to 2d
    s=np.empty([len(d),91,9919])
    i=0
#    keys=list(d.keys())
#    random.shuffle(keys)
    for k in d:
        s[i]=d[k]/np.max(d[k])
        i+=1
    #normal
   # s=s/np.max(s)
    print('Done.')
    return s

def j_save(imgs,labels,cmap):
    with open('classifier_data/images.json','w') as w:
        json.dump(imgs,w,cls=NumpyArrayEncoder)
    with open('classifier_data/labels.json','w') as w:
        json.dump(labels,w,cls=NumpyArrayEncoder)
    with open('classifier_data/map.txt','w') as w:
        for c in cmap:
            w.write('%s -> %s\n' % (c,cmap[c]))

def build_model(img_dim):
    model = keras.Sequential([
        keras.layers.Conv1D(32,3, activation="relu",input_shape=(img_dim[1], img_dim[2])),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.25),
      
        #keras.layers.Conv1D(32,3, activation="relu"),
        #keras.layers.MaxPooling1D(2),
        #keras.layers.Dropout(0.25),
        #keras.layers.Conv1D(32,3, activation="relu"),
        #keras.layers.MaxPooling1D(2),

        #keras.layers.Dropout(0.1),
        keras.layers.Flatten(input_shape=(img_dim[1], img_dim[2])),
        keras.layers.Dense(20,activation="relu"),
        keras.layers.Dense(16),
        keras.layers.Dense(2)
    ])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

def predict(model,test_images,test_labels):
    print('Creating probability model...',end='')
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print('Done.')
    correct=0
    for i in range(len(test_labels)):
        if np.argmax(predictions[i])==test_labels[i]:
            correct+=1
    print('Prediction accuracy: %d/%d' % (correct,len(test_labels)))

def shuffle_splits(imgs,labels,split):
    if split == 0:
        return imgs, labels
    i=int(len(labels)*.8)
    a=imgs[0:i]
    b=imgs[i:]
    c=labels[0:i]
    d=labels[i:]
    ba=tf.concat((b,a),axis=0)
    dc=np.concatenate([d,c])
    return [ba, dc]

def classify_images(imgs,labels):
    tot=len(labels)
    for split in range(0,5):
        img,labels=shuffle_splits(imgs,labels,split)
        train=int(tot*0.8)
        train_labels=labels[0:train]
        print('M - training set: %d/%d' % (np.sum(train_labels),train))
        train_images=imgs[0:train]
        test_labels=labels[train:]
        test_images=imgs[train:]
        model=build_model(train_images.shape)
        model.fit(train_images, train_labels, validation_split=0.2, epochs=5)
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
    #predict(model,test_images,test_labels)
        
def main():
    t=time.time()
    [all_labels, mapping,subj] = get_labels('classes_restricted.csv')
    hdrs = iterate_dir('../../../SN_betaweights_n812')
    imgs = load(hdrs,subj)
    all_images = format_data(imgs)
    classify_images(all_images,all_labels)
    #j_save(all_images,all_labels,mapping)
    print("It took %ds to run" % (time.time() - t))

if __name__ == '__main__':
    main()
