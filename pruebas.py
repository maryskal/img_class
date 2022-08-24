import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import pickle
import h5py as f
from tensorflow import keras
import funciones_complementarias.evaluation as ev


model_name = 'prueba_EffNet3_fine-05_batch-8_lr-0001_auc-92.h5'
model = os.path.join('/home/mr1142/Documents/Data/models/neumonia', model_name)
model = keras.models.load_model(model)

dataframes = f.File("/datagpu/datasets/mr1142/cxr_consensus_dataset_nocompr.h5", "r")
for key in dataframes.keys():
    globals()[key] = dataframes[key]

with open("/home/mr1142/Documents/img_class/indices/val_subset", "rb") as fp:
    index = pickle.load(fp)

index = index[0:500]
index.sort()

mask = False

import funciones_complementarias.prediction as predi
X = X_train
y = y_train

y_pred = predi.prediction_tensor(model, X, index, mask)
y_binar = predi.extract_max(y_pred.copy())
y_real = y[index]

pred = y_binar[:,0]
real = y_real[:,0]


pred = np.stack([y_pred[:,0], 1-y_pred[:,0]], axis = 1)
real = np.stack([y_real[:,0], 1-y_real[:,0]], axis = 1)


fpr, tpr, _ = metrics.roc_curve(real, pred)
metrics.auc(fpr, tpr)

predi.extract_max(pred)

index = index[28000:29000]

malas = X_train[28250:28270]

for i, mal in enumerate(malas):

    cv2.imwrite('img/' + str(i)+'.png', mal)
cv2.imwrite('img/' + str(28254)+'.png', X_train[28254])
import cv2

