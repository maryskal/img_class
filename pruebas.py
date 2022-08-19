import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import pickle
import h5py as f
from tensorflow import keras
import funciones_complementarias.evaluation as ev


model_name = 'prueba_mask_EffNet3_fine-08_batch-8_lr--05_auc-56.h5'
model = os.path.join('/home/mr1142/Documents/Data/models/neumonia', model_name)
model = keras.models.load_model(model)

dataframes = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")
for key in dataframes.keys():
    globals()[key] = dataframes[key]

with open("/home/mr1142/Documents/img_class/indices/val_subset", "rb") as fp:
    index = pickle.load(fp)

index.sort()

mask = True

import funciones_complementarias.predicciones as predi
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