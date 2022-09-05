import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import pickle
import h5py as f
from tensorflow import keras
import funciones_complementarias.evaluation as ev
from sklearn import metrics
import matplotlib.pyplot as plt
import re

modelos = ['completo_EffNet3_fine-07_batch-8_lr-0001_auc-99',
 'completo_IncResNet_fine-05_batch-8_lr-0001_auc-99']

for model_name in modelos:
    model_name = model_name + '.h5'
    model = os.path.join('/home/mr1142/Documents/Data/models/neumonia', model_name)
    model = keras.models.load_model(model)
    dataframes = f.File("/datagpu/datasets/mr1142/cxr_consensus_dataset_nocompr.h5", "r")
    for key in dataframes.keys():
        globals()[key] = dataframes[key]
    with open("/home/mr1142/Documents/img_class/indices/val_subset", "rb") as fp:
        index = pickle.load(fp)
    index.sort()
    if bool(re.search('mask', model_name)):
        mask = True
    else:
        mask = False
    import funciones_complementarias.prediction as pred
    pred.save_metricas(model_name[:-3], model, X_train, y_train, index, mask)






import funciones_complementarias.metrics_and_plots as met

X = X_train
y = y_train

y_pred = predi.prediction_tensor(model, X, index, mask)
y_binar = met.extract_max(y_pred.copy())
y_real = y[index]


metricas, plots = met.metricas_dict(y_real, y_pred)

pred = y_pred[:,2]
real = y_real[:,2]
precision, recall, pr_thresholds = metrics.precision_recall_curve(real, pred)

a = met.pred_recall_plot(precision, recall, pr_thresholds)
a.savefig('a.png')

suma = precision+recall
fig, ax = plt.subplots()
ax.plot(suma)
i = np.argmax(suma)
ax.plot([i], [suma[i]], "ro", label="th="+str(round(th,2))) 
fig.savefig('plt.png')

fig, ax = plt.subplots()
ax.plot(recall, precision, "g-")
x = precision[i]
y = recall[i]
th = pr_thresholds[i]
ax.plot([y], [x], "ro", label="th="+str(round(th,2))) 
fig.savefig('plt2.png')


import cv2
cv2.imwrite('plot.png', )