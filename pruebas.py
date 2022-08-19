import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import pickle
import h5py as f
from tensorflow import keras
import funciones_complementarias.predicciones as pred
import funciones_complementarias.evaluation as ev


model_name = 'prueba_EffNet3_fine-05_batch-8_lr-0001_auc-92.h5'
model = os.path.join('/home/mr1142/Documents/Data/models/neumonia', model_name)
model = keras.models.load_model(model)

dataframes = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")
for key in dataframes.keys():
    globals()[key] = dataframes[key]

# VALIDACION
with open("/home/mr1142/Documents/img_class/indices/val_subset", "rb") as fp:
    val_index = pickle.load(fp)

results = ev.evaluate(model, X_train, y_train, val_index, mask = False)
ev.save_eval(model_name, results)

with open("/home/mr1142/Documents/img_class/indices/val_subset", "rb") as fp:
    index = pickle.load(fp)

index[:10]
index.sort()



pred.save_metricas(model_name, model, X_train, y_train, index)
