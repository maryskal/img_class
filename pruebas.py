import os
import pickle
import h5py as f
from tensorflow import keras
import img_class.funciones_complementarias.predicciones as pred

model_name = 'prueba_EffNet3_fine-03_batch-8_lr-0001_auc-95.h5'
model = os.path.join('/home/mr1142/Documents/Data/models/neumonia', model_name)
model = keras.models.load_model(model)

with open("/home/mr1142/Documents/img_class/indices/val_subset", "rb") as fp:
    index = pickle.load(fp)

index[:10]
index.sort()

dataframes = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")
for key in dataframes.keys():
    globals()[key] = dataframes[key]

pred.save_metricas(model_name, model, X_train, y_train, index)
