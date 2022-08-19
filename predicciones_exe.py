import os
import pickle
import h5py as f
from tensorflow import keras
import funciones_complementarias.predicciones as pred

def predicciones_modelo(model_name, subset = True):
    model = os.path.join('/home/mr1142/Documents/Data/models/neumonia', model_name)
    model = keras.models.load_model(model)

    dataframes = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")
    for key in dataframes.keys():
        globals()[key] = dataframes[key]

    if subset:
        with open("/home/mr1142/Documents/img_class/indices/val_subset", "rb") as fp:
            index = pickle.load(fp)

        index.sort()
        pred.save_metricas(model_name[:-3], model, X_train, y_train, index)
    else:
        pred.save_metricas(model_name[:-3], model, X_val, y_val, list(range(len(y_val))))
    


if __name__ == '__main__':
    modelos = ['prueba_EffNet3_fine-03_batch-8_lr-0001_auc-95.h5']
    for model_name in modelos:
        predicciones_modelo(model_name)