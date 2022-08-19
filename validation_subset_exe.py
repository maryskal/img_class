import os
import re
import pickle
import h5py as f
import argparse
from tensorflow import keras


def predicciones_modelo(model_name, subset = True):
    if bool(re.search('mask', model_name)):
        mask = True
    else:
        mask = False

    model = os.path.join('/home/mr1142/Documents/Data/models/neumonia', model_name)
    model = keras.models.load_model(model)
    dataframes = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")
    for key in dataframes.keys():
        globals()[key] = dataframes[key]
    with open("/home/mr1142/Documents/img_class/indices/val_subset", "rb") as fp:
        index = pickle.load(fp)
    index.sort()
    pred.save_metricas(model_name[:-3], model, X_train, y_train, index, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=3)
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    import funciones_complementarias.predicciones as pred

    modelos = ['prueba_EffNet3_fine-03_batch-8_lr-001_auc-94.h5',
        'prueba_IncResNet_fine-03_batch-8_lr-0001_auc-94.h5',
        'prueba_mask_IncResNet_fine-05_batch-8_lr-0001_auc-95.h5',
        'prueba_mask_IncResNet_fine-03_batch-8_lr-0001_auc-91.h5']
    for model_name in modelos:
        print(f'\nmodel: {model_name}')
        predicciones_modelo(model_name)

