import os
import re
import pickle
import h5py as f
import argparse
from tensorflow import keras


def predicciones_modelo(model_name):
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
    results = ev.evaluate(model, X_train, y_train, index, mask)
    print('results calculados')
    ev.save_eval(model_name[:-3], results)
    print('results guardados en tabla csv')
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
    import funciones_complementarias.evaluation as ev
    import funciones_complementarias.predicciones as pred

    path = '/home/mr1142/Documents/Data/models/neumonia'
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    modelos = [file for file in onlyfiles if not bool(re.search('completo', file))]
 
    for model_name in modelos:
        print(f'\nmodel: {model_name}')
        predicciones_modelo(model_name)

