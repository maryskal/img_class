import os
import re
import pickle
import pandas as pd
import h5py as f
import argparse
from tensorflow import keras


def predicciones_modelo(model_name, ix):
    if bool(re.search('mask', model_name)):
        mask = True
    else:
        mask = False

    model = os.path.join('/home/mr1142/Documents/Data/models/neumonia', model_name + '.h5')
    model = keras.models.load_model(model)

    print('model loaded')

    dataframes = f.File("/datagpu/datasets/mr1142/cxr_consensus_dataset_nocompr.h5", "r")
    for key in dataframes.keys():
        globals()[key] = dataframes[key]

    with open("/home/mr1142/Documents/img_class/indices/" + ix, "rb") as fp:
        index = pickle.load(fp)

    index.sort()

    len(index)
    results = ev.evaluate(model, X_train, y_train, index, mask = mask)
    print('results calculados')
    ev.save_eval(model_name + '_resto', results)
    print('results guardados en tabla csv')
    pred.save_metricas(model_name  + '_resto', model, X_train, y_train, index, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=3)
    parser.add_argument('-ix',
                        '--ix',
                        help="GPU device",
                        type=str,
                        default='val_rest')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    ix = args.ix
    import funciones_evaluacion.evaluation as ev
    import funciones_evaluacion.prediction as pred

    path = '/home/mr1142/Documents/Data/models/neumonia/validation_results/prediction_validation_metrics.csv'
    df = pd.read_csv(path)
    p = '/home/mr1142/Documents/Data/models/neumonia'
    modelos = os.listdir(p)
    modelos = [modelo[:-3] for modelo in modelos if os.path.isfile(os.path.join(p, modelo))]
    modelos = [modelo for modelo in modelos if not bool(re.search('completo', modelo))]
    modelos_evaluados = list(df['name'])

    if ix == 'val_rest':
        modelos_evaluados_resto = [modelo for modelo in 
                                    modelos_evaluados if bool(re.search('resto', modelo))]
        modelos_evaluados_resto = [re.split('_resto', modelo)[0] for 
                                    modelo in modelos_evaluados_resto]
        modelos_a_evaluar = list(set(modelos)-set(modelos_evaluados_resto))
    
    if ix == 'val_subset':
        modelos_a_evaluar = list(set(modelos)-set(modelos_evaluados))
        
    print(len(modelos_a_evaluar))
    print(modelos_a_evaluar)

    for model_name in modelos_a_evaluar:
        print(f'\nmodel: {model_name}')
        predicciones_modelo(model_name, ix)



