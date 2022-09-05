import os
import json
import numpy as np
import pandas as pd
import funciones_imagenes.prepare_img_fun as fu
import funciones_complementarias.metrics_and_plots as met

def img_predict(model, img, mask = False, pix = 512):
    try:
        img = fu.get_prepared_img(img, pix, mask)
    except:
        img = np.random.randint(0,255,512*512).reshape((512,512, 1))
    return model.predict(img[np.newaxis,:])


def prediction_tensor(model, X, index, mask = False, pix = 512):
    y_pred = np.zeros((len(index), 3))
    for i in range(y_pred.shape[0]):
        y_pred[i,...] = img_predict(model, X[index[i]], mask, pix)
    return y_pred


def save_json(path, data):
    with open(os.path.join(path, 'metricas.json'), 'w') as j:
        json.dump(data, j)


def save_in_csv(path, name, metricas, subname):
    df = pd.read_csv(os.path.join(path, 'prediction_validation_metrics' + subname + '.csv'))
    save = [name] + list(metricas.values())
    try:
        # Si ya existe el modelo, se sobreescriben las métricas
        i = df[df['name'] == name].index
        df.loc[i[0]] = save
    except:
        df.loc[len(df.index)] = save
    df.reset_index(drop=True)
    df.to_csv(os.path.join(path, 'prediction_validation_metrics.csv'), index = False)


def save_metricas(name, model, X, y, index, mask = False, subname = ''):
    y_pred = prediction_tensor(model, X, index, mask)
    y_real = y[index]
    print('prediccion realizada')
    metricas, plots = met.metricas_dict(y_real, y_pred)
    print('metricas realizadas')
    p = '/home/mr1142/Documents/Data/models/neumonia/validation_results'
    path = os.path.join(p, name)
    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")
    try:
        save_json(path, metricas)
        print('json guardado')
    except:
        print('json no saved')
    save_in_csv(p, name, metricas, subname)
    print('guardado en tabla csv')
    for k, v in plots.items():
        met.save_plot(v, path, k)
    print('plots guardados')
    met.class_report(y_real, y_pred, path)

