import os
import json
import numpy as np
import pandas as pd
import funciones_imagenes.prepare_img_fun as fu
import funciones_complementarias.metrics_and_plots as met

def img_predict(model, img, mask = False, pix = 512):
    img = fu.get_prepared_img(img, pix, mask)
    return model.predict(img[np.newaxis,:])


def prediction_tensor(model, X, index, mask = False, pix = 512):
    y_pred = np.zeros((len(index), 3))
    for i in range(y_pred.shape[0]):
        y_pred[i,...] = img_predict(model, X[index[i]], mask, pix)
    return y_pred


def save_json(path, name, data):
    with open(os.path.join(path, name +'.json', 'w')) as j:
        json.dump(data, j)


def save_in_csv(path, name, metrics):
    df = pd.read_csv(os.path.join(path, 'prediction_validation_metrics.csv'))
    d = pd.DataFrame(metrics)
    d.insert(0, 'name', name)
    df = pd.concat([df, d], ignore_index=True)
    df.to_csv(path, index = False)


def save_metricas(name, model, X, y, index, mask = False):
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
    save_json(path, name, metricas)
    print('json guardado')
    save_in_csv(p, name, metricas)
    print('guardado en tabla csv')
    for k, v in plots.items():
        met.save_plot(v, path, k)
    print('plots guardados')
    met.class_report(name, y_real, y_pred)

