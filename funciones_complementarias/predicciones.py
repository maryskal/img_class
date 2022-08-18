import numpy as np
import pandas as pd
from sklearn import metrics
import funciones_imagenes.prepare_img_fun as fu


def img_predict(model, img, pix = 512, mask = False):
    img = fu.get_prepared_img(img, pix, mask)
    return model.predict(img[np.newaxis,:])


def prediction_tensor(model, X_pred, index, pix = 512, mask = False):
    y_pred = np.zeros((len(index), 3))
    for i in range(y_pred.shape[0]):
        y_pred[i,...] = img_predict(model, X_pred[index[i]], pix, mask)
    return y_pred


def younden_idx(real, pred):
    fpr, tpr, thresholds = metrics.roc_curve(real, pred)
    return thresholds[np.argmax(tpr-fpr)]


def extract_max(array):
    for i in range(array.shape[0]):
        max = np.argmax(array[i,:])
        array[i,:] = 0
        array[i,max] = 1
    return array


def metricas_dict(name, y_real, y_pred):
    metrics_dict = {}
    for i in range(3):
        pred = y_pred[:,i]
        real = y_real[:,i]
        fpr, tpr, _ = metrics.roc_curve(real, pred)
        metrics_dict['general_auc_' + str(i)] = [metrics.auc(fpr, tpr)]
        metrics_dict['younden_' + str(i)] = [younden_idx(real, pred)]

    for combination in [[0,1], [0,2], [1,2]]:
        print(combination)
        pred = extract_max(y_pred[:,combination])
        real = extract_max(y_real[:,combination])
        metrics_dict['f1_score' + str(combination)] = [metrics.f1_score(real, pred, average = 'weighted')]
        metrics_dict['accuracy_score' + str(combination)] = [metrics.accuracy_score(real, pred)]
        metrics_dict['precision_score' + str(combination)] = [metrics.precision_score(real, pred, average = 'weighted')]
        metrics_dict['recall_score' + str(combination)] = [metrics.recall_score(real, pred, average = 'weighted')]
        metrics_dict['roc_auc_score' + str(combination)] = [metrics.roc_auc_score(real, pred)]

    return metrics_dict


def class_report(name, y_real, y_pred):
    y_binar = extract_max(y_pred)
    m = metrics.classification_report(y_real, y_binar, target_names = ['normal', 'moderado', 'severo'], output_dict = True)
    d = pd.DataFrame(m).transpose()
    d.to_csv('/home/mr1142/Documents/Data/models/neumonia/validation_results/prediction_metrics_reports/' + name + '.csv', index = False)


def save_metricas(name, model, X, y, index):
    y_pred = prediction_tensor(model, X, index)
    y_real = y[index]
    class_report(name, y_real, y_pred)
    metricas = metricas_dict(name, y_real, y_pred)
    path = '/home/mr1142/Documents/Data/models/neumonia/validation_results/prediction_validation_metrics.csv'
    df = pd.read_csv(path)
    d = pd.DataFrame(metricas)
    d.insert(0, 'name', name)
    df = pd.concat([df, d]).reset_index()
    df.to_csv(path, index = False)
