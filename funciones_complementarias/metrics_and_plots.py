import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


# METRICAS
def younden_idx(real, pred):
    fpr, tpr, thresholds = metrics.roc_curve(real, pred)
    return thresholds[np.argmax(tpr-fpr)]


def pred_recall_thres(precision, recall, thresholds):
    pr_max = thresholds[np.argmax(precision+recall)]
    pr_cut = thresholds[np.where(precision == recall)]
    return pr_max, pr_cut


# PLOTS
def AUC_plot(fpr, tpr, thresholds, auc):
    i = np.argmax(tpr-fpr)
    th = thresholds[i]
    x = fpr[i]
    y = tpr[i]
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot([x], [y], "ro", label="th="+str(round(th,2))) 
    plt.legend(loc=4)
    return plt


def pred_recall_plot(precision, recall):
    plt.plot(recall, precision)
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.legend(loc=4)
    plt.title('Precision-Recall curve')
    return plt


def plot_precision_recall_vs_threshold(precision, recall, thresholds):
    x = thresholds[np.where(precision == recall)]
    y = precision[np.where(precision == recall)]
    x_max = thresholds[np.argmax(precision+recall)]
    y_max = precision[np.argmax(precision+recall)]
    plt.axis([0 ,1, np.min(precision), 1])
    plt.plot(thresholds, precision[:-1], "k--", label="Precision", linewidth=2)
    plt.plot(thresholds, recall[:-1], "g-", label="Recall", linewidth=2)
    plt.plot([x], [y], "ro", label="th="+str(round(x[0],2))) 
    plt.plot([x, x], [0, y], "r:")
    plt.plot([x_max], [y_max], "bo", label="th="+str(round(x_max,2))) 
    plt.plot([x_max, x_max], [0, y_max], "b:")
    plt.xlabel("Threshold")
    plt.grid(True)
    plt.legend()
    return plt


def save_plot(fig, folder, title):
    fig.savefig(os.path.join(folder, title + '.png'))


# DICCIONARIOS
def extract_max(array):
    for i in range(array.shape[0]):
        max = np.argmax(array[i,:])
        array[i,:] = 0
        array[i,max] = 1
    return array


# Cada una de las clases genera un diccionario con las metricas y los plots
def metrics_per_class(name, real, pred):
    metricas = {}
    fpr, tpr, auc_thresholds = metrics.roc_curve(real, pred)
    metricas['auc_' + name] = metrics.auc(fpr, tpr)
    metricas['younden_'+ name] = younden_idx(real, pred)
    precision, recall, pr_thresholds = metrics.precision_recall_curve(pred, real)
    metricas['pr_max_'+ name], metricas['pr_cut_'+ name] = pred_recall_thres(precision, 
                                                                            recall, 
                                                                            pr_thresholds)
    print(f'metricas clase {name} calculadas')
    plots = {}
    plots['pred_rec_plot_' + name] = pred_recall_plot(precision, recall)
    plots['auc_plot_' + name] = AUC_plot(fpr, tpr, auc_thresholds)
    plots['pr_re_th_plot_' + name] = plot_precision_recall_vs_threshold(precision, 
                                                                        recall, 
                                                                        pr_thresholds)
    print(f'plots clase {name} realizados')
    return metricas, plots


# Por cada prediccion se generan metricas por clase, por combinaciones binarias y por maximo
def metricas_dict(y_real, y_pred):
    metrics_dict = {}
    plot_dict = {}
    for i in range(3):
        pred = y_pred[:,i]
        real = y_real[:,i]
        metricas, plots = metrics_per_class(real, pred, str(i))
        metrics_dict.update(metricas)
        plot_dict.update(plots)

    y_binar = extract_max(y_pred.copy())
    for i in range(3):
        pred = y_binar[:,i]
        real = y_real[:,i]
        metrics_dict['f1_score_' + str(i)] = [metrics.f1_score(real, pred, 
                                                                average = 'weighted')]
        metrics_dict['precision_score_' + str(i)] = [metrics.precision_score(real, 
                                                                            pred, 
                                                                            average = 'weighted')]
        metrics_dict['recall_score_' + str(i)] = [metrics.recall_score(real, 
                                                                        pred, 
                                                                        average = 'weighted')]
        metrics_dict['accuracy_score_' + str(i)] = [metrics.accuracy_score(real, pred)]

    for combination in [[0,1], [0,2], [1,2]]:
        pred = extract_max(y_pred[:,combination])
        real = extract_max(y_real[:,combination])
        metrics_dict['f1_score' + str(combination)] = [metrics.f1_score(real, 
                                                                        pred, 
                                                                        average = 'weighted')]
        metrics_dict['precision_score' + str(combination)] = [metrics.precision_score(real, 
                                                                                    pred,
                                                                                    average = 'weighted')]
        metrics_dict['recall_score' + str(combination)] = [metrics.recall_score(real, 
                                                                                pred, 
                                                                                average = 'weighted')]
        metrics_dict['accuracy_score' + str(combination)] = [metrics.accuracy_score(real, pred)]
    print('metricas binarias calculadas')

    return metrics_dict, plot_dict


def class_report(y_real, y_pred, path):
    y_binar = extract_max(y_pred.copy())
    m = metrics.classification_report(y_real, y_binar, 
                                        target_names = ['normal', 'moderado', 'severo'], 
                                        output_dict = True)
    d = pd.DataFrame(m).transpose()
    d.to_csv(path)
