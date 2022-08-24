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
    pr_cut= thresholds[np.where(precision == recall)][0] 
    return pr_max, pr_cut


def f1_score(precision, recall):
    return 2*(precision*recall)/(precision+recall)


# PLOTS
def AUC_plot(fpr, tpr, thresholds, auc):
    fig, ax = plt.subplots()
    i = np.argmax(tpr-fpr)
    th = thresholds[i]
    x = fpr[i]
    y = tpr[i]
    ax.plot(fpr,tpr, "g-", label="AUC="+str(round(auc, 2)))
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    try:
        ax.plot([x, x], [0, y], "r:")
        ax.plot([0, x], [y, y], "r:")
    except:
        print('plot except')
    ax.plot([x], [y], "ro", label="th="+str(round(th,2))) 
    ax.legend(loc=4)
    return fig


def pred_recall_plot(precision, recall, thresholds):
    fig, ax = plt.subplots()
    i = np.argmax(precision+recall)
    x = precision[i]
    y = recall[i]
    th = thresholds[i]
    ax.plot(recall, precision, "g-")
    ax.set_ylabel('Recall')
    ax.set_xlabel('Precision')
    try:
        ax.plot([x, x], [0, y], "r:")
        ax.plot([0, x], [y, y], "r:")
    except:
        print('plot except')
    ax.plot([x], [y], "ro", label="th="+str(round(th,2))) 
    ax.legend(loc=4)
    ax.set_title('Precision-Recall curve')
    return fig


def plot_precision_recall_vs_threshold(precision, recall, thresholds):
    fig, ax = plt.subplots()
    f1 = f1_score(precision, recall)
    x_f = thresholds[np.argmax(f1)]
    y_f = f1[np.argmax(f1)]
    x = thresholds[np.where(precision == recall)]
    y = precision[np.where(precision == recall)]
    ax.axis([0 ,1, np.min(precision), 1])
    ax.plot(thresholds, precision[:-1], "r-", label="Precision", linewidth=2)
    ax.plot(thresholds, recall[:-1], "b-", label="Recall", linewidth=2)
    ax.plot(thresholds, f1[:-1], "m--", label="F1 score", linewidth=2)
    ax.plot([x], [y], "bo", label="th="+str(round(x[0],2)))
    try:
        ax.plot([x, x], [0, y], "k:")
    except:
        print('plot except')
    ax.plot([x_f, x_f], [0, y_f], "k:")
    ax.plot([x_f], [y_f], "ro", label="th="+str(round(x_f,2))) 
    ax.set_xlabel("Threshold")
    ax.grid(True)
    ax.legend()
    return fig


def save_plot(plot, folder, title):
    plot.savefig(os.path.join(folder, title + '.png'))


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
    auc = metrics.auc(fpr, tpr)
    metricas['auc_' + name] = auc
    metricas['younden_'+ name] = younden_idx(real, pred)
    precision, recall, pr_thresholds = metrics.precision_recall_curve(real, pred)
    metricas['pr_max_'+ name], metricas['pr_cut_'+ name] = pred_recall_thres(precision, 
                                                                            recall, 
                                                                            pr_thresholds)
    print(f'metricas clase {name} calculadas')
    plots = {}
    plots['pred_rec_plot_' + name] = pred_recall_plot(precision, recall, pr_thresholds)
    plots['auc_plot_' + name] = AUC_plot(fpr, tpr, auc_thresholds, auc)
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
        metricas, plots = metrics_per_class(str(i), real, pred)
        metrics_dict.update(metricas)
        plot_dict.update(plots)

    y_binar = extract_max(y_pred.copy())
    for i in range(3):
        pred = y_binar[:,i]
        real = y_real[:,i]
        metrics_dict['f1_score_' + str(i)] = metrics.f1_score(real, pred, 
                                                                average = 'weighted')
        metrics_dict['precision_score_' + str(i)] = metrics.precision_score(real, 
                                                                            pred, 
                                                                            average = 'weighted')
        metrics_dict['recall_score_' + str(i)] = metrics.recall_score(real, 
                                                                        pred, 
                                                                        average = 'weighted')
        metrics_dict['accuracy_score_' + str(i)] = metrics.accuracy_score(real, pred)

    for combination in [[0,1], [0,2], [1,2]]:
        pred = extract_max(y_pred[:,combination])
        real = extract_max(y_real[:,combination])
        metrics_dict['f1_score' + str(combination)] = metrics.f1_score(real, 
                                                                        pred, 
                                                                        average = 'weighted')
        metrics_dict['precision_score' + str(combination)] = metrics.precision_score(real, 
                                                                                    pred,
                                                                                    average = 'weighted')
        metrics_dict['recall_score' + str(combination)] = metrics.recall_score(real, 
                                                                                pred, 
                                                                                average = 'weighted')
        metrics_dict['accuracy_score' + str(combination)] = metrics.accuracy_score(real, pred)
    print('metricas binarias calculadas')

    return metrics_dict, plot_dict


# El report automatico de sklearn
def class_report(y_real, y_pred, path):
    y_binar = extract_max(y_pred.copy())
    m = metrics.classification_report(y_real, y_binar, 
                                        target_names = ['normal', 'moderado', 'severo'], 
                                        output_dict = True)
    d = pd.DataFrame(m).transpose()
    d.to_csv(os.path.join(path, 'class_report.csv'))
