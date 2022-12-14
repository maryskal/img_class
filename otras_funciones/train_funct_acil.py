import h5py as f
import otras_funciones.logs as logs
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf


def crear_modelo(input_shape, backbone_name, frozen_backbone_prop, layer = False, pix = 512):
    if backbone_name == 'IncResNet':
        backbone = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name == 'EffNet3':
        backbone = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name == 'Xception':
        backbone = Xception(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(layers.Conv2D(3,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_inicial'))
    model.add(backbone)
    if layer:
        model.add(layers.Conv2D(3000,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_salida'))
    model.add(layers.GlobalMaxPooling2D(name="general_max_pooling"))
    model.add(layers.Dropout(0.2, name="dropout_out_1"))
    model.add(layers.Dense(768, activation="elu"))
    model.add(layers.Dense(128, activation="elu"))
    model.add(layers.Dropout(0.2, name="dropout_out_2"))
    model.add(layers.Dense(32, activation="elu"))
    model.add(layers.Dense(3, activation="softmax", name="fc_out"))

    # Se coge una proporción del modelo que dejar fija
    fine_tune_at = int(len(backbone.layers)*frozen_backbone_prop)
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return model


def generate_index(trainprop = 0.8):
    with open("/home/mr1142/Documents/scripts/img_class/indices/ht_train_subset", "rb") as fp:
        index = pickle.load(fp)
    np.random.shuffle(index)
    idtrain = index[:int(len(index)*trainprop)]
    idtest = index[int(len(index)*trainprop):]
    return idtrain, idtest


def add_to_csv(data, path):
    df = pd.read_csv(path)
    df.loc[len(df.index)] = data
    df.to_csv(path, index = False)


def train(backbone, frozen_prop, lr, mask, layer = False, evaluation_type = 'internal'):
    if layer:
        batch = 5
    else:
        batch = 8
    epoch = 200
    pix = 512

    # DATAFRAME
    df = f.File("/datagpu/datasets/mr1142/cxr_consensus_dataset_nocompr.h5", "r")
    for key in df.keys():
        globals()[key] = df[key]

    # DATA GENERATORS
    idtrain, idtest = generate_index()

    from funciones_imagenes.data_generator import DataGenerator as gen
    traingen = gen(X_train, y_train, batch, pix, idtrain, mask)
    testgen = gen(X_train, y_train, batch, pix, idtest, mask)

    # MODELO
    input_shape = (pix,pix,3)
    model = crear_modelo(input_shape, backbone, frozen_prop, layer)    

    # Compilado
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                    loss = 'categorical_crossentropy',
                    metrics = ['BinaryAccuracy', 'Precision', 'AUC'])

    # CALLBACK
    callb = [logs.early_stop(5)]

    # TRAIN
    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        callbacks = callb,
                        epochs = epoch,
                        shuffle = True)
    

    # MÉTRICAS
    characteristics = [backbone, frozen_prop, batch, lr, mask, pix]
    #ev.save_train_in_table(history.history, 'ht', characteristics,
    #                            '/home/mr1142/Documents/Data/models/neumonia/ht/train_max_completo_layer.csv')

    if evaluation_type == 'internal':
        import funciones_evaluacion.prediction as pred
        import funciones_evaluacion.metrics_and_plots as met
    
        with open("/home/mr1142/Documents/scripts/img_class/indices/ht_val_subset", "rb") as fp:
                val_index = pickle.load(fp)

        # Ver resultados sobre el test
        y_pred = pred.prediction_tensor(model, X_train, val_index, mask)
        y_real = y_train[val_index]

        metricas, _ = met.metricas_dict(y_real, y_pred)
        add_to_csv(characteristics + list(metricas.values()), '/home/mr1142/Documents/Data/models/neumonia/ht/prediction_validation_metrics_completo_layer.csv')
        metricas['f1_score_mean']= (metricas['f1_score_0']+metricas['f1_score_1']+metricas['f1_score_2'])/3
        return metricas['f1_score_mean']

    if evaluation_type == 'external':
        import funciones_evaluacion.external_evaluation as ex_ev
        path = '/home/mr1142/Documents/Data/global_pneumonia_selection/val'
        images_names, prediction = ex_ev.prediction_tensor(model, path, mask = mask)
        df = ex_ev.results_dataframe(images_names, prediction)
        results = ex_ev.calculate_metrics(df, path)
        add_to_csv(characteristics[:-1] + 
                    [max(history.history['val_auc'])] + 
                    list(results[0].values()), 
                    '/home/mr1142/Documents/Data/models/neumonia/ht/prediction_validation_external_metrics.csv')
        return results[0]['auc_']


