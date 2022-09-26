import os
import h5py as f
import otras_funciones.logs as logs
import funciones_evaluacion.evaluation as ev
import argparse
import numpy as np
import pickle
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf


def crear_modelo(input_shape, backbone_name, frozen_backbone_prop):
    if backbone_name == 'IncResNet':
        backbone = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name == 'EffNet3':
        backbone = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name == 'Xception':
        backbone = Xception(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(layers.Conv2D(3,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_inicial'))
    model.add(backbone)
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


def generate_index(subset_bool = False, trainprop = 0.8):
    if subset_bool:
        with open("/home/mr1142/Documents/scripts/img_class/indices/train_subset", "rb") as fp:
            index = pickle.load(fp)
    else:
        with open("/home/mr1142/Documents/scripts/img_class/indices/train", "rb") as fp:
            index = pickle.load(fp)

    np.random.shuffle(index)
    idtrain = index[:int(len(index)*trainprop)]
    idtest = index[int(len(index)*trainprop):]

    return idtrain, idtest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=3)
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name of the model")
    parser.add_argument('-s',
                        '--subset_bool',
                        type=bool,
                        default=False,
                        help="use a subset of 1000")
    parser.add_argument('-mo',
                        '--modelo',
                        type=str,
                        default='Xception',
                        help="which model")                      
    parser.add_argument('-f',
                        '--frozen_prop',
                        type=float,
                        default=0.7,
                        help="proportion of layers to frozen from backbone")
    parser.add_argument('-b',
                        '--batch',
                        type=int,
                        default=8,
                        help="batch")
    parser.add_argument('-lr',
                        '--lr',
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument('-m',
                        '--mask',
                        type=bool,
                        default=False,
                        help="apply mask")


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    name = args.name
    subset_bool = args.subset_bool
    backbone = args.modelo
    frozen_prop = args.frozen_prop
    batch = args.batch
    lr = args.lr
    mask = args.mask
    trainprop = 0.8
    epoch = 200
    pix = 512

    # DATAFRAME
    df = f.File("/datagpu/datasets/mr1142/cxr_consensus_dataset_nocompr.h5", "r")
    for key in df.keys():
        globals()[key] = df[key]

    # DATA GENERATORS
    idtrain, idtest = generate_index(subset_bool, trainprop)

    from funciones_imagenes.data_generator import DataGenerator as gen
    traingen = gen(X_train, y_train, batch, pix, idtrain, mask)
    testgen = gen(X_train, y_train, batch, pix, idtest, mask)

    # MODELO
    input_shape = (pix,pix,3)
    model = crear_modelo(input_shape, backbone, frozen_prop)    

    # Compilado
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                    loss = 'categorical_crossentropy',
                    metrics = ['BinaryAccuracy', 'Precision', 'AUC'])

    # NAME 
    name = name + '_' + backbone + '_' + 'fine-0' + str(frozen_prop)[2:] + '_batch-' + str(batch) + '_lr-' + str(lr)[2:]

    # CALLBACK
    callb = [logs.tensorboard(name), logs.early_stop(5)]

    # TRAIN
    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        callbacks = callb,
                        epochs = epoch,
                        shuffle = True)
    

    # GUARDAR MÉTRICAS
    import funciones_evaluacion.prediction as pred

    if subset_bool:
        # Guardar el train
        name = ev.save_training(history, name, 
                [backbone, frozen_prop, batch, lr, mask, trainprop, pix, subset_bool])
        print('TRAINING GUARDADO')

        # Guardar modelo
        model.save('/home/mr1142/Documents/Data/models/neumonia/' + name + '.h5')
        print('MODELO GUARDADO')

        # TEST (subset) - VALIDACION (completo)
        with open("/home/mr1142/Documents/scripts/img_class/indices/val_subset", "rb") as fp:
            val_index = pickle.load(fp)
        
        val_index.sort()
        ev.save_eval(name, ev.evaluate(model, X_train, y_train, val_index, mask = mask))
        print('EVALUATE GUARDADO')
        pred.save_metricas(name, model, X_train, y_train, val_index, mask=mask)
        print('METRICAS GUARDADO')
    else:
        # Guardar el train
        name = ev.save_training(history, name, 
                [backbone, frozen_prop, batch, lr, mask, trainprop, pix, subset_bool],
                subname = '_completo')
        print('TRAINING GUARDADO')

        # Guardar modelo
        model.save('/home/mr1142/Documents/Data/models/neumonia/' + name + '.h5')
        print('MODELO GUARDADO')

        # TEST (subset) - VALIDACION (completo)
        idtest.sort()
        results = ev.evaluate(model, X_train, y_train, idtest, mask=mask)
        ev.save_eval(name + '_val', results, subname = '_completo')
        print('EVALUATE GUARDADO')
        pred.save_metricas(name, model, X_train, y_train, idtest, mask=mask, subname = '_completo')
        print('METRICAS GUARDADO')

