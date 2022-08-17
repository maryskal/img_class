import os
import h5py as f
import logs
import argparse
import numpy as np
import pickle
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf


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
                        '--subset',
                        type=bool,
                        default=True,
                        help="train with a subset of 1000")
    parser.add_argument('-mo',
                        '--model',
                        type=str,
                        default='EffNet3',
                        help="train with a subset of 1000")                      
    parser.add_argument('-f',
                        '--fine_tune',
                        type=float,
                        default=0.8,
                        help="proportion of layers to tune from backbone")
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
    subset = args.subset
    modelo = args.model
    fine_tune = args.fine_tune
    batch = args.batch
    lr = args.lr
    mask = args.mask
    trainprop = 0.8
    epoch = 200
    pix = 512

    # DATAFRAME
    df = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")

    for key in df.keys():
        globals()[key] = df[key]

    # DATA GENERATORS
    from data_generator import DataGenerator as gen

    if subset:
        with open("/home/mr1142/Documents/img_class/index_subset", "rb") as fp:
            index = pickle.load(fp)
    else:
        with open("/home/mr1142/Documents/img_class/index", "rb") as fp:
            index = pickle.load(fp)

    np.random.shuffle(index)
    idtrain = index[:int(len(index)*trainprop)]
    idtest = index[int(len(index)*trainprop):]

    traingen = gen(X_train, y_train, batch, pix, idtrain, mask)
    testgen = gen(X_train, y_train, batch, pix, idtest, mask)

    # MODELO
    # Se elige que modelo usar
    input_shape = (pix,pix,3)
    if modelo == 'IncResNet':
        conv_base = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif modelo == 'EffNet3':
        conv_base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    elif modelo == 'EffNet7':
        conv_base = EfficientNetB7(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(layers.Conv2D(3,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_inicial'))
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="general_max_pooling"))
    model.add(layers.Dropout(0.2, name="dropout_out_1"))
    model.add(layers.Dense(768, activation="elu"))
    model.add(layers.Dense(128, activation="elu"))
    model.add(layers.Dropout(0.2, name="dropout_out_2"))
    model.add(layers.Dense(32, activation="elu"))
    model.add(layers.Dense(3, activation="softmax", name="fc_out"))

    # Fine tunning
    # Se coge una proporción del modelo que dejar fija
    fine_tune_at = int(len(conv_base.layers)*fine_tune)
    conv_base.trainable = True
    for layer in conv_base.layers[:fine_tune_at]:
        layer.trainable = False

    # Compilado
    name = name + '_' + modelo + '_' + 'fine-0' + str(fine_tune)[2:] + '_batch-' + str(batch) + '_lr-' + str(lr)[2:]
    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    loss = loss = 'categorical_crossentropy'
    met = ['BinaryAccuracy', 'Precision', 'AUC']
    
    model.compile(optimizer=opt, loss = loss , metrics = met)

    # CALLBACK
    callb = [logs.tensorboard(name), logs.early_stop(10)]

    # TRAIN
    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        callbacks = callb,
                        epochs = epoch,
                        shuffle = True)

    model.save('/home/mr1142/Documents/Data/models/neumonia/' + name + '.h5')

