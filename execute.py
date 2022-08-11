import os
import h5py as f
import logs
import argparse
import numpy as np
from tensorflow.keras.applications import EfficientNetB3
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
    parser.add_argument('-f',
                        '--fine_tune',
                        type=int,
                        default=384,
                        help="layer of effNet")
    parser.add_argument('-b',
                        '--batch',
                        type=int,
                        default=8,
                        help="batch")
    parser.add_argument('-lr',
                        '--lr',
                        type=int,
                        default=1e-4,
                        help="learning rate")                    
    parser.add_argument('-tr',
                        '--train_split',
                        type=int,
                        default=0.8,
                        help="train proportion")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    name = args.name
    fine_tune_at = args.fine_tune
    batch = args.batch
    trainprop = args.train_split
    epoch = 200
    pix = 512
    opt = tf.keras.optimizers.Adam(learning_rate = args.lr)
    loss = loss = 'binary_crossentropy'
    met = ['BinaryAccuracy', 'Precision', 'AUC']

    from data_generator import DataGenerator as gen

    df = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")

    for key in df.keys():
        globals()[key] = df[key]


    input_shape = (pix,pix,3)
    conv_base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(layers.Conv2D(3,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_inicial'))
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="general_max_pooling"))
    model.add(layers.Dropout(0.2, name="dropout_out_1"))
    model.add(layers.Dense(768, activation="elu"))
    model.add(layers.Dense(256, activation="elu"))
    model.add(layers.Dense(128, activation="elu"))
    model.add(layers.Dropout(0.2, name="dropout_out_2"))
    model.add(layers.Dense(64, activation="elu"))
    model.add(layers.Dense(32, activation="elu"))
    model.add(layers.Dropout(0.2, name="dropout_out_3"))
    model.add(layers.Dense(16, activation="elu"))
    model.add(layers.Dense(3, activation="sigmoid", name="fc_out"))


    conv_base.trainable = True
    for layer in conv_base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=opt, loss = loss , metrics = met)

    ids = np.r_[0:len(df['X_train'])]
    np.random.shuffle(ids)
    idtrain = ids[:int(len(df['X_train'])*trainprop)]
    idtest = ids[int(len(df['X_train'])*trainprop):]

    traingen = gen(X_train, y_train, batch, pix, idtrain)
    testgen = gen(X_train, y_train, batch, pix, idtest)

    callb = [logs.tensorboard(name), logs.early_stop(5)]

    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        callbacks = callb,
                        epochs = epoch,
                        shuffle = True)

    model.save('/home/mr1142/Documents/Data/models/neumonia/' + name + '.h5')
