import os
import re
import tensorflow as tf
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import math
import funciones_imagenes.prepare_img_fun as fu
import funciones_imagenes.mask_funct as msk
import albumentations as A


# DATAFRAME ---------------------------------------------------------------

def create_dataframe(folder, path = '/home/mr1142/Documents/Data/chest_xray'):
    path = os.path.join('/home/mr1142/Documents/Data/chest_xray', folder)
    for fold in os.listdir(path):
        globals()[fold] = {}
        imgs = os.listdir(os.path.join(path, fold))
        globals()[fold]['path'] = [os.path.join(path, fold)] * len(imgs)
        globals()[fold]['img_name'] = imgs
        globals()[fold]['normal'] = [1 if fold == 'NORMAL' else 0 for _ in range(len(imgs))]
        globals()[fold]['viral'] = [1 if re.search('virus', imgs[i]) else 0 for i in range(len(imgs))]
        globals()[fold]['bacteria'] = [1 if re.search('bacteria', imgs[i]) else 0 for i in range(len(imgs))]
        globals()[fold]['real'] = [0 if fold == 'NORMAL' else 1 if re.search('virus', imgs[i]) else 2 for i in range(len(imgs))]

    for k, v in PNEUMONIA.items():
        v.extend(NORMAL[k])

    df = pd.DataFrame(PNEUMONIA)

    return df


# DATA AUGMENTATION ------------------------------------------------------------

def albumentation(input_image):
    transform = A.Compose([
        A.Rotate(limit=90, border_mode = None, interpolation=2, p=1),
        A.OneOf([
            A.RandomCrop(p= 1, width=230, height=230),
            A.GridDistortion (num_steps=5, distort_limit=0.3, interpolation=2, border_mode=None, p=1),
            A.MotionBlur(blur_limit=7, always_apply=False, p=0.5),
            A.ElasticTransform(alpha=0.5, sigma=50, alpha_affine=50, interpolation=1, border_mode=None, always_apply=False, p=1)
        ], p=0.8),
    ])
    transformed = transform(image=input_image.astype(np.float32))
    input_image = fu.get_prepared_img(transformed['image'])
    return input_image


# DATA GENERATOR ---------------------------------------------------------------

class DataGenerator(Sequence):
    
    def __init__(self, df, batch_size, pix, mask):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.batch_size = batch_size
        self.pix = pix
        self.mask = mask

    def __len__(self):
        # numero de batches
        return math.ceil(len(self.df['real']) / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        # Lo que hago es recorrer el indice
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size].reset_index(drop = True)
        batch_x = np.zeros((len(batch_df), self.pix, self.pix, 1))
        batch_y = np.array(batch_df[['normal', 'viral', 'bacteria']])
        for i in range(len(batch_df)):
            try:
                img = cv2.imread(os.path.join(batch_df['path'][i], batch_df.img_name[i]))
                batch_x[i,...] = fu.get_prepared_img(img, self.pix, mask = self.mask, clahe_bool=True)
            except:
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i,...] = msk.normalize(img)
                print('e')
        # batch_x = fu.augment_tensor(batch_x)
        return batch_x, batch_y


class DataGenerator_augment(Sequence):
    
    def __init__(self, df, batch_size, pix, mask):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.batch_size = batch_size
        self.pix = pix
        self.mask = mask

    def __len__(self):
        # numero de batches
        return math.ceil(len(self.df['real']) / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        # Lo que hago es recorrer el indice
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size].reset_index(drop = True)
        batch_x = np.zeros((len(batch_df), self.pix, self.pix, 1))
        batch_x_augment = np.zeros((len(batch_df), self.pix, self.pix, 1))
        batch_y = np.array(batch_df[['normal', 'viral', 'bacteria']])
        for i in range(len(batch_df)):
            try:
                img = cv2.imread(os.path.join(batch_df['path'][i], batch_df.img_name[i]))
                batch_x[i,...] = fu.get_prepared_img(img, self.pix, mask = self.mask, clahe_bool=True)
                batch_x_augment[i,...] = albumentation(img)
            except:
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i,...] = msk.normalize(img)
                batch_x_augment[i,...] = albumentation(img)
                print('e')

        batch_x = np.concatenate((batch_x, batch_x_augment), axis = 0)
        batch_y = np.concatenate((batch_y, batch_y), axis = 0)
        # batch_x = fu.augment_tensor(batch_x)
        return batch_x, batch_y


# MODELO ---------------------------------------------------------------

def squeeze_and_excitation(inputs, ratio=16):
    b, _, _, c = inputs.shape
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(c//ratio, activation="relu", use_bias=False)(x)
    x = tf.keras.layers.Dense(c, activation="sigmoid", use_bias=False)(x)
    x = tf.expand_dims(x, 1)
    x = tf.expand_dims(x, 1)
    x = inputs * x
    return x
 

def downsample_block(downsampling_output):
    pix = downsampling_output.shape[2]
    deep = downsampling_output.shape[3]
    name = re.sub(':', '', downsampling_output.name) + '_new'
    x = tf.keras.layers.Conv2D(deep*2,3, padding = 'same', name = name + '_conv')(downsampling_output)
    x = squeeze_and_excitation(x)
    x = tf.keras.layers.MaxPool2D(int(pix/8), name = name + '_max')(x)
    maxpool = tf.keras.layers.GlobalMaxPooling2D()(x)
    return maxpool


def global_max_concat(maxpool_output, previous_layer):
    dense = tf.keras.layers.Dense(128, activation="elu")(maxpool_output)
    unification = tf.keras.layers.concatenate([dense, previous_layer])
    return unification


def modelo(pixels, fine_tune_at = 18, mask = False):
    if mask:
        # model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + 'mask_' + str(pixels) + '.h5'
        model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + str(pixels) + '.h5'
    else:
        model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + str(pixels) + '.h5'
    backbone = tf.keras.models.load_model(model_path)

    inputs = backbone.input

    downsampling_pretrained_output = backbone.layers[18].output
    intermedium = downsample_block(downsampling_pretrained_output)

    dropout_1 = tf.keras.layers.Dropout(0.2, name = "drop_out_1")(intermedium)

    dense_1 = tf.keras.layers.Dense(768, activation="elu")(dropout_1)
    dense_union_1 = global_max_concat(downsample_block(backbone.layers[15].output), dense_1)
    dense_2 = tf.keras.layers.Dense(128, activation="elu")(dense_union_1)
    dense_union_2 = global_max_concat(downsample_block(backbone.layers[11].output), dense_2)
    dense_2 = tf.keras.layers.Dense(128, activation="elu")(dense_union_2)

    dropout_2 = tf.keras.layers.Dropout(0.2, name="dropout_out_2")(dense_2)

    dense_final = tf.keras.layers.Dense(32, activation="elu")(dropout_2)
    outputs = tf.keras.layers.Dense(3, activation="sigmoid", name="fc_out")(dense_final)

    model = tf.keras.Model(inputs, outputs, name="U-Net")

    backbone.trainable = True
    print('\ntrainable variables: {}'.format(len(backbone.trainable_variables)))

    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return model


def modelo2(pixels, fine_tune_at = 18, mask = False):
    if mask:
        # model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + 'mask_' + str(pixels) + '.h5'
        model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + str(pixels) + '.h5'
    else:
        model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + str(pixels) + '.h5'
    
    backbone = tf.keras.models.load_model(model_path)

    inputs = backbone.input

    downsampling_pretrained_output = backbone.layers[18].output
    maxpool_intermedium = tf.keras.layers.GlobalMaxPooling2D()(downsampling_pretrained_output)

    dropout_1 = tf.keras.layers.Dropout(0.2, name = "drop_out_1")(maxpool_intermedium)

    dense_1 = tf.keras.layers.Dense(768, activation="elu")(dropout_1)
    dense_union_1 = global_max_concat(tf.keras.layers.GlobalMaxPooling2D()(backbone.layers[15].output), dense_1)
    dense_2 = tf.keras.layers.Dense(128, activation="elu")(dense_union_1)
    dense_union_2 = global_max_concat(tf.keras.layers.GlobalMaxPooling2D()(backbone.layers[11].output), dense_2)
    dense_2 = tf.keras.layers.Dense(128, activation="elu")(dense_union_2)

    dropout_2 = tf.keras.layers.Dropout(0.2, name="dropout_out_2")(dense_2)

    dense_final = tf.keras.layers.Dense(32, activation="elu")(dropout_2)
    outputs = tf.keras.layers.Dense(3, activation="sigmoid", name="fc_out")(dense_final)

    model = tf.keras.Model(inputs, outputs, name="U-Net")

    backbone.trainable = True
    print('\ntrainable variables: {}'.format(len(backbone.trainable_variables)))

    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return model


# LOSSES ---------------------------------------------------------------

def custom_binary_loss(y_true, y_pred): 
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_true = K.cast(y_true, 'float32')
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  # Cancels out when target is 1 
    term_1 = y_true * K.log(y_pred + K.epsilon()) # Cancels out when target is 0
    suma = term_0 + term_1
    return -K.mean(suma, axis=1)+K.std(suma, axis = 1)


def custom_binary_loss_2(y_true, y_pred): 
    y_true = K.cast(y_true, 'float32')
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    san = K.sum(1-y_true, axis = 0)
    enf = K.sum(y_true, axis = 0)
    dif_sanos = (1 - y_true) * K.abs(y_true-y_pred)  # Cancels out when target is 1 
    dif_sanos = K.sum(dif_sanos, axis = 0)/san
    dif_enf = y_true * K.abs(y_true-y_pred) # Cancels out when target is 0
    dif_enf = K.sum(dif_enf, axis = 0)/enf
    suma = dif_enf + dif_sanos + K.abs(dif_enf-dif_sanos)
    return K.mean(suma)+K.std(suma)