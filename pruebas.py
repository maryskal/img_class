import os
import h5py as f
import numpy as np
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
name = ''
subset = True
modelo = 'EffNet3'
fine_tune = 0.8
batch = 8
lr = 1e-4
mask = False
trainprop = 0.8
epoch = 200
pix = 512

# DATAFRAME
df = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")

for key in df.keys():
    globals()[key] = df[key]

# DATA GENERATORS
from img_class.funciones_imagenes.data_generator import DataGenerator as gen

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
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf
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

# TRAIN
history = model.fit(traingen, 
                    validation_data = testgen,
                    batch_size = batch,
                    epochs = 2,
                    shuffle = True)

model.save('/home/mr1142/Documents/Data/models/neumonia/' + name + '.h5')

idx = 0
index = idtrain[idx * 8:(idx + 1) * 8]
index.sort()
print(index)
temp_x = X_train[index]
batch_y = y_train[index]
batch_x = np.zeros((temp_x.shape[0], pix, pix, 1))
import funciones_imagenes.mask_funct as msk
import funciones_imagenes.prepare_img_fun as fu

for i in range(temp_x.shape[0]):
    try:
        batch_x[i] = fu.get_prepared_img(temp_x[i], pix, mask)
    except:
        img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
        batch_x[i] = msk.normalize(img)
        print('e')

for i, set in enumerate(traingen):
    a = set[0]
    print(np.min(a))
    print(np.max(a))

import pandas as pd
datos = history.history
datos = pd.DataFrame(datos)
path = '/home/mr1142/Documents/Data/models/neumonia/training_data'
if not os.path.exists(os.path.join(path, name)):
    os.makedirs(os.path.join(path, name))
datos.to_csv(os.path.join(path, name, name + '_data.csv'))