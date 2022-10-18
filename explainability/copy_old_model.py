from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
import tensorflow as tf


def crear_modelo(input_shape = (512,512,3)):
    K.clear_session()
    inputs = layers.Input(shape=(512,512,1))
    x = layers.Conv2D(3,3,padding="same", activation='elu', name = 'conv_inicial')(inputs)
    backbone = Xception(weights=None, include_top=False, input_shape=input_shape, input_tensor = x)
    back_output = backbone.get_layer('block14_sepconv1_act').output
    x = layers.SeparableConv2D(2048,(3,3), name = 'block14_sepconv2', padding = 'same', use_bias = False)(back_output)
    x = layers.BatchNormalization(name = 'block14_sepconv2_bn')(x)
    x = layers.Activation(activation = 'relu', name = 'block14_sepconv2_act')(x)

    x = layers.GlobalMaxPooling2D(name="general_max_pooling")(x)
    x = layers.Dropout(0.2, name="dropout_out_1")(x)
    x = layers.Dense(768, activation="elu", name = 'dense')(x)
    x = layers.Dense(128, activation="elu", name = 'dense_1')(x)
    x = layers.Dropout(0.2, name="dropout_out_2")(x)
    x = layers.Dense(32, activation="elu", name = 'dense_2')(x)
    output = layers.Dense(3, activation="softmax", name="fc_out")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs= output , name="U-Net")
    return model


def cargar_pesos(old, new):
    for layer in new.layers:
        try:
            new.get_layer(layer.name).set_weights(old.get_layer(layer.name).get_weights())
        except:
            new.get_layer(layer.name).set_weights(old.get_layer('xception').get_layer(layer.name).get_weights())
            try: 
                new.get_layer(layer.name).set_weights(old.get_layer('xception').get_layer(layer.name).get_weights())
            except:
                print(str(layer.name) + ' not found')
    return new


def copy_model(old_model):
    new_model = crear_modelo()
    new_model = cargar_pesos(old_model, new_model)
    return new_model