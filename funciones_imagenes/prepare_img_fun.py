import os
import numpy as np

import cv2
import tensorflow.keras as keras

import funciones_imagenes.mask_funct as msk
import funciones_imagenes.losses as ex


model = os.path.join('./modelos', 'unet_final_renacimiento_validation_6.h5')
model = keras.models.load_model(model, 
                                    custom_objects={"MyLoss": ex.MyLoss, 
                                                    "loss_mask": ex.loss_mask, 
                                                    "dice_coef_loss": ex.dice_coef_loss,
                                                    "dice_coef": ex.dice_coef})


def clahe(img):
    clahe = cv2.createCLAHE()
    img = np.uint8(img)
    final_img = clahe.apply(img)
    final_img = np.expand_dims(final_img, axis=-1)
    return final_img


def get_prepared_img(img, pix, mask = True, clahe_bool = False):
    if mask:
        img = msk.des_normalize(msk.apply_mask(img, model))
    img = msk.recolor_resize(img, pix)
    if clahe_bool:
        img = clahe(img)
    img = msk.normalize(img)
    return img
