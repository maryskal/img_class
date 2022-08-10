import os

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
    clahe = cv2.createCLAHE(clipLimit = 20)
    final_img = clahe.apply(img)
    return final_img


def get_prepared_img(img, pix):
    segmented = msk.des_normalize(msk.apply_mask(img, model))
    segmented = msk.recolor_resize(segmented, pix)
    segmented = msk.normalize(segmented)
    return segmented
