import os
import numpy as np

import cv2
import tensorflow.keras as keras
import funciones_imagenes.mask_funct as msk


def clahe(img):
    clahe = cv2.createCLAHE()
    img = np.uint8(img)
    final_img = clahe.apply(img)
    final_img = np.expand_dims(final_img, axis=-1)
    return final_img


def get_prepared_img(img, pix, mask = True, clahe_bool = False):
    if mask:
        import funciones_imagenes.mask_model as model
        img = msk.des_normalize(msk.apply_mask(img, model.model))
    img = msk.recolor_resize(img, pix)
    if clahe_bool:
        img = clahe(img)
    img = msk.normalize(img)
    return img
