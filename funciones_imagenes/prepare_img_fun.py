import os
import numpy as np

import cv2
import funciones_imagenes.mask_funct as msk
from skimage import exposure, filters


def equalize(img, clip_limit=0.01, output_type='uint16'):
    if img.dtype is np.dtype(np.float32):
        img_norm = img / img.max()                                    # Format adaptation
    else:
        img_norm = img.astype('float32') / np.iinfo(img.dtype).max                  # Format adaptation
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=clip_limit)        # CLAHE
    img_clahe_median = filters.median(img_clahe,np.ones((3,3,1))).astype('float32')   # Median Filter

    lower, upper = np.percentile(img_clahe_median.flatten(), [2, 98])
    img_clip = np.clip(img_clahe_median,lower, upper)
    img_out = (img_clip - lower)/(upper - lower)

    if output_type is not None:
        max_val=np.iinfo(output_type).max
        img_out=(max_val*img_out).astype(output_type)
    else:
        max_val=1.0
    return img_out


def clahe(img):
    clahe = cv2.createCLAHE()
    img = np.uint8(img)
    final_img = clahe.apply(img)
    final_img = np.expand_dims(final_img, axis=-1)
    return final_img


def get_prepared_img(img, pix, mask = True, clahe_bool = False, equalize_bool = False):
    if equalize_bool:
        img = equalize(img)
    if mask:
        import funciones_imagenes.mask_model as model
        img = msk.des_normalize(msk.apply_mask(img, model.model))
    img = msk.recolor_resize(img, pix)
    if clahe_bool:
        img = clahe(img)
    img = msk.normalize(img)
    return img
