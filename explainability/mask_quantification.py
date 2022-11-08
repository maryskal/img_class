import funciones_imagenes.mask_funct as msk
import explainability.grand_cam as gc
import numpy as np


def apply_mask(img, model):
    # Paso la imagen a escala de grises
    img = msk.recolor(img)
    # Creo una nueva imagen con las dimensiones de entrada al modelo
    img_2 = msk.normalize(msk.recolor_resize(img, 256))[np.newaxis,...]
    # Genero la mascara
    mask = model.predict(img_2, verbose = 0)[0,...,0]
    mask = msk.quitar_trozos(mask > 0.5)
    return mask


def extract_proportion(heatmap, mask, th = 0.1):
    binary_hm = (heatmap > th) *1
    suma = binary_hm + mask
    external_activation = binary_hm - (suma==2)*1
    external_area = (mask == 0)*1
    proportion = len(np.where(external_activation.flatten() == 1)[0])/len(np.where(external_area.flatten() == 1)[0])
    return proportion

 
def list_proportions(image_list, model, mask):
    import funciones_imagenes.mask_model as mask_model
    proportions = []
    for image in image_list:
        mask_img = apply_mask(image, mask_model.model)
        _, heatmap = gc.apply_grandcam(model, mask, image)
        proportions.append(extract_proportion(heatmap, mask_img))
    return proportions