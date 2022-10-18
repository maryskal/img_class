import numpy as np
import argparse
import os
import tensorflow as tf
import cv2
import re
import explainability.grand_cam as gc
import explainability.mask_quantification as msk
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=3)
    parser.add_argument('-m',
                        '--model',
                        help="model name",
                        type=str,
                        default='')
    parser.add_argument('-im',
                        '--image',
                        help="images path",
                        type=str,
                        default='')
    parser.add_argument('-th',
                        '--threshold',
                        help="heatmap threshold",
                        type=float,
                        default=0.1)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_path = os.path.join('/home/mr1142/Documents/Data/models/neumonia',args.model + '.h5')
    images_path = args.image
    th = args.threshold

    # Comprobamos si el modelo es con mascara
    if bool(re.search('mask', args.model)):
        mask = True
    else:
        mask = False

    # Cargamos las imagenes a comprobar
    images = [filename for filename in os.listdir(images_path) if 
                filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    # Cargamos el modelo de mascara para las proporciones
    import funciones_imagenes.mask_model as mask_model
    proportions = []

    # Creamos el directorio donde se van a guardar los heatmap y las proporciones
    save_dir = os.path.join('/home/mr1142/Documents/Data/heatmaps', args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image_path in tqdm(images):
        # Grand cam image
        image = cv2.imread(os.path.join(images_path, image_path))
        model = tf.keras.models.load_model(model_path)
        grand_cam_im, heatmap = gc.apply_grandcam(model, mask, image)
        cv2.imwrite(os.path.join(save_dir, re.split('/', image_path)[-1]), np.array(grand_cam_im))

        # Heatmap inside mask
        mask_img = msk.apply_mask(image, mask_model.model)
        heatmap = cv2.resize(heatmap, (256, 256))
        binary_hm = (heatmap > th) *1
        proportions.append(msk.extract_proportion(binary_hm, mask_img))
    
    # Save proportions
    with open(os.path.join(save_dir, "proportions"), "wb") as fp:
         pickle.dump(proportions, fp)
