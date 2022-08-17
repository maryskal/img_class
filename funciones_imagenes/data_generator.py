from tensorflow.keras.utils import Sequence
import numpy as np
import math
import funciones_imagenes.prepare_img_fun as fu
import funciones_imagenes.mask_funct as msk

class DataGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size, pix, index, mask):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.pix = pix
        self.index = index
        self.mask = mask

    def __len__(self):
        # numero de batches
        return math.ceil(len(self.index) / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        # Lo que hago es recorrer el indice
        index = self.index[idx * self.batch_size:(idx + 1) * self.batch_size]
        index.sort()
        temp_x = self.x[index]
        batch_y = self.y[index]
        batch_x = np.zeros((temp_x.shape[0], self.pix, self.pix, 1))
        for i in range(temp_x.shape[0]):
            try:
                batch_x[i] = fu.get_prepared_img(temp_x[i], self.pix, self.mask)
            except:
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i] = msk.normalize(img)
                print('e')
        return np.array(batch_x), np.array(batch_y)