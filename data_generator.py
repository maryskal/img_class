from tensorflow.keras.utils import Sequence
import numpy as np
import math
import funciones_imagenes.prepare_img_fun as fu
import funciones_imagenes.mask_funct as msk

class TrainDataGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size, pix, train_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.pix = pix
        self.size = train_size

    def __len__(self):
        # numero de batches
        return math.ceil(len(self.x)*self.size / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        temp_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        batch_x = np.zeros((self.batch_size, self.pix, self.pix, 1))
        for i in range(self.batch_size):
            long = len(temp_x[i][temp_x[i]>0.1][temp_x[i][temp_x[i]>0.1]<0.9])/len(temp_x[i].flatten())
            if long < 0.5:
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i] = msk.normalize(img)
            else:
                batch_x[i] = fu.get_prepared_img(temp_x[i], self.pix)
        return np.array(batch_x), np.array(batch_y)

class TestDataGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size, pix, test_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.pix = pix
        self.size = test_size

    def __len__(self):
        # numero de batches
        return math.ceil(len(self.x)*self.size / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        idx = idx + len(self.x)*(1-self.size)/self.batch_size
        # batch 0: idx = 0 + train_batches
        temp_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        batch_x = np.zeros((self.batch_size, self.pix, self.pix, 1))
        for i in range(self.batch_size):
            long = len(temp_x[i][temp_x[i]>0.1][temp_x[i][temp_x[i]>0.1]<0.9])/len(temp_x[i].flatten())
            if long < 0.5:
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i] = msk.normalize(img)
            else:
                batch_x[i] = fu.get_prepared_img(temp_x[i], self.pix)
        return np.array(batch_x), np.array(batch_y)