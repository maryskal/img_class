
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    import h5py as f
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras import layers
    from tensorflow.keras import models
    import tensorflow as tf
    from data_generator import TrainDataGenerator as tr_gen
    from data_generator import TestDataGenerator as ts_gen


    df = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")

    for key in df.keys():
        globals()[key] = df[key]


    pix = 512
    name = ''
    epoch = 200
    batch = 8
    fine_tune_at = 384
    lr = 1e-4
    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    loss = loss = 'binary_crossentropy'
    met = ['BinaryAccuracy', 'Precision', 'AUC']


    input_shape = (pix,pix,3)
    conv_base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(layers.Conv2D(3,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_inicial'))
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="general_max_pooling"))
    model.add(layers.Dropout(0.2, name="dropout_out_1"))
    model.add(layers.Dense(768, activation="elu"))
    model.add(layers.Dense(256, activation="elu"))
    model.add(layers.Dense(128, activation="elu"))
    model.add(layers.Dropout(0.2, name="dropout_out_2"))
    model.add(layers.Dense(64, activation="elu"))
    model.add(layers.Dense(32, activation="elu"))
    model.add(layers.Dropout(0.2, name="dropout_out_3"))
    model.add(layers.Dense(16, activation="elu"))
    model.add(layers.Dense(3, activation="sigmoid", name="fc_out"))


    conv_base.trainable = True
    for layer in conv_base.layers[:fine_tune_at]:
        layer.trainable = False


    model.compile(optimizer=opt, loss = loss , metrics = met)

    traingen = tr_gen(X_train, y_train, batch, pix, 0.8)
    testgen = ts_gen(X_train, y_train, batch, pix, 0.2)


    log_dir = "/home/mr1142/Documents/Data/logs/fit/" + name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                        update_freq='batch',
                                                        histogram_freq=1)

    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        #callbacks = tensorboard_callback,
                        epochs = epoch,
                        shuffle = True)

    model.save('/home/mr1142/Documents/Data/models/neumonia/' + name + '.h5')
