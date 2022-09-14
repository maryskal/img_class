import os
import cv2
import numpy as np
import otras_funciones.logs as logs
import tensorflow as tf
from sklearn.model_selection import train_test_split


def train(modelo, frozen_layer, lr, pixels, loss, mask, augment):
    batch = 8
    epoch = 200
    name = modelo + '_' + str(frozen_layer) + '_' + str(lr) + '_' + str(pixels) + '_' + loss + '_' + str(mask)
    #----------------------------------------------------
    import funciones_imagenes.pediatric_funct as fine

    # DATA ----------------------------------------------------
    df = fine.create_dataframe('train')
    train, test = train_test_split(df, test_size=0.2, stratify = df.real)

    # Agumentation, pixels and mask
    if augment:
        traingen = fine.DataGenerator_augment(train, 8, pixels, mask)
        testgen = fine.DataGenerator_augment(test, 8, pixels, mask)
    else:
        traingen = fine.DataGenerator(train, 8, pixels, mask)
        testgen = fine.DataGenerator(test, 8, pixels, mask)

    # MODELO ----------------------------------------------------
    # Type of model, pixels and frozen layer
    if modelo == 1:
        model = fine.modelo(pixels, frozen_layer)
    else:
        model = fine.modelo2(pixels, frozen_layer)

    # Loss
    if loss == 'loss1':
        ls = fine.custom_binary_loss
    elif loss == 'loss2':
        ls = fine.custom_binary_loss_2
    else:
        ls = 'binary_crossentropy'

    # Compilado
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                    loss = ls,
                    metrics = ['binary_accuracy', 'Precision', 'AUC'])

    # CALLBACK ----------------------------------------------------
    callb = [logs.early_stop(5)]

    # TRAIN ----------------------------------------------------
    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        callbacks = callb,
                        epochs = epoch,
                        shuffle = True)

    # PREDICTION ----------------------------------------------------
    import funciones_evaluacion.metrics_and_plots as met
    import funciones_imagenes.prepare_img_fun as fu

    # Ver resultados sobre el test
    y_real = np.array(test[['normal', 'viral', 'bacteria']])
    y_pred = np.zeros((len(test), 3))
    for i in range(len(test)):
        img = cv2.imread(os.path.join(test['path'].iloc[i], test.img_name.iloc[i]))
        y_pred[i,...] = model.predict(np.expand_dims(fu.get_prepared_img(img, pixels, mask, clahe_bool=True), 0))

    # Nos quedamos con auc_mean
    metricas, _ = met.metricas_dict(y_real, y_pred)
    metricas['auc_mean']= (metricas['auc_0']+metricas['auc_1']+metricas['auc_2'])/3

    # SAVE TRAIN ----------------------------------------------------
    name = name + '_' + str(round(metricas['auc_mean'],2))[2:]

    import funciones_evaluacion.evaluation as ev

    # Guardar el train
    name = ev.save_training(history, name, 
            [name, frozen_layer, '8', lr, mask, 0.8, pixels, None], '_unsupervised')
    print('TRAINING GUARDADO')

    # EVALUATION ----------------------------------------------------
    results = model.evaluate(testgen, batch_size=batch)
    ev.save_eval(name + '_val', results, subname = '_unsupervised')
    print('EVALUATE GUARDADO')

    # MODEL SAVE ----------------------------------------------------
    model.save('/home/mr1142/Documents/Data/models/neumonia_pediatric/' + name + '.h5')
    print('MODELO GUARDADO')

    return metricas['auc_score_mean']
