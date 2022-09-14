import argparse
import os
import otras_funciones.logs as logs
import tensorflow as tf
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
lr = 1e-4
name = 'prueba_2'
fzl = 18
mask = True
pixels = 512

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name of the model")
    parser.add_argument('-lr',
                        '--lr',
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument('-m',
                        '--mask',
                        type=bool,
                        default=True,
                        help="apply mask")
    parser.add_argument('-fz',
                        '--frozen_layer',
                        type=int,
                        default=18,
                        help="frozen layer")       
    parser.add_argument('-px',
                        '--pixels',
                        type=int,
                        default=512,
                        help="pixels")
    parser.add_argument('-lo',
                        '--loss',
                        type=str,
                        default='basic',
                        help="loss")
    parser.add_argument('-mo',
                        '--model',
                        type=int,
                        default=1,
                        help="model")
    parser.add_argument('-aug',
                        '--augment',
                        type=bool,
                        default=True,
                        help="augmentation")  

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    name = args.name
    lr = args.lr
    mask = args.mask
    fzl = args.frozen_layer
    pixels = args.pixels
    loss = args.loss
    modelo = args.model
    augment = args.augment
    batch = 8
    epoch = 200

    #----------------------------------------------------
    import funciones_imagenes.pediatric_funct as fine

    # DATA ----------------------------------------------------
    df = fine.create_dataframe('train')
    train, test = train_test_split(df, test_size=0.2, stratify = df.real)

    # Agumentation
    if augment:
        traingen = fine.DataGenerator_augment(train, 8, pixels, mask)
        testgen = fine.DataGenerator_augment(test, 8, pixels, mask)
    else:
        traingen = fine.DataGenerator(train, 8, pixels, mask)
        testgen = fine.DataGenerator(test, 8, pixels, mask)

    # MODELO ----------------------------------------------------
    # Model selection, pixels, mask and frozen layer
    if modelo == 1:
        model = fine.modelo(pixels, fzl, mask)
    else:
        model = fine.modelo2(pixels, fzl, mask)

    # COMPILADO ----------------------------------------------------
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
    callb = [logs.tensorboard(name), logs.early_stop(5)]

    # TRAIN ----------------------------------------------------
    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        callbacks = callb,
                        epochs = epoch,
                        shuffle = True)

    # TRAIN AND SAVE MODEL ----------------------------------------------------
    import funciones_evaluacion.evaluation as ev
    # Guardar el train
    name = ev.save_training(history, name, 
            [name, fzl, '8', lr, mask, 0.8, pixels, None], '_unsupervised')
    print('TRAINING GUARDADO')

    # Guardar modelo
    model.save('/home/mr1142/Documents/Data/models/neumonia_pediatric/' + name + '.h5')
    print('MODELO GUARDADO')

    # TEST - VALIDACION ----------------------------------------------------
    results = model.evaluate(testgen, batch_size=batch)
    ev.save_eval(name + '_val', results, subname = '_unsupervised')
    print('EVALUATE GUARDADO')
