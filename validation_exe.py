import os
import re
import h5py as f
import argparse
from tensorflow import keras

def predicciones_modelo(model_name):
    if bool(re.search('mask', model_name)):
        mask = True
    else:
        mask = False

    model = os.path.join('/home/mr1142/Documents/Data/models/neumonia', model_name)
    model = keras.models.load_model(model)

    dataframes = f.File("/datagpu/datasets/mr1142/cxr_consensus_dataset_nocompr.h5", "r")
    for key in dataframes.keys():
        globals()[key] = dataframes[key]
    
    results = ev.evaluate(model, X_val, y_val, list(range(len(y_val))), mask=mask)
    ev.save_eval(model_name[:-3], results)
    pred.save_metricas(model_name[:-3], model, X_val, y_val, list(range(len(y_val))), mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=3)
    parser.add_argument('-mo',
                        '--model_name',
                        help="nombre del modelo",
                        type=str,
                        default='prueba_EffNet3_fine-03_batch-8_lr-001_auc-94.h5')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_name = args.model_name
    import funciones_complementarias.prediction as pred
    import funciones_complementarias.evaluation as ev
    predicciones_modelo(model_name)