import pandas as pd
import h5py as f
import os

def save_training(history, name):
    datos = history.history
    name = name + '_auc-' + str(max(datos['val_auc']))[2:4]
    save_in_table(datos, name)
    path = '/home/mr1142/Documents/Data/models/neumonia/training_data'   
    pd.DataFrame(datos).to_csv(os.path.join(path, name + '_data.csv'), index = False)
    return name


def save_in_table(datos, name):
    path = '/home/mr1142/Documents/Data/models/neumonia/training_data/train_max.csv'
    df = pd.read_csv(path)
    for k, v in datos.items():
        datos[k] = [max(v)]
    row = pd.DataFrame(datos)
    row.insert(0, 'model_name', name)
    df = pd.concat([df, row], ignore_index=True)
    df.to_csv(path, index = False)


def evaluate(model):
    df = f.File("/home/rs117/covid-19/data/cxr_consensus_dataset_nocompr.h5", "r")
    for key in df.keys():
        globals()[key] = df[key]
    results = model.evaluate(X_val, y_val, batch_size=8)
    print(results)
    return results


def save_eval(name, results):
    path = '/home/mr1142/Documents/Data/models/neumonia/validation_results/image_class.csv'
    df = pd.read_csv(path)
    save = [name] + results
    df.loc[len(df.index)] = save
    df.to_csv(path, index = False)
