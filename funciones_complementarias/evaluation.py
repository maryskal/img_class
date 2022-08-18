import pandas as pd
import h5py as f
import os

def save_training(history, name, otros_datos):
    datos = history.history
    name = name + '_auc-' + str(max(datos['val_auc']))[2:4]
    save_train_in_table(datos, name, otros_datos)
    path = '/home/mr1142/Documents/Data/models/neumonia/training_data'   
    pd.DataFrame(datos).to_csv(os.path.join(path, name + '_data.csv'), index = False)
    return name


def save_train_in_table(datos, name, otros_datos):
    path = '/home/mr1142/Documents/Data/models/neumonia/training_data/train_max.csv'
    df = pd.read_csv(path)
    values = [name]
    values.extend(otros_datos)
    for v in datos.values():
        values.append(max(v))
    df.loc[len(df)] = values
    df.to_csv(path, index = False)


def evaluate(model, X_val, y_val, index, batch = 8, pix = 512, mask = False):
    from funciones_imagenes.data_generator import DataGenerator as gen
    generator = gen(X_val, y_val, batch, pix, index, mask)
    results = model.evaluate(generator, batch_size=batch)
    print(results)
    return results


def save_eval(name, results):
    path = '/home/mr1142/Documents/Data/models/neumonia/validation_results/image_class_evaluation.csv'
    df = pd.read_csv(path)
    save = [name] + results
    df.loc[len(df.index)] = save
    df.to_csv(path, index = False)