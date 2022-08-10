import pandas as pd
import h5py as f


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
