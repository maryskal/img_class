import os
import otras_funciones.train_funct as tr
import argparse
import numpy as np
from mango import Tuner, scheduler


# search space for KNN classifier's hyperparameters
# n_neighbors can vary between 1 and 50, with different choices of algorithm
param_space = dict(backbone=['Xception', 'IncResNet', 'EffNet3'],
                    frozen_prop = np.arange(0,1, 0.1),
                    lr= np.arange(1e-5, 1e-3, 1e-5),
                    mask = [True, False])

@scheduler.serial
def objective(**params):
    print('--------NEW COMBINATION--------')
    print(params)
    results = []
    for x in range(3):
        results.append(tr.train(**params))
        print('results {}: {}'.format(x, results[x]))
    print('FINAL RESULTS {}'.format(np.mean(results)))
    return np.mean(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=3)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    tuner = Tuner(param_space, objective)
    results = tuner.maximize()
    print('best parameters:', results['best_params'])
    print('best accuracy:', results['best_objective'])