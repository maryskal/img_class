import os
import otras_funciones.train_funct as tr
import argparse
import json
import numpy as np
from mango import Tuner, scheduler


# PARAM SPACE
param_space = dict(backbone=['Xception', 'IncResNet', 'EffNet3'],
                    frozen_prop = np.arange(0,1, 0.1),
                    lr= np.arange(1e-5, 1e-3, 1e-5),
                    mask = [True, False])

# TUNER CONFIGURATION
# Early stop
def early_stop(results):
    '''
        stop if best objective does not improve for 5 iterations
        results: dict (same keys as dict returned by tuner.minimize/maximize)
    '''
    current_best = results['best_objective']
    patience_window = results['objective_values'][-6:]
    return min(patience_window) > current_best

# Configuration
conf_dict = dict(num_iteration=100, early_stopping = early_stop)


# OBJETIVE
# f1 score of tree trainings with the same model
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


# EXECUTION
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=3)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    tuner = Tuner(param_space, objective, conf_dict)
    results = tuner.maximize()

    print('best parameters:', results['best_params'])
    print('best f1score:', results['best_objective'])

    with open('/home/mr1142/Documents/Data/models/neumonia/ht/results.json', 'w') as j:
        json.dump(results, j)