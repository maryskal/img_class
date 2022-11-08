import os
import re
import argparse
from tensorflow import keras
import funciones_evaluacion.external_evaluation as ev



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=0)
    parser.add_argument('-p',
                        '--path',
                        help="images path",
                        type=str,
                        default='/home/mr1142/Documents/Data/global_pneumonia_selection/val')
    parser.add_argument('-m',
                        '--model_name',
                        help="model to apply",
                        type=str,
                        default='DEFINITIVO_2_mask_Xception_fine-04_batch-8_lr-0001_auc-99')
    parser.add_argument('-sp',
                        '--save_plots',
                        help="save results plots",
                        type=bool,
                        default=False)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    path = args.path
    save_plots = args.save_plots
    modelos = [args.model_name]

    # p = '/home/mr1142/Documents/Data/models/neumonia'
    # modelos = os.listdir(p)
    # modelos = [modelo[:-3] for modelo in modelos if os.path.isfile(os.path.join(p, modelo))]
    # modelos = [modelo for modelo in modelos if bool(re.search('DEFINITIVO', modelo))]

    for model_name in modelos:
        print(model_name)

        model_path = '/home/mr1142/Documents/Data/models/neumonia/'+ model_name + '.h5'
        if bool(re.search('mask', model_name)):
            mask = True
        else:
            mask = False
        
        model = keras.models.load_model(model_path)

        images_names, prediction = ev.prediction_tensor(model, path, mask = mask)

        df = ev.results_dataframe(images_names, prediction)
        df.to_csv(os.path.join(path,'model_results', model_name + '_results.csv'), index = False)

        results = ev.calculate_metrics(df, path)
        ev.save_in_csv(path, model_name, results)

        if save_plots:
            ev.save_plots_fun(results, model_name+'_external')
            
