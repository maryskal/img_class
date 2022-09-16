# DATOS
## Dataset completo
Se utilizan las imagenes etiquetadas por ACIL. Son 59439 imagenes con las siguientes proporciones:

- Normal: 0.469590672790592
- Moderado: 0.3504601356012046
- Grave: 0.17994919160820336

Se balancean (SITK/neural_networks/img_class/seleccion_subset) para que todas tengan la misma cantidad,
queda un subset de 32088 imagenes (indices/train)

## Dataset parcial
Además, para no realizar diferentes pruebas sobre todo el dataset completo se crea un subset de entrenamiento con
333 imagenes de cada clase (***indices/train_subset***) y se crea otro subset de test(***indices/val_subset***), también de 1000 imagenes
para testear los modelos entrenados con estas imagenes.

Los resultados del entrenamiento de estos modelos se guardaron en: 
- /home/mr1142/Documents/Data/models/neumonia/training_data/train_max.csv

Los resultados del test de los modelos entrenados de esta manera se guardaron en un dataframe:
- /home/mr1142/Documents/Data/models/neumonia/validation_results/prediction_validation_metrics.csv
- /home/mr1142/Documents/Data/models/neumonia/validation_results/image_class_evaluation.csv

Además estos modelos evaluaron también sobre todo el resto del dataset 59439-1000 (***indices/val_rest***).

## Dataset hyperparameter tunning

Este dataset se realizó con 1000 imágenes de cada clase para el entrenamiento (indices/ht_train_subset)y
con 1000 imagenes de cada clase para la validación (***indices/ht_val_subset***).


# PREPROCESADO
Las imágenes ya venían con un preprocesado del dataset de ACIL que se puede comprobar en (github.com/acil-bwh/slowdown-covid19)

En algunos casos se aplicó un modelo que enmascara el tórax antes de todo el preprocesado (***funciones_imagenes/mask_function.py***):
- Paso a escala de grises
- Resize a 256,256
- Aplicación del modelo y extracción de la máscara
- Resize de la máscara al tamaño de la imagen original
- Quitar agujeros y labels extra a la mascara
- Aplicar la máscara sobre la imagen original
- Desnormalizar el resultado

Además a todas las imágenes se les aplicó (***funciones_imagenes/prepare_img_fun.py***):
- Paso a escala de grises: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) -> (pix,pix)
- Resize: img = cv2.resize(img, (512,512)) -> (512,512)
- Expandir dimension: img = np.expand_dims(img, axis=-1)
- Normalización con z-score: (img - np.mean(img))/ np.std(img)

Después de hacer todas estas pruebas se crearon otros dos dataset para ejecutar ***ht_exe.py***, que aplica mango
para la gestión de los hiperparámetros


# MODELOS
Los modelos estan construidos sobre un backbone preentrenado que podía ser:
- Xception
- IncResNet
- EffNet 3

Como estos modelos tienen un input de (pix, pix, 3) se añade una conv2D inicial para transformar (512,512,1) en
tres canales.
Posteriormente se añade el backbone
Se hace Global max pooling sobre el output del backbone
En último lugar se añaden tres capas densas, la última con softmax hasta conseguir un output de (3,)

## Hiperparámetros
Los hiperparámetros que se han mantenido fijos han sido
- batch size de 8
- pixeles de 512
- train - test proportion de 0.8-0.2
- Optimizador Adam
- loss categorical crossentropy

Los hiperparámetros que se han ido variando han sido
- Backbone
- learning rate
- proporción del backbone desbloqueada para ajustar pesos
- aplicación de máscara sobre el input o no

## Modelos utilizados
Para entrenar con máscara se ha necesitado utilizar el modelo ***unet_final_renacimiento_validation_6.h5***.


# ENTRENAMIENTOS
## Entrenamientos independientes

Se ha entrenado cada una de las combinaciones de hiperparámetros una vez (90 en total) (***execute.py***).
Los entrenamientos se han guardado en: 
- /home/mr1142/Documents/Data/models/neumonia/training_data/train_max.csv

Se ha testeado cada uno de estos entrenamientos sobre val_subset y sobre val_resto (***validation_subset_exe.py***)

Los test se han guardado en:
- /home/mr1142/Documents/Data/models/neumonia/validation_results/prediction_validation_metrics.csv
- /home/mr1142/Documents/Data/models/neumonia/validation_results/image_class_evaluation.csv

## Hyperparameter tunning

Además se ha aplicado la herramienta mango, utilizando ht_train_subset y ht_val_subset para train y validación
con todos los hiperparámetros variables comentados (***ht_exe.py***, ***otras_funciones/train_funct_acil.py***).
Los entrenamientos no se han guardado. Los resultados están guardados en:
- /home/mr1142/Documents/Data/models/neumonia/ht/results.json

## Modelos definitivos

Los modelos definitivos se han entrenado sobre train (***execute.py***). El set de validación es el llamado X_test, ya considerado
en el dataframe original del ACIL.

Los entrenamientos se han guardado en:
- /home/mr1142/Documents/Data/models/neumonia/training_data/train_max_completos.csv

La validación se ha realizado sobre su propio val split y se ha guardado en:
- /home/mr1142/Documents/Data/models/neumonia/validation_results/prediction_validation_metrics_completos.csv
- /home/mr1142/Documents/Data/models/neumonia/validation_results/image_class_evaluation_completos.csv

Para el test definitivo el modelo se aplicará ***validation_exe.py***


# EVALUACIÓN

Para la evaluación se han utilizado dos métodos:
- model.evaluate
- model.predict

Con model.evaluate se han guardado las métricas que salen automáticamente (***funciones_evaluacion/prediction.py***).
Con model.predict se han utilizado métricas customizadas (***funciones_evaluacion/evaluation.py***) que están en ***funciones_evaluacion/metrics_and_plots.py***

***Entrenamientos independientes***: Se han evaluado sobre *val_subset* y *val_rest*, estos entrenamientos se han guardado en:

***Hyperparameter tunning***: Se han evaluado sobre *ht_val_subset*.

***Modelos definitivos***: Se han validado sobre su propio val split, los resultados están guardados en:
- /home/mr1142/Documents/Data/models/neumonia/validation_results/prediction_validation_metrics_completos.csv
- /home/mr1142/Documents/Data/models/neumonia/validation_results/image_class_evaluation_completos.csv