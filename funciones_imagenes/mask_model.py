import os
import tensorflow.keras as keras
import funciones_imagenes.losses as ex


model = os.path.join('./modelos', 'unet_final_renacimiento_validation_6.h5')
model = keras.models.load_model(model, 
                                    custom_objects={"loss_mask": keras.losses.BinaryCrossentropy, 
                                                    "dice_coef_loss": ex.dice_coef_loss,
                                                    "dice_coef": ex.dice_coef})

print('\n\n MASK MODEL LOADED \n\n')