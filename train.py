from config import *
import random
random.seed(seed)
import os
import sys
import numpy as np
import tensorflow
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import utils
import tempfile
import pickle



# Arguments
fold = sys.argv[1]

print("Run " + sys.argv[0] + " K" + fold)

# Create the base pre-trained model
base_model = ResNet152V2(input_shape=(patch_height, patch_width, 3), weights=weights, include_top=False)

x = base_model.output
x = Flatten()(x)
outputs = Dense(3, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Regularization L2
"""
def add_regularization(model, regularizer):

    if not isinstance(regularizer, tensorflow.keras.regularizers.Regularizer):
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # Save the weights
    model_json = model.to_json()
    tmp_weights_path = 'init_weights_K' + str(fold) + '.h5'
    model.save_weights(tmp_weights_path)

    # Load the weights
    model = tensorflow.keras.models.model_from_json(model_json)
    model.load_weights(tmp_weights_path, by_name=True)

    return model

model = add_regularization(model, tensorflow.keras.regularizers.L2(l2=l2_regularization))
"""

# Fine-tuning
model.trainable = True

if transfer_learning:
    k = nbr_layers_to_retrain
    layers = base_model.layers[:-k] if k != 0 else base_model.layers
    for layer in layers:
        layer.trainable = False

print(model.summary())

# Compile model
opt = RMSprop(lr=learning_rate)
model.compile(loss=loss, optimizer=opt, metrics=metrics)

# Initiate the train, validation and test generators with data Augumentation
train_datagen = ImageDataGenerator(rescale = rescale,
                                   horizontal_flip = horizontal_flip, 
                                   vertical_flip = vertical_flip,
                                   rotation_range = rotation_range,
                                   shear_range = shear_range,
                                   zoom_range = zoom_range)
generator = train_datagen.flow_from_directory(directory = (dataset_path + "patches_organized_per_split/fold_" + fold + "/Train/"),
                                              target_size = (patch_height, patch_width),
                                              batch_size = batch_size,
                                              shuffle = True,
                                              seed = seed)

val_datagen = ImageDataGenerator(rescale = rescale)
val_generator = val_datagen.flow_from_directory(directory = (dataset_path + "patches_organized_per_split/fold_" + fold + "/Val/"),
                                              target_size = (patch_height, patch_width),
                                              batch_size = batch_size,
                                              shuffle = True,
                                              seed = seed)

test_datagen = ImageDataGenerator(rescale = rescale)
test_generator = test_datagen.flow_from_directory(directory = (dataset_path + "patches_organized_per_split/fold_" + fold + "/Test/"),
                                              target_size = (patch_height, patch_width),
                                              batch_size = 1,
                                              class_mode = None,
                                              shuffle = False,
                                              seed = seed)

# Train the model
#earlystopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=earlystopping_patience, verbose=2, mode='max')
savebestmodel = ModelCheckpoint(("results/model_K" + fold + ".h5"), save_best_only=True, monitor='val_loss', verbose=2, mode='min')
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=reducelr_factor, patience=reducelr_patience, min_lr=min_learning_rate)
callbacks = [savebestmodel,reducelr] #,earlystopping]

hist = model.fit_generator(generator,
                           validation_data=val_generator,
                           steps_per_epoch=generator.n//generator.batch_size,
                           validation_steps=val_generator.n//val_generator.batch_size,
                           epochs=num_epochs,
                           verbose=1,
                           callbacks=callbacks)

# Save accuracy / loss during training to pickle file
pickle.dump(hist.history, open(("results/history_K" + fold + ".pkl"), 'wb'))

print('history saved')



# Test
model = load_model(("results/model_K" + fold + ".h5"))

y_pred = model.predict_generator(test_generator)
y_classes = y_pred.argmax(axis=-1)

# Save class predictions for each patch
pickle.dump(y_classes, open("results/predictions_K" + fold + ".pkl", 'wb'))

print('Predicted class for each patch saved')
