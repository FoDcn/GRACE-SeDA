# -*- coding: utf-8 -*-
"""
This script is for training. We use the WGHM version until 2019 and also include
GRACE-FO data.

Junyang Gou
2023.06.22
"""
import numpy as np
import random
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from model import build_model, custom_loss, MAE_GRACE, MAE_model, Pearson_model
from dataloader import get_dataset


if __name__ == "__main__":
    # TODO: Please modify the parameters when necessary
    # Common settings----------------------------------------------------------
    BatchSize = 512
    LR = 0.001
    NumEpoch = 150
    Region = 'Global'
    NumFeatures = 9
    SEED = 1996
    model_name = Region + '_2019_' + str(NumFeatures) + "f_" + str(SEED)
    print(SEED)

    outpath = "[Give the path where you would like to save your outputs]"

    data_path = "[Give the path to your .tfrecords]"
    #--------------------------------------------------------------------------

    # Check the enviroment
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Fix randomness
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Get current time --> Use for name of folder
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Set path. Creat folders
    creat_model_folder = (outpath / (model_name + '_' + Region + '_' +  current_time)).mkdir(parents=True, exist_ok=True)
    checkpoint_path = outpath / (model_name + '_' + Region + '_' + current_time) /  ("weights.{epoch:03d}-{val_loss:.2f}.hdf5")
    model_path = outpath / (model_name + '_' + Region + '_' + current_time) /  (model_name + '.h5')


    FILENAMES_unshuffle = tf.io.gfile.glob(data_path + "*.tfrecords")
    FILENAMES = FILENAMES_unshuffle
    random.shuffle(FILENAMES)
    # Generate datasets
    train_dataset = get_dataset(FILENAMES, NumFeatures) # All training

    # Build the model
    model = build_model(C=NumFeatures)

    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    model.compile(optimizer=opt,
                  loss=custom_loss,
                  metrics=[MAE_GRACE, MAE_model, Pearson_model])

    model.summary()
    model.save(model_path)

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss',
                                   verbose=2, save_best_only=False, mode='min')
    callbacks_list = [checkpoint]

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=NumEpoch,
        callbacks=callbacks_list
    )

    history_list = []
    history_list.append(history)
