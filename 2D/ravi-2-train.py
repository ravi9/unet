#!/usr/bin/env python
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

"""
This module loads the data from data.py, creates a TensorFlow/Keras model
from model.py, trains the model on the data, and then saves the
best model.
"""

import datetime
import os

import tensorflow as tf  # conda install -c anaconda tensorflow
import settings   # Use the custom settings.py file for default parameters

from model import unet
from data import load_data

import numpy as np

from argparser import args

"""
For best CPU speed set the number of intra and inter threads
to take advantage of multi-core systems.
See https://github.com/intel/mkl-dnn
"""
CONFIG = tf.ConfigProto(intra_op_parallelism_threads=args.num_threads,
                        inter_op_parallelism_threads=args.num_inter_threads)

SESS = tf.Session(config=CONFIG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K

print("TensorFlow version: {}".format(tf.__version__))
print("Intel MKL-DNN is enabled = {}".format(tf.pywrap_tensorflow.IsMklEnabled()))

print("Keras API version: {}".format(K.__version__))

K.backend.set_session(SESS)


def train_and_predict(data_path, data_filename, batch_size, n_epoch):
    """
    Create a model, load the data, and train it.
    """

    """
    Step 1: Load the data
    """
    hdf5_filename = os.path.join(data_path, data_filename)
    print("-" * 30)
    print("Loading the data from HDF5 file ...")
    print("-" * 30)

    imgs_train, msks_train, imgs_validation, msks_validation, imgs_testing, msks_testing = \
        load_data(hdf5_filename, args.batch_size,
                  [args.crop_dim, args.crop_dim])

    np.random.seed(816)

    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 2: Define the model
    """

    unet_model = unet()
    model = unet_model.create_model(imgs_train.shape, msks_train.shape)

    model_filename, model_callbacks = unet_model.get_callbacks()

    # If there is a current saved file, then load weights and start from
    # there.
    saved_model = os.path.join(args.output_path, args.inference_filename)
    saved_model_new = os.path.join(args.output_path, args.inference_filename_new)
    
    if os.path.isfile(saved_model):
        model.load_weights(saved_model)
        print("-" * 30)
        print("Loaded a saved model: " + saved_model)
        print("-" * 30)
        model.summary()
        
        print("-" * 30)
        print("Saving redefined model: " + saved_model_new)
        print("-" * 30)
        model.save(saved_model_new)
        
    """
    Step 3: Train the model on the data
    """
#     print("-" * 30)
#     print("Fitting model with training data ...")
#     print("-" * 30)

#     model.fit(imgs_train, msks_train,
#               batch_size=batch_size,
#               epochs=n_epoch,
#               validation_data=(imgs_validation, msks_validation),
#               verbose=1, shuffle="batch",
#               callbacks=model_callbacks)

    """
    Step 4: Evaluate the best model
    """
    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    unet_model.evaluate_model(saved_model_new, imgs_testing, msks_testing)


if __name__ == "__main__":

    # os.system("lscpu")
    
    START_TIME = datetime.datetime.now()
    print("Started script on {}".format(START_TIME))

    #print("args = {}".format(args))
    import argparse

    args = argparse.Namespace()
    args.batch_size=128
    args.blocktime=1
    args.channels_first=False
    args.crop_dim=-1
    args.data_filename='decathlon_brats.h5'
    args.data_path='/home/bduser/tony/data/decathlon//240x240'
    args.epochs=0
    args.featuremaps=32
    args.fms=32
    args.concat_axis=-1
    args.learningrate=0.0001
    args.optimizer = K.optimizers.Adam(lr=args.learningrate)
    args.inference_filename='unet_decathlon_4_8814_128x128_randomcrop.hdf5'
    args.inference_filename_new='unet_decathlon_4_8814_128x128_randomcrop-fullimage-input.h5'
    #args.inference_filename='unet_decathlon_4_8814_128x128_randomcrop-fullimg.hdf5'
    args.keras_api=True

    args.num_inter_threads=1
    args.num_threads=56
    args.output_path='/home/bduser/tony/unet_tiling/'
    args.print_model=False
    args.use_augmentation=True
    args.use_dropout=True
    args.use_upsampling=True
    args.weight_dice_loss=0.9
    args.loss = 'mse'
    args.data_format = "channels_last"
    args.metrics = ["accuracy"]
    
    #os.system("uname -a")
    print("TensorFlow version: {}".format(tf.__version__))

    train_and_predict(args.data_path, args.data_filename,
                      args.batch_size, args.epochs)

    print(
        "Total time elapsed for program = {} seconds".format(
            datetime.datetime.now() -
            START_TIME))
    print("Stopped script on {}".format(datetime.datetime.now()))
