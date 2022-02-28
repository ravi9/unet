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

from distutils.sysconfig import get_python_lib
from tensorflow.contrib.session_bundle import exporter
import keras

import h5py

import tensorflow as tf  # conda install -c anaconda tensorflow
import settings   # Use the custom settings.py file for default parameters

from model import unet
from data import load_data
import keras as K
import numpy as np

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
args.inference_filename_new='unet_decathlon_4_8814_128x128_randomcrop-any-input.h5'
args.keras_api=True

args.num_inter_threads=1
args.num_threads=56
args.output_path='/home/bduser/tony/unet_tiling/'
args.print_model=True
args.use_augmentation=True
args.use_dropout=True
args.use_upsampling=True
args.weight_dice_loss=0.9
args.loss = 'mse'
args.data_format = "channels_last"
args.metrics = ["accuracy"]


sess = keras.backend.get_session()

data_fn = os.path.join(args.data_path, args.data_filename)
model_fn = os.path.join(args.output_path, args.inference_filename)


# Load data
print("Loading Data... ")
df = h5py.File(data_fn, "r")
imgs_train = df["imgs_train"]
msks_train = df["msks_train"]
print("Data loaded successfully from: " + data_fn)

"""
Step 2: Define the model
"""

unet_model_obj = unet()
unet_model = unet_model_obj.create_model(imgs_train.shape, msks_train.shape)
saved_model_fname = os.path.join(args.output_path, args.inference_filename)
saved_model_fname_new = os.path.join(args.output_path, args.inference_filename_new)

unet_model.load_weights(saved_model_fname)
print("Loaded a saved model: " + saved_model_fname)
unet_model.summary()

print("Saving redefined model: " + saved_model_fname_new)
unet_model.save(saved_model_fname_new)


