# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:04:26 2023

@author: Aitor
"""

"CONFIGURACIÓN INICIAL"
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
#import tensorflow_text as text
#from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')


"""Descarga el conjunto de datos de IMDB"""
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

