import numpy as np
import pandas as pd
import time
import tensorflow as tf

import pathlib
import shutil
import tempfile
import pathlib
import shutil
import tempfile
import os
import argparse

# Suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

print("Version: ", tf.__version__)
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
print('Number of GPUs :',len(tf.config.list_physical_devices('GPU')))

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# Parse input arguments

parser = argparse.ArgumentParser(description='Transfer Learning Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--log-dir', default=logdir,
                    help='tensorboard log directory')

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')

parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')

parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')

parser.add_argument('--patience', type=float, default=2,
                    help='Number of epochs that meet target before stopping')

parser.add_argument('--use-checkpointing', default=False, action='store_true')

args = parser.parse_args()

# Steps

if os.path.exists('dataset.pkl'):
    df = pd.read_pickle('dataset.pkl')
else:
    df = pd.read_csv('https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip', 
             compression='zip', low_memory=False)
    df.to_pickle('dataset.pkl')

train_df, remaining = train_test_split(df, random_state=42, train_size=0.5, stratify=df.target.values)
valid_df, _  = train_test_split(remaining, random_state=42, train_size=0.01, stratify=remaining.target.values)
print("The shape of training {} and validation {} datasets.".format(train_df.shape, valid_df.shape))
print("##-------------------------##")

buffer_size = train_df.size
train_dataset = tf.data.Dataset.from_tensor_slices((train_df.question_text.values, train_df.target.values)).shuffle(buffer_size).batch(args.batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_df.question_text.values, valid_df.target.values)).batch(args.batch_size)

module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
embeding_size = 128
name_of_model = 'nnlm-en-dim128'

print("Batch size :", args.batch_size)

callbacks = []
if args.use_checkpointing:
    # callbacks.append(tfdocs.modeling.EpochDots()),
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')),
    callbacks.append(tf.keras.callbacks.TensorBoard(logdir/name))

def train_and_evaluate_model(module_url, embed_size, name, trainable=False):
    hub_layer = hub.KerasLayer(module_url, input_shape=[], output_shape=[embed_size], dtype = tf.string, trainable=trainable)
    model = tf.keras.models.Sequential([
                                      hub_layer,
                                      tf.keras.layers.Dense(256, activation='relu'),
                                      tf.keras.layers.Dense(64, activation='relu'),
                                      tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.base_lr),
                loss = tf.losses.BinaryCrossentropy(),
                metrics = [tf.metrics.BinaryAccuracy(name='accuracy')])
    
    history = model.fit(train_df['question_text'], train_df['target'],
                      epochs = args.epochs,
                      batch_size = args.batch_size,
                      validation_data=(valid_df['question_text'], valid_df['target']),
                      callbacks=callbacks,
                      verbose = 1
                      )
    return history

#with tf.device("GPU:0"):
#    start = time.time()
#    print("\n##-------------------------##")
#    print("Training (pada dataframes) starts ...")
#    history = train_and_evaluate_model(module_url, embed_size=embeding_size, name=name_of_model)
#    endt = time.time()-start
#    print("Elapsed Time: {} ms".format(1000*endt))
#    print("##-------------------------##")

def train_and_evaluate_model_ds(module_url, embed_size, name, trainable=False):
    hub_layer = hub.KerasLayer(module_url, input_shape=[], output_shape=[embed_size], dtype = tf.string, trainable=trainable)
    model = tf.keras.models.Sequential([
                                      hub_layer,
                                      tf.keras.layers.Dense(256, activation='relu'),
                                      tf.keras.layers.Dense(64, activation='relu'),
                                      tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.base_lr),
                loss = tf.losses.BinaryCrossentropy(),
                metrics = [tf.metrics.BinaryAccuracy(name='accuracy')])
    
    history = model.fit(train_dataset,
                      epochs = args.epochs,
                      validation_data=valid_dataset,
                      callbacks=callbacks,
                      verbose = 1
                      )
    return history

with tf.device("GPU:0"):
    start = time.time()
    print("\n##-------------------------##")
    print("Training starts ...")
    history = train_and_evaluate_model_ds(module_url, embed_size=embeding_size, name=name_of_model, trainable=True)
    endt = time.time()-start
    print("Elapsed Time: {} ms".format(1000*endt))
    print("##-------------------------##")
