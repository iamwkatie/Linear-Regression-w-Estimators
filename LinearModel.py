#Setup
#Import TensorFlow, feature column support, and supporting modules:

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.feature_column as fc 

import os
import sys

import matplotlib.pyplot as plt
from IPython.display import clear_output

tf.enable_eager_execution()



## Command line usage
#First add the path to tensorflow/models to PYTHONPATH.

#export PYTHONPATH=${PYTHONPATH}:"$(pwd)/models"
#running from python you need to set the `os.environ` or the subprocess will not see the directory.

if "PYTHONPATH" in os.environ:
  os.environ['PYTHONPATH'] += os.pathsep +  models_path
else:
  os.environ['PYTHONPATH'] = models_path
  

  
#Read the Census data
#This project uses the US Census Dataset from 1994 and 1995.
#Since this is a binary classification proble
m, we'll construct a label column named "label" whose value is 1 if the income is over 50K, and 0 otherwise.
#Look at the data to see which columns we can use to predict the target label:

!ls  /tmp/census_data/

train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"

import pandas

train_df = pandas.read_csv(train_file, header = None, names = census_dataset._CSV_COLUMNS)
test_df = pandas.read_csv(test_file, header = None, names = census_dataset._CSV_COLUMNS)

train_df.head()



## Converting Data into Tensors
#Make a `tf.data.Dataset` by slicing the pandas.DataFrame:

def input_function(df, label_key, num_epochs, shuffle, batch_size):
  label = df[label_key]
  ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

  if shuffle:
    ds = ds.shuffle(10000)

  ds = ds.batch(batch_size).repeat(num_epochs)

  return ds


#Inspect the resulting dataset:
ds = input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys())[:5])
  print()
  print('A batch of Ages  :', feature_batch['age'])
  print()
  print('A batch of Labels:', label_batch )
  
 
