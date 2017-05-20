from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "train.csv"
IRIS_TEST = "test.csv"


# Define the training inputs
def get_train_inputs(training_set):
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y

# Define the test inputs
def get_test_inputs(test_set):
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y

def main():
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.int)

  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.int)

  init = tf.global_variables_initializer()

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=20)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[20, 64,128,64,10],
                                              n_classes=10,
                                              model_dir="../model/sha1_model")

  # Fit model
  classifier.fit(x=training_set.data, y=training_set.target, batch_size=100, steps=1000000)
  #classifier.fit(input_fn=lambda: get_train_inputs(training_set),  steps=2000)

  # Evaluate accuracy
  #accuracy_score = classifier.evaluate(input_fn=lambda: get_test_inputs(test_set),steps=1)["accuracy"]
  #accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target, steps=1)["accuracy"]
  accuracy_score = classifier.evaluate(x=training_set.data, y=training_set.target, steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

if __name__ == "__main__":
    main()
