import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub # download preprocess and BERT-model from hub
import tensorflow_text as text
from official.nlp import optimization # adamW

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

import pandas as pd
import numpy as np
import re

from keras import backend as K

def balanced_recall(y_true, y_pred):
    """This function calculates the balanced recall metric
    recall = TP / (TP + FN)
    """
    recall_by_class = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(y_pred.shape[1]):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        recall_by_class = recall_by_class + recall
    return recall_by_class / y_pred.shape[1]

def balanced_precision(y_true, y_pred):
    """This function calculates the balanced precision metric
    precision = TP / (TP + FP)
    """
    precision_by_class = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(y_pred.shape[1]):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        precision_by_class = precision_by_class + precision
    # return average balanced metric for each class
    return precision_by_class / y_pred.shape[1]

def balanced_f1_score(y_true, y_pred):
    """This function calculates the F1 score metric"""
    precision = balanced_precision(y_true, y_pred)
    recall = balanced_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

epochs = 20

# optimizer adamw
steps_per_epoch = 965
#steps_per_epoch = tf.data.experimental.cardinality(my_train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)
init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model = tf.keras.models.load_model("natural_v4", custom_objects={'balanced_recall':balanced_recall, 'balanced_precision':balanced_precision, 'balanced_f1_score':balanced_f1_score, 'AdamWeightDecay': optimizer})

classifier_model.summary()

example_sentences = ["so bad",
                     "very good",
                     "don't buy it please",
                      "you must buy it."]

in_sentences = example_sentences
result = classifier_model(tf.constant(in_sentences))
my_labels = ["neutral ","positive","negative"]

print(my_labels)
for re, sentence in zip(result, in_sentences):
  print(f'{re} {my_labels[np.argmax(re)]} - {sentence}')