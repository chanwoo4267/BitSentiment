import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub  # download preprocess and BERT-model from hub
import tensorflow_text as text
from official.nlp import optimization  # adamW

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

import pandas as pd
import numpy as np
import re

from keras import backend as K


def _balanced_recall(y_true, y_pred):
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


def _balanced_precision(y_true, y_pred):
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


def _balanced_f1_score(y_true, y_pred):
    """This function calculates the F1 score metric"""
    precision = _balanced_precision(y_true, y_pred)
    recall = _balanced_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


_epochs = 20

# optimizer adamw
_steps_per_epoch = 965
# steps_per_epoch = tf.data.experimental.cardinality(my_train_ds).numpy()
_num_train_steps = _steps_per_epoch * _epochs
_num_warmup_steps = int(0.1 * _num_train_steps)
_init_lr = 3e-5
_optimizer = optimization.create_optimizer(init_lr=_init_lr,
                                           num_train_steps=_num_train_steps,
                                           num_warmup_steps=_num_warmup_steps,
                                           optimizer_type='adamw')

_classifier_model = tf.keras.models.load_model("natural_v4", custom_objects={'balanced_recall': _balanced_recall,
                                                                            'balanced_precision': _balanced_precision,
                                                                            'balanced_f1_score': _balanced_f1_score,
                                                                            'AdamWeightDecay': _optimizer})

_classifier_model.summary()


def get_emotion(content: str) -> int:
    result = _classifier_model(tf.constant([content]))
    for re in result:
        return [0, 1, -1][np.argmax(re)]


# example_sentences = ["so bad",
#                      "very good",
#                      "don't buy it please",
#                      "you must buy it."]
#
# in_sentences = example_sentences
#
# result = _classifier_model(tf.constant(in_sentences))
# my_labels = ["neutral ", "positive", "negative"]
#
# for re, sentence in zip(result, in_sentences):
#     print(f'{re} {my_labels[np.argmax(re)]} - {sentence}')
