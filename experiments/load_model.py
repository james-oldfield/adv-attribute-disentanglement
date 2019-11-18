from collections import namedtuple

import tensorflow as tf

import sys
sys.path.append('./')
from Model import Model


def load_model(n_attributes, attribute_names, input_files, target_files, batch_size):
    argsStruct = namedtuple('args', 'img_size n_epochs n_decay learning_rate beta1 from_checkpoint data_dir n_attributes attribute_names sample_dir')
    args = argsStruct(128, 0, 0, 0, 0, '', '', n_attributes, attribute_names, '')

    model = Model()
    model.img_size = 128
    model.build_encoders(args)

    with tf.Session(graph=model.graph):
        model.input_set = tf.data.Dataset.from_tensor_slices((tf.constant(input_files))) \
            .map(model._parse_function) \
            .shuffle(100) \
            .apply(tf.contrib.data.batch_and_drop_remainder(batch_size)) \
            .repeat() \
            .make_one_shot_iterator()
        model.X_i = model.input_set.get_next()

        model.target_set = tf.data.Dataset.from_tensor_slices((tf.constant(target_files))) \
            .map(model._parse_function) \
            .shuffle(100) \
            .apply(tf.contrib.data.batch_and_drop_remainder(batch_size)) \
            .repeat() \
            .make_one_shot_iterator()
        model.X_t = model.target_set.get_next()

    return model