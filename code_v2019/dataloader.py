# -*- coding: utf-8 -*-
"""
The dataloader to generate training samples from TFRecords.

Junyang Gou
2022.06.02
"""
import numpy as np
from functools import partial

import tensorflow as tf

def decode_TFRecord(record_bytes):
    """
    Here the returns will be tuple.
    """
    tfrecord_format = (
        {"GRACE_ori": tf.io.VarLenFeature(dtype=tf.float32),
         "WGHM_ori": tf.io.VarLenFeature(dtype=tf.float32),
         "GRACE": tf.io.VarLenFeature(dtype=tf.float32),
         "WGHM": tf.io.VarLenFeature(dtype=tf.float32),
         "P": tf.io.VarLenFeature(dtype=tf.float32),
         "ET": tf.io.VarLenFeature(dtype=tf.float32),
         "Qs": tf.io.VarLenFeature(dtype=tf.float32),
         "Qsb": tf.io.VarLenFeature(dtype=tf.float32),
         "Qsm": tf.io.VarLenFeature(dtype=tf.float32),
         "Lat": tf.io.VarLenFeature(dtype=tf.float32),
         "Lon": tf.io.VarLenFeature(dtype=tf.float32),
         "Shape": tf.io.FixedLenFeature(shape=(2,), dtype=tf.int64)}
        )
    data = tf.io.parse_single_example(record_bytes, tfrecord_format)
    data["GRACE_ori"] = tf.sparse.to_dense(data["GRACE_ori"])
    data["WGHM_ori"] = tf.sparse.to_dense(data["WGHM_ori"])
    data["GRACE"] = tf.sparse.to_dense(data["GRACE"])
    data["WGHM"] = tf.sparse.to_dense(data["WGHM"])
    data["P"] = tf.sparse.to_dense(data["P"])
    data["ET"] = tf.sparse.to_dense(data["ET"])
    data["Qs"] = tf.sparse.to_dense(data["Qs"])
    data["Qsb"] = tf.sparse.to_dense(data["Qsb"])
    data["Qsm"] = tf.sparse.to_dense(data["Qsm"])
    data["Lat"] = tf.sparse.to_dense(data["Lat"])
    data["Lon"] = tf.sparse.to_dense(data["Lon"])
    data["Shape"] = data["Shape"]
    return data


def getSample(record_bytes, NumFeatures=9):
    data = decode_TFRecord(record_bytes)
    shape = data["Shape"]
    if NumFeatures == 9:
        X = tf.concat(values=[tf.expand_dims(tf.reshape(data["GRACE"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["WGHM"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["P"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["ET"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Qs"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Qsb"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Qsm"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Lat"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Lon"], shape), axis=2)],
                      axis=2)
    elif NumFeatures == 2: # Only GRACE and WGHM
        X = tf.concat(values=[tf.expand_dims(tf.reshape(data["GRACE"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["WGHM"], shape), axis=2)],
                      axis=2)
    elif NumFeatures == 4: # Only GRACE and WGHM + Geocoordinates
        X = tf.concat(values=[tf.expand_dims(tf.reshape(data["GRACE"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["WGHM"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Lat"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Lon"], shape), axis=2)],
                      axis=2)
    elif NumFeatures == 7: # Without geocoordinates
        X = tf.concat(values=[tf.expand_dims(tf.reshape(data["GRACE"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["WGHM"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["P"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["ET"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Qs"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Qsb"], shape), axis=2),
                              tf.expand_dims(tf.reshape(data["Qsm"], shape), axis=2)],
                      axis=2)

    Y = tf.concat(values=[tf.expand_dims(tf.reshape(data["GRACE_ori"], shape), axis=2),
                          tf.expand_dims(tf.reshape(data["WGHM_ori"], shape), axis=2)],
                  axis=2)
    return X, Y


def load_dataset(filenames, NumFeatures=9, Order=False):
    if Order:
        # automatically interleaves reads from multiple files
        dataset = tf.data.TFRecordDataset(filenames)

        # returns a dataset of (X, Y) pairs
        dataset = dataset.map(getSample, NumFeatures=NumFeatures)
    else:
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed

        # automatically interleaves reads from multiple files
        dataset = tf.data.TFRecordDataset(filenames)

        # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.with_options(ignore_order)

        # returns a dataset of (X, Y) pairs
        dataset = dataset.map(partial(getSample, NumFeatures=NumFeatures), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def get_dataset(filenames, NumFeatures=9, BatchSize=512, Order=False):
    dataset = load_dataset(filenames, NumFeatures=NumFeatures, Order=False)
    dataset = dataset.shuffle(1996)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(BatchSize)
    return dataset


if __name__ == "__main__":
    print("The script contains the functions about TFRecord")
