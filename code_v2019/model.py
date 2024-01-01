"""
This script includes the necessary components for building the model using
Tensorflow.

Author: Junyang Gou
2022.08.31
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers


def ResBlock(x, num_filters, kernelsize):
    fx = layers.Conv2D(num_filters, kernelsize, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(num_filters, kernelsize, padding='same')(fx)
    out = layers.Add()([x, fx])
    out = layers.ReLU()(out)
    out = layers.BatchNormalization()(out)
    return out


def custom_loss(y_label, y_pred):
    '''
    Inputs:
        y_label [#Batch, W, H, 2] ([GRACE, WGHM])
        y_pred [#Batch, W, H, 1] (GRACE_SR)

    '''
    #-------------------------------------------------------------------------#
    '''
    The GRACE part --> Compute AE

    For GRACE, we should firstly to compute the averge over pathes, then compute
    MAE so that the feiner structure will not be limited
    '''
    loss_MAE_GRACE = tf.math.abs(
        tf.math.reduce_mean(y_label[:, :, :, 0:1], axis=(1, 2, 3)) -
        tf.math.reduce_mean(y_pred, axis=(1, 2, 3)))


    #-------------------------------------------------------------------------#
    '''
    The WGHM part --> Compute 2D Pearson and pixel-wise MAE

    For Model, we should compute the pixel-wise MAE to enhance the structure
    '''
    loss_MAE_model = tf.math.reduce_mean(tf.math.abs(y_label[:, :, :, 1:2] - y_pred),
                                         axis=(1, 2, 3))

    '''
    Similarity is the key to enhance the structure
    '''
    loss_Pearson = Pearson_2D(y_label, y_pred)

    '''
    Combine the three parts
    '''
    custom_loss = loss_MAE_GRACE + tf.math.multiply(loss_MAE_model, (1 - loss_Pearson))

    return custom_loss


def MAE_GRACE(y_label, y_pred):
    '''
    For GRACE MAE, we compute the basin-wise MAE
    '''
    # The GRACE part --> Compute MAE
    MAE_GRACE = tf.math.abs(
        tf.math.reduce_mean(y_label[:, :, :, 0:1], axis=(1, 2, 3)) -
        tf.math.reduce_mean(y_pred, axis=(1, 2, 3)))
    return MAE_GRACE


def MAE_model(y_label, y_pred):
    '''
    For model MAE, we compute the pixel-wise MAE
    '''
    MAE_model = tf.math.reduce_mean(tf.math.abs(y_label[:, :, :, 1:2] - y_pred), axis=(1, 2, 3))
    return MAE_model


def Pearson_model(y_label, y_pred):
    Pearson_model = Pearson_2D(y_label, y_pred)
    return Pearson_model


def Pearson_2D(y_label, y_pred):
    '''
    Compute the 2D Pearson correlation coefficient for two images.

    Inputs:
        y_label [#Batch, W, H, 2] (GRACE, WGHM)
        y_pred [#Batch, W, H, 1] (GRACE_SR)

    Output:
        rho [#Batch]
    '''
    A = y_label[:, :, :, 1:2]
    B = y_pred

    num_Batch = tf.shape(A)[0]
    A_mean = tf.reshape(tf.math.reduce_mean(A, axis=[1, 2, 3]), [num_Batch, 1, 1, 1])
    B_mean = tf.reshape(tf.math.reduce_mean(B, axis=[1, 2, 3]), [num_Batch, 1, 1, 1])

    A_red = A - A_mean
    B_red = B - B_mean

    AB = tf.math.multiply(A_red, B_red)
    numerator = tf.math.reduce_sum(AB, axis=[1, 2, 3])

    Asquare = tf.math.reduce_sum(tf.math.multiply(A_red, A_red), axis=[1, 2, 3])
    Bsquare = tf.math.reduce_sum(tf.math.multiply(B_red, B_red), axis=[1, 2, 3])

    denominator = tf.math.sqrt(tf.math.multiply(Asquare, Bsquare))

    rho = tf.math.divide(numerator, denominator + 1e-10)

    return rho


def build_model(W=32, H=32, C=9):
    """
    Build the model for downscaling GRACE measurements.
    
    Inputs: W -- Width of the patch
            H -- Height of the patch
            C -- Number of channels
    """
    inputs = keras.Input(shape=(W, H, C))
    #--------------------------Encoder part----------------------------------------
    Conv_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=2,
                      activation='relu', padding='same')(inputs)
    Res_11 = ResBlock(Conv_1, 16, (3, 3))
    Res_12 = ResBlock(Res_11, 16, (3, 3))
    Res_13 = ResBlock(Res_12, 16, (3, 3))
    
    Conv_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=2,
                      activation='relu', padding='same')(Res_13)
    Res_21 = ResBlock(Conv_2, 32, (3, 3))
    Res_22 = ResBlock(Res_21, 32, (3, 3))
    Res_23 = ResBlock(Res_22, 32, (3, 3))
    
    Conv_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=2,
                      activation='relu', padding='same')(Res_23)
    Res_31 = ResBlock(Conv_3, 64, (3, 3))
    Res_32 = ResBlock(Res_31, 64, (3, 3))
    Res_33 = ResBlock(Res_32, 64, (3, 3))
    
    # #--------------------------Decoder part----------------------------------------
    Up_1 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(Res_33)
    Conv_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                      activation='relu', padding='same')(Up_1)
    Res_41 = ResBlock(Conv_4, 64, (3, 3))
    Res_42 = ResBlock(Res_41, 64, (3, 3))
    Res_43 = ResBlock(Res_42, 64, (3, 3))
    
    Up_2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(Res_43)
    Conv_5 = Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                      activation='relu', padding='same')(Up_2)
    Res_51 = ResBlock(Conv_5, 32, (3, 3))
    Res_52 = ResBlock(Res_51, 32, (3, 3))
    Res_53 = ResBlock(Res_52, 32, (3, 3))
    
    Up_3 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(Res_53)
    Conv_6 = Conv2D(filters=16, kernel_size=(3, 3), strides=1,
                      activation='relu', padding='same')(Up_3)
    Res_61 = ResBlock(Conv_6, 16, (3, 3))
    Res_62 = ResBlock(Res_61, 16, (3, 3))
    Res_63 = ResBlock(Res_62, 16, (3, 3))
    
    outputs = Conv2D(filters=1, activation=None, kernel_size=(3, 3), padding='same')(Res_63)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet")
    return model