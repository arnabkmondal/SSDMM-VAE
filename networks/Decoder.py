import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam


def Conv(n_filters, filter_width, strides=2, activation="relu", name=None):
    return Conv2D(n_filters, filter_width,
                  strides=strides, padding="same", activation=activation, name=name)


def Deconv(n_filters, filter_width, strides=2, activation="relu", name=None):
    return Conv2DTranspose(n_filters, filter_width,
                           strides=strides, padding="same", activation=activation, name=name)


class Decoder(tf.keras.Model):
    def __init__(self, dataset, kernel_init, kernel_reg, kernel_constraint):
        super(Decoder, self).__init__()
        self.dataset = dataset
        self.dec_kernel_initializer = kernel_init
        self.dec_kernel_reg = kernel_reg
        self.dec_kernel_constraint = kernel_constraint
        if self.dataset == 'mnist' or self.dataset == 'fashion' or self.dataset == 'mad_base':
            self.dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu,
                                                kernel_constraint=self.dec_kernel_constraint,
                                                kernel_initializer=self.dec_kernel_initializer,
                                                kernel_regularizer=self.dec_kernel_reg)
            self.dense2 = tf.keras.layers.Dense(units=784, activation=tf.nn.sigmoid,
                                                kernel_constraint=self.dec_kernel_constraint,
                                                kernel_initializer=self.dec_kernel_initializer,
                                                kernel_regularizer=self.dec_kernel_reg)

            self.reshape = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        elif self.dataset == 'dsprite' or self.dataset == 'smallnorb':
            self.dense1 = Dense(256, activation='relu')
            self.dense2 = Dense(256, activation='relu')
            self.dense3 = Dense(512, activation='relu')
            self.reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 32))
            self.trans_conv1 = Deconv(32, 4)
            self.trans_conv2 = Deconv(32, 4)
            self.trans_conv3 = Deconv(32, 4)
            self.trans_conv4 = Deconv(1, 4, activation="sigmoid")
        elif self.dataset == 'shapes3d':
            self.dense1 = Dense(256, activation='relu')
            self.dense2 = Dense(256, activation='relu')
            self.dense3 = Dense(512, activation='relu')
            self.reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 32))
            self.trans_conv1 = Deconv(32, 4)
            self.trans_conv2 = Deconv(32, 4)
            self.trans_conv3 = Deconv(32, 4)
            self.trans_conv4 = Deconv(3, 4, activation="sigmoid")
        elif self.dataset == 'mpi3d_toy':
            self.dense1 = Dense(256, activation='relu')
            self.dense2 = Dense(256, activation='relu')
            self.dense3 = Dense(512, activation='relu')
            self.reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 32))
            self.trans_conv1 = Deconv(32, 4)
            self.trans_conv2 = Deconv(32, 4)
            self.trans_conv3 = Deconv(32, 4)
            self.trans_conv4 = Deconv(3, 4, activation="sigmoid")
        elif self.dataset == 'svhn':
            self.dense1 = tf.keras.layers.Dense(units=np.prod((2, 2, 128)), activation=tf.nn.relu,
                                                kernel_constraint=self.dec_kernel_constraint,
                                                kernel_initializer=self.dec_kernel_initializer,
                                                kernel_regularizer=self.dec_kernel_reg)
            self.reshape = tf.keras.layers.Reshape(target_shape=(2, 2, 128))
            self.trans_conv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                                               padding='same', activation=tf.nn.relu,
                                                               kernel_initializer=self.dec_kernel_initializer,
                                                               kernel_constraint=self.dec_kernel_constraint,
                                                               kernel_regularizer=self.dec_kernel_reg)
            self.trans_conv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                                               padding='same', activation=tf.nn.relu,
                                                               kernel_initializer=self.dec_kernel_initializer,
                                                               kernel_constraint=self.dec_kernel_constraint,
                                                               kernel_regularizer=self.dec_kernel_reg)
            self.trans_conv3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                               padding='same', activation=tf.nn.relu,
                                                               kernel_initializer=self.dec_kernel_initializer,
                                                               kernel_constraint=self.dec_kernel_constraint,
                                                               kernel_regularizer=self.dec_kernel_reg)
            self.trans_conv4 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2),
                                                               padding='same', activation=tf.nn.sigmoid,
                                                               kernel_initializer=self.dec_kernel_initializer,
                                                               kernel_constraint=self.dec_kernel_constraint,
                                                               kernel_regularizer=self.dec_kernel_reg)

        elif self.dataset == 'mnist_text' or self.dataset == 'dsprite_text':
            self.dense0 = tf.keras.layers.Dense(units=12, activation=tf.nn.relu,
                                                kernel_constraint=self.dec_kernel_constraint,
                                                kernel_initializer=self.dec_kernel_initializer,
                                                kernel_regularizer=self.dec_kernel_reg)
            self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu,
                                                kernel_constraint=self.dec_kernel_constraint,
                                                kernel_initializer=self.dec_kernel_initializer,
                                                kernel_regularizer=self.dec_kernel_reg)
            self.trans_conv1 = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=4, strides=1,
                                                               padding='valid', activation=tf.nn.relu,
                                                               kernel_initializer=self.dec_kernel_initializer,
                                                               kernel_constraint=self.dec_kernel_constraint,
                                                               kernel_regularizer=self.dec_kernel_reg)
            self.trans_conv2 = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=4, strides=1,
                                                               padding='same', activation=tf.nn.relu,
                                                               kernel_initializer=self.dec_kernel_initializer,
                                                               kernel_constraint=self.dec_kernel_constraint,
                                                               kernel_regularizer=self.dec_kernel_reg)
            self.conv3 = tf.keras.layers.Conv1D(filters=70, kernel_size=1, strides=2,
                                                padding='same', activation=tf.nn.softmax,
                                                kernel_initializer=self.dec_kernel_initializer,
                                                kernel_constraint=self.dec_kernel_constraint,
                                                kernel_regularizer=self.dec_kernel_reg)

        elif self.dataset == 'mpi3d_toy_text' or self.dataset == 'shapes3d_text' or self.dataset == 'smallnorb_text':
            self.dense0 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu,
                                                kernel_constraint=self.dec_kernel_constraint,
                                                kernel_initializer=self.dec_kernel_initializer,
                                                kernel_regularizer=self.dec_kernel_reg)
            self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu,
                                                kernel_constraint=self.dec_kernel_constraint,
                                                kernel_initializer=self.dec_kernel_initializer,
                                                kernel_regularizer=self.dec_kernel_reg)
            self.trans_conv1 = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=4, strides=1,
                                                               padding='valid', activation=tf.nn.relu,
                                                               kernel_initializer=self.dec_kernel_initializer,
                                                               kernel_constraint=self.dec_kernel_constraint,
                                                               kernel_regularizer=self.dec_kernel_reg)
            self.trans_conv2 = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=4, strides=1,
                                                               padding='same', activation=tf.nn.relu,
                                                               kernel_initializer=self.dec_kernel_initializer,
                                                               kernel_constraint=self.dec_kernel_constraint,
                                                               kernel_regularizer=self.dec_kernel_reg)
            self.conv3 = tf.keras.layers.Conv1D(filters=70, kernel_size=1, strides=2,
                                                padding='same', activation=tf.nn.softmax,
                                                kernel_initializer=self.dec_kernel_initializer,
                                                kernel_constraint=self.dec_kernel_constraint,
                                                kernel_regularizer=self.dec_kernel_reg)
        else:
            raise Exception(f'Encoder architecture is not defined for {self.dataset}.')

    def call(self, z, training=True):
        if self.dataset == 'mnist' or self.dataset == 'fashion' or self.dataset == 'mad_base':
            x_hat = self.reshape(self.dense2(self.dense1(z)))
        elif self.dataset == 'dsprite' or self.dataset == 'shapes3d' or self.dataset == 'mpi3d_toy'\
            or self.dataset == 'smallnorb':
            X = self.dense1(z)
            X = self.dense2(X)
            X = self.dense3(X)
            X = self.reshape(X)
            X = self.trans_conv1(X)
            X = self.trans_conv2(X)
            X = self.trans_conv3(X)
            x_hat = self.trans_conv4(X)
        elif self.dataset == 'svhn':
            x_hat = self.trans_conv4(self.trans_conv3(self.trans_conv2(self.trans_conv1(self.reshape(self.dense1(z))))))
        elif self.dataset == 'mnist_text' or self.dataset == 'dsprite_text' or self.dataset == 'shapes3d_text'\
            or self.dataset == 'mpi3d_toy_text' or self.dataset == 'smallnorb_text':
            z = self.dense0(z)
            x_hat = self.conv3(self.trans_conv2(self.trans_conv1(self.dense1(tf.expand_dims(z, axis=-1)))))
        return x_hat
