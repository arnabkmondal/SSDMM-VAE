import tensorflow as tf

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


class Encoder(tf.keras.Model):
    def __init__(self, dataset, kernel_init, kernel_reg, kernel_constraint, cont_z_dim, disc_z_dim, private_z_dim,
                 private_lat_flag, cont_lat_flag, disc_lat_flag):
        super(Encoder, self).__init__()
        self.dataset = dataset
        self.enc_kernel_initializer = kernel_init
        self.enc_kernel_reg = kernel_reg
        self.enc_kernel_constraint = kernel_constraint
        self.cont_z_dim = cont_z_dim
        self.disc_z_dim = disc_z_dim
        self.private_z_dim = private_z_dim
        self.private_lat_flag = private_lat_flag
        self.cont_lat_flag = cont_lat_flag
        self.disc_lat_flag = disc_lat_flag

        if self.dataset == 'mnist' or self.dataset == 'fashion' or self.dataset == 'mad_base':
            self.flatten = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu,
                                             kernel_initializer=self.enc_kernel_initializer,
                                             kernel_regularizer=self.enc_kernel_reg,
                                             kernel_constraint=self.enc_kernel_constraint)
        elif self.dataset == 'dsprite' or self.dataset == 'shapes3d':
            self.conv1 = Conv(32, 4)
            self.conv2 = Conv(32, 4)
            self.conv3 = Conv(32, 4)
            self.conv4 = Conv(32, 4)
            self.flatten = Flatten()
            self.dense1 = Dense(256, activation='relu')
            self.dense2 = Dense(256, activation='relu')
        elif self.dataset == 'mpi3d_toy' or self.dataset == 'smallnorb':
            self.conv1 = Conv(32, 4)
            self.conv2 = Conv(32, 4)
            self.conv3 = Conv(32, 4)
            self.conv4 = Conv(32, 4)
            self.flatten = Flatten()
            self.dense1 = Dense(256, activation='relu')
            self.dense2 = Dense(256, activation='relu')
        elif self.dataset == 'svhn':
            self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                padding='same', activation=tf.nn.relu,
                                                kernel_initializer=self.enc_kernel_initializer,
                                                kernel_regularizer=self.enc_kernel_reg,
                                                kernel_constraint=self.enc_kernel_constraint)
            self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                                padding='same', activation=tf.nn.relu,
                                                kernel_initializer=self.enc_kernel_initializer,
                                                kernel_regularizer=self.enc_kernel_reg,
                                                kernel_constraint=self.enc_kernel_constraint)
            self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                                padding='same', activation=tf.nn.relu,
                                                kernel_initializer=self.enc_kernel_initializer,
                                                kernel_regularizer=self.enc_kernel_reg,
                                                kernel_constraint=self.enc_kernel_constraint)
            self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2),
                                                padding='same', activation=tf.nn.relu,
                                                kernel_initializer=self.enc_kernel_initializer,
                                                kernel_regularizer=self.enc_kernel_reg,
                                                kernel_constraint=self.enc_kernel_constraint)
            self.flatten = tf.keras.layers.Flatten()
        elif self.dataset == 'mnist_text' or self.dataset == 'dsprite_text' or self.dataset == 'shapes3d_text' \
            or self.dataset == 'mpi3d_toy_text' or self.dataset == 'smallnorb_text':
            self.feature_extractor = tf.keras.models.Sequential([
                tf.keras.layers.Conv1D(filters=2 * self.disc_z_dim, kernel_size=1,
                                       padding='same', activation=tf.nn.relu,
                                       kernel_initializer=self.enc_kernel_initializer,
                                       kernel_regularizer=self.enc_kernel_reg,
                                       kernel_constraint=self.enc_kernel_constraint),
                tf.keras.layers.Conv1D(filters=2 * self.disc_z_dim, kernel_size=4,
                                       padding='same', activation=tf.nn.relu,
                                       kernel_initializer=self.enc_kernel_initializer,
                                       kernel_regularizer=self.enc_kernel_reg,
                                       kernel_constraint=self.enc_kernel_constraint),
                tf.keras.layers.Conv1D(filters=2 * self.disc_z_dim, kernel_size=4,
                                       padding='same', activation=tf.nn.relu,
                                       kernel_initializer=self.enc_kernel_initializer,
                                       kernel_regularizer=self.enc_kernel_reg,
                                       kernel_constraint=self.enc_kernel_constraint)
            ])
            self.flatten = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu,
                                             kernel_initializer=self.enc_kernel_initializer,
                                             kernel_regularizer=self.enc_kernel_reg,
                                             kernel_constraint=self.enc_kernel_constraint)
        else:
            raise Exception(f'Encoder architecture is not defined for {self.dataset}.')

        if self.private_lat_flag:
            self.private_cont_mu_fc = tf.keras.layers.Dense(units=self.private_z_dim, activation=None,
                                                            kernel_initializer=self.enc_kernel_initializer,
                                                            kernel_regularizer=self.enc_kernel_reg,
                                                            kernel_constraint=self.enc_kernel_constraint)
            self.private_cont_log_var_fc = tf.keras.layers.Dense(units=self.private_z_dim, activation=None,
                                                                 kernel_initializer=self.enc_kernel_initializer,
                                                                 kernel_regularizer=self.enc_kernel_reg,
                                                                 kernel_constraint=self.enc_kernel_constraint)

        if self.cont_lat_flag:
            self.shared_cont_mu_fc = tf.keras.layers.Dense(units=self.cont_z_dim, activation=None,
                                                           kernel_initializer=self.enc_kernel_initializer,
                                                           kernel_regularizer=self.enc_kernel_reg,
                                                           kernel_constraint=self.enc_kernel_constraint)
            self.shared_cont_log_var_fc = tf.keras.layers.Dense(units=self.cont_z_dim, activation=None,
                                                                kernel_initializer=self.enc_kernel_initializer,
                                                                kernel_regularizer=self.enc_kernel_reg,
                                                                kernel_constraint=self.enc_kernel_constraint)

        if self.disc_lat_flag:
            self.shared_disc_logits_fc = tf.keras.layers.Dense(units=self.disc_z_dim, activation=None,
                                                               kernel_initializer=self.enc_kernel_initializer,
                                                               kernel_regularizer=self.enc_kernel_reg,
                                                               kernel_constraint=self.enc_kernel_constraint)

    def call(self, x, training=True):
        if self.dataset == 'mnist' or self.dataset == 'fashion' or self.dataset == 'mad_base':
            h = self.fc1(self.flatten(x))
        elif self.dataset == 'dsprite' or self.dataset == 'shapes3d' or self.dataset == 'mpi3d_toy'\
            or self.dataset == 'smallnorb':
            X = self.conv1(x)
            X = self.conv2(X)
            X = self.conv3(X)
            X = self.conv4(X)
            X = self.flatten(X)
            X = self.dense1(X)
            h = self.dense2(X)
        elif self.dataset == 'svhn':
            h = self.flatten(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        elif self.dataset == 'mnist_text' or self.dataset == 'dsprite_text' or self.dataset == 'shapes3d_text' \
            or self.dataset == 'mpi3d_toy_text' or self.dataset == 'smallnorb_text':
            h = self.fc1(self.flatten(self.feature_extractor(x)))

        if self.private_lat_flag:
            private_cont_mu = self.private_cont_mu_fc(h)
            private_cont_log_var = self.private_cont_log_var_fc(h)

        if self.cont_lat_flag:
            shared_cont_mu = self.shared_cont_mu_fc(h)
            shared_cont_log_var = self.shared_cont_log_var_fc(h)

        if self.disc_lat_flag:
            shared_disc_logits = self.shared_disc_logits_fc(h)
            shared_disc_probs = tf.nn.softmax(shared_disc_logits)

        if self.private_lat_flag and self.cont_lat_flag and self.disc_lat_flag:
            return private_cont_mu, private_cont_log_var, shared_cont_mu, shared_cont_log_var, shared_disc_logits, \
                   shared_disc_probs
        elif self.private_lat_flag and self.cont_lat_flag:
            return private_cont_mu, private_cont_log_var, shared_cont_mu, shared_cont_log_var
        # elif self.dataset == 'mnist_text' or self.dataset == 'dsprite_text':
        #     return shared_disc_logits, shared_disc_probs
        elif self.private_lat_flag and self.disc_lat_flag:
            return private_cont_mu, private_cont_log_var, shared_disc_logits, shared_disc_probs
        elif self.cont_lat_flag and self.disc_lat_flag:
            return shared_cont_mu, shared_cont_log_var, shared_disc_logits, shared_disc_probs
        elif self.cont_lat_flag:
            return shared_cont_mu, shared_cont_log_var
