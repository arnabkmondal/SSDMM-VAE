import argparse
import tensorflow as tf
import numpy as np
from networks.Encoder import Encoder
from networks.Decoder import Decoder
from networks.Classifier import ImageClassifier, TextClassifier
import pickle
from utils import *

THRESHOLD = 10000
tau = 0.67
dll_gamma = 1000

private_min, private_max, private_num_iters, private_gamma = 0.0, 1.0, 30000, 150
disc_min, disc_max, disc_num_iters, disc_gamma = 0.0, 1.0, 30000, 150
cont_min, cont_max, cont_num_iters, cont_gamma = 0.0, 1.0, 30000, 150

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()


class SSDMMVAE:
    def __init__(self, d1_name, d2_name, bs, private_lat_flag, cont_lat_flag, disc_lat_flag, cont_z_dim1, disc_z_dim1,
                 private_z_dim1, cont_z_dim2, disc_z_dim2, private_z_dim2, lat_sampler, lr, supervision_percent,
                 sup_tr_frac, models_dir, samples_dir, gen_dir, recon_dir, training_steps, save_interval,
                 plot_interval):
        self.d1_name = d1_name
        self.d2_name = d2_name
        self.bs = bs
        self.private_lat_flag = private_lat_flag
        self.cont_lat_flag = cont_lat_flag
        self.disc_lat_flag = disc_lat_flag
        self.cont_z_dim1 = cont_z_dim1
        self.disc_z_dim1 = disc_z_dim1
        self.private_z_dim1 = private_z_dim1
        self.cont_z_dim2 = cont_z_dim2
        self.disc_z_dim2 = disc_z_dim2
        self.private_z_dim2 = private_z_dim2
        self.lat_sampler = lat_sampler
        self.lr = lr
        self.supervision_percent = supervision_percent
        self.sup_tr_frac = sup_tr_frac
        self.model_dir = models_dir
        self.samples_dir = samples_dir
        self.gen_dir = gen_dir
        self.recon_dir = recon_dir
        self.training_steps = training_steps
        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.kernel_initializer = tf.compat.v1.glorot_normal_initializer()
        self.bias_initializer = None
        self.enc_kernel_reg = None
        self.dec_kernel_reg = None
        self.enc_kernel_constraint = None
        self.dec_kernel_constraint = None
        self.disc_kernel_constraint = None
        self.gen_kernel_constraint = None
        self.lat_reg = None
        self.d1_train_images, self.d1_train_labels, self.d1_test_images, self.d1_test_labels, self.d1_side, \
        self.d1_channels = load_data(self.d1_name)
        self.d2_train_text, self.d2_train_labels, self.d2_test_text, self.d2_test_labels, self.d2_side, \
        self.d2_channels = load_data(self.d2_name)

        self.n_supervision = int(self.supervision_percent * max(self.d1_train_images.shape[0],
                                                                self.d2_train_text.shape[0]) * 0.01)

        self.d1_encoder = Encoder(
            self.d1_name, self.kernel_initializer, self.enc_kernel_reg, self.enc_kernel_constraint, self.cont_z_dim1,
            self.disc_z_dim1, self.private_z_dim1, self.private_lat_flag, self.cont_lat_flag, self.disc_lat_flag)
        self.d2_encoder = Encoder(
            self.d2_name, self.kernel_initializer, self.enc_kernel_reg, self.enc_kernel_constraint, self.cont_z_dim2,
            self.disc_z_dim2, self.private_z_dim2, self.private_lat_flag, self.cont_lat_flag, self.disc_lat_flag)
        self.d1_decoder = Decoder(
            self.d1_name, self.kernel_initializer, self.dec_kernel_reg, self.dec_kernel_constraint)
        self.d2_decoder = Decoder(
            self.d2_name, self.kernel_initializer, self.dec_kernel_reg, self.dec_kernel_constraint)

        self.d1_classifier = ImageClassifier(len(np.unique(self.d1_train_labels)))
        self.d2_classifier = TextClassifier(len(np.unique(self.d2_train_labels)))

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr, name='ssdmmVAE-opt')

        self.n_d1_pixel = self.d1_side * self.d1_side * self.d1_channels
        self.n_d2_pixel = self.d2_side * self.d2_side * self.d2_channels

    def _product_of_gaussian_expert(self, mu, log_var, eps=1e-8):
        var = tf.math.exp(log_var) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = tf.reduce_sum(mu * T, axis=0) / tf.reduce_sum(T, axis=0)
        pd_var = 1. / tf.reduce_sum(T, axis=0)
        pd_log_var = tf.math.log(pd_var + eps)
        return pd_mu, pd_log_var

    def _reparameterize_gaussian(self, mu, log_var):
        sd = tf.exp(0.5 * log_var, name='exp-log-var')
        lat = tf.add(mu, tf.multiply(tf.random.normal(sd.shape, name='standard-Gaussian'), sd, name='scale'),
                     name='shift')
        return lat

    def _sample_gumbel(self, shape, eps=1e-20):
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def _reparameterize_gumbel_softmax(self, probs, temperature, hard=False, eps=1e-20):
        gumbel_softmax_sample = tf.math.log(probs + eps) + self._sample_gumbel(probs.shape)
        y = tf.nn.softmax(gumbel_softmax_sample / temperature)

        if hard:
            y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(y, 1, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y

        return y

    @tf.function
    def train_step(self, d1_x=None, d2_x=None, d1_y=None, d2_y=None, target=None, private_cap=0, disc_cap=0,
                   cont_cap=0):
        with tf.GradientTape() as tape:
            batch_size = d1_x.shape[0] if d1_x is not None else d2_x.shape[0]
            cont_shared_mu = tf.zeros((1, batch_size, self.cont_z_dim), dtype=tf.float32) if self.cont_lat_flag \
                else None
            cont_shared_log_var = tf.zeros((1, batch_size, self.cont_z_dim), dtype=tf.float32) if self.cont_lat_flag \
                else None

            if d1_x is not None:
                if self.private_lat_flag and self.cont_lat_flag and self.disc_lat_flag:
                    d1_private_lat_mu, d1_private_lat_log_var, d1_cont_shared_mu, d1_cont_shared_log_var, \
                    d1_disc_shared_logits, d1_disc_shared_probs = self.d1_encoder(d1_x)
                elif self.private_lat_flag and self.cont_lat_flag:
                    d1_private_lat_mu, d1_private_lat_log_var, d1_cont_shared_mu, d1_cont_shared_log_var = \
                        self.d1_encoder(d1_x)
                elif self.private_lat_flag and self.disc_lat_flag:
                    d1_private_lat_mu, d1_private_lat_log_var, d1_disc_shared_logits, d1_disc_shared_probs = \
                        self.d1_encoder(d1_x)
                elif self.cont_lat_flag and self.disc_lat_flag:
                    d1_cont_shared_mu, d1_cont_shared_log_var, d1_disc_shared_logits, d1_disc_shared_probs = \
                        self.d1_encoder(d1_x)
                elif self.cont_lat_flag:
                    d1_cont_shared_mu, d1_cont_shared_log_var = self.d1_encoder(d1_x)
                elif self.disc_lat_flag:
                    d1_disc_shared_logits, d1_disc_shared_probs = self.d1_encoder(d1_x)

                if self.private_lat_flag:
                    d1_private_lat = self._reparameterize_gaussian(d1_private_lat_mu, d1_private_lat_log_var)

                if self.cont_lat_flag:
                    d1_cont_shared_mu = tf.expand_dims(d1_cont_shared_mu, axis=0)
                    d1_cont_shared_log_var = tf.expand_dims(d1_cont_shared_log_var, axis=0)

                    cont_shared_mu = tf.concat((cont_shared_mu, d1_cont_shared_mu), axis=0)
                    cont_shared_log_var = tf.concat((cont_shared_log_var, d1_cont_shared_log_var), axis=0)

            else:
                d1_private_lat = tf.random.normal((batch_size, self.private_z_dim1)) if self.private_lat_flag else None

            if d2_x is not None:
                if self.private_lat_flag and self.cont_lat_flag and self.disc_lat_flag:
                    d2_private_lat_mu, d2_private_lat_log_var, d2_cont_shared_mu, d2_cont_shared_log_var, \
                    d2_disc_shared_logits, d2_disc_shared_probs = self.d2_encoder(d2_x)
                elif self.private_lat_flag and self.cont_lat_flag:
                    d2_private_lat_mu, d2_private_lat_log_var, d2_cont_shared_mu, d2_cont_shared_log_var = \
                        self.d2_encoder(d2_x)
                # elif self.d2_name == 'mnist_text' or self.d2_name == 'dsprite_text':
                #     d2_disc_shared_logits, d2_disc_shared_probs = self.d2_encoder(d2_x)
                elif self.private_lat_flag and self.disc_lat_flag:
                    d2_private_lat_mu, d2_private_lat_log_var, d2_disc_shared_logits, d2_disc_shared_probs = \
                        self.d2_encoder(d2_x)
                elif self.cont_lat_flag and self.disc_lat_flag:
                    d2_cont_shared_mu, d2_cont_shared_log_var, d2_disc_shared_logits, d2_disc_shared_probs = \
                        self.d2_encoder(d2_x)
                elif self.cont_lat_flag:
                    d2_cont_shared_mu, d2_cont_shared_log_var = self.d2_encoder(d2_x)
                elif self.disc_lat_flag:
                    d2_disc_shared_logits, d2_disc_shared_probs = self.d2_encoder(d2_x)

                if self.private_lat_flag:
                    d2_private_lat = self._reparameterize_gaussian(d2_private_lat_mu, d2_private_lat_log_var)

                if self.cont_lat_flag:
                    d2_cont_shared_mu = tf.expand_dims(d2_cont_shared_mu, axis=0)
                    d2_cont_shared_log_var = tf.expand_dims(d2_cont_shared_log_var, axis=0)

                    cont_shared_mu = tf.concat((cont_shared_mu, d2_cont_shared_mu), axis=0)
                    cont_shared_log_var = tf.concat((cont_shared_log_var, d2_cont_shared_log_var), axis=0)
            else:
                d2_private_lat = tf.random.normal(
                    (batch_size, self.private_z_dim2)) if self.private_lat_flag else None

            if self.disc_lat_flag:
                if d1_x is not None and d2_x is not None:
                    disc_shared_lat = 0.5 * (d1_disc_shared_probs + d2_disc_shared_probs)
                elif d1_x is None:
                    disc_shared_lat = d2_disc_shared_probs
                else:
                    disc_shared_lat = d1_disc_shared_probs

            if self.cont_lat_flag:
                cont_shared_mu, cont_shared_log_var = self._product_of_gaussian_expert(cont_shared_mu,
                                                                                       cont_shared_log_var)
                cont_shared_lat = self._reparameterize_gaussian(cont_shared_mu, cont_shared_log_var)

            if self.private_lat_flag and self.disc_lat_flag and self.cont_lat_flag:
                d1_z = tf.concat((d1_private_lat, cont_shared_lat, disc_shared_lat), axis=1)
                d2_z = tf.concat((d2_private_lat, cont_shared_lat, disc_shared_lat), axis=1)
            elif self.private_lat_flag and self.disc_lat_flag:
                d1_z = tf.concat((d1_private_lat, disc_shared_lat), axis=1)
                # if self.d2_name == 'mnist_text' or self.d2_name == 'dsprite_text':
                #     d2_z = disc_shared_lat
                # else:
                d2_z = tf.concat((d2_private_lat, disc_shared_lat), axis=1)
            elif self.private_lat_flag and self.cont_lat_flag:
                d1_z = tf.concat((d1_private_lat, cont_shared_lat), axis=1)
                d2_z = tf.concat((d2_private_lat, cont_shared_lat), axis=1)
            elif self.cont_lat_flag and self.disc_lat_flag:
                d1_z = tf.concat((cont_shared_lat, disc_shared_lat), axis=1)
                d2_z = tf.concat((cont_shared_lat, disc_shared_lat), axis=1)
            elif self.cont_lat_flag:
                d1_z = cont_shared_lat
                d2_z = cont_shared_lat

            d1_x_hat = self.d1_decoder(d1_z)
            d2_x_hat = self.d2_decoder(d2_z)

            # d1_recon_loss = bce(d1_y, d1_x_hat) if d1_x is not None else mae(self.d1_classifier(d1_y, False),
            #                                                                  self.d1_classifier(d1_x_hat, False))

            d1_recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(d1_y, d1_x_hat), axis=(1, 2)
                )
                ) if d1_x is not None else tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.mean_absolute_error(self.d1_classifier(d1_y, False), self.d1_classifier(d1_x_hat, False))#, axis=(1)
                    )
                )

            if self.d2_name == 'mnist_text' or self.d2_name == 'dsprite_text' or self.d2_name == 'shapes3d_text' \
                or self.d2_name == 'mpi3d_toy_text' or self.d2_name == 'smallnorb_text':
                # d2_recon_loss = bce(d2_y, d2_x_hat) if d2_x is not None else mae(self.d2_classifier(d2_y, False),
                #                                                                  self.d2_classifier(d2_x_hat, False))
                d2_recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(d2_y, d2_x_hat), axis=(1)
                )
                ) if d2_x is not None else tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.mean_absolute_error(self.d2_classifier(d2_y, False), self.d2_classifier(d2_x_hat, False))#, axis=(1)
                    )
                )
            else:
                d2_recon_loss = bce(d2_y, d2_x_hat) + ssim_loss(d2_y, d2_x_hat)
            d1_private_kld_loss = kld_loss(d1_private_lat_mu, d1_private_lat_log_var) if (self.private_lat_flag and
                                                                                          d1_x is not None) else 0
            d2_private_kld_loss = kld_loss(d2_private_lat_mu, d2_private_lat_log_var) if (self.private_lat_flag and
                                                                                          d2_x is not None) else 0
            cont_kld_loss = kld_loss(cont_shared_mu, cont_shared_log_var) if self.cont_lat_flag else None
            classification_loss = bce(target, disc_shared_lat) if target is not None else 0

            # total_loss = self.n_d1_pixel * d1_recon_loss + 100 * d2_recon_loss
            total_loss = d1_recon_loss + d2_recon_loss

            if self.private_lat_flag:
                if d1_x is not None:
                    total_loss = total_loss + private_gamma * tf.abs(
                        tf.cast(private_cap, tf.float32) - d1_private_kld_loss)
                if d2_x is not None:
                    total_loss = total_loss + private_gamma * tf.abs(
                        tf.cast(private_cap, tf.float32) - d2_private_kld_loss)

            if self.disc_lat_flag:
                total_loss = total_loss + dll_gamma * classification_loss if target is not None else total_loss

            if self.cont_lat_flag:
                total_loss = total_loss + cont_gamma * tf.abs(tf.cast(cont_cap, tf.float32) - cont_kld_loss)

        if d1_x is not None and d2_x is not None:
            trainable_parameters = self.d1_encoder.trainable_variables + self.d1_decoder.trainable_variables + \
                                   self.d2_encoder.trainable_variables + self.d2_decoder.trainable_variables
        elif d1_x is not None:
            trainable_parameters = self.d1_encoder.trainable_variables + self.d1_decoder.trainable_variables + \
                                   self.d2_decoder.trainable_variables
        elif d2_x is not None:
            trainable_parameters = self.d2_encoder.trainable_variables + self.d2_decoder.trainable_variables + \
                                   self.d1_decoder.trainable_variables

        g = tape.gradient(total_loss, trainable_parameters)
        self.opt.apply_gradients(zip(g, trainable_parameters))

        return d1_recon_loss, d2_recon_loss, d1_private_kld_loss, d2_private_kld_loss, \
               classification_loss, cont_kld_loss, total_loss

    @tf.function
    def reconstruct_or_generate(self, d1_x=None, d2_x=None, bs=100):
        if d1_x is None and d2_x is None:
            batch_size = bs
            d1_private_lat = tf.random.normal((batch_size, self.private_z_dim1)) if self.private_lat_flag else None
            d2_private_lat = tf.random.normal((batch_size, self.private_z_dim2)) if self.private_lat_flag else None
            disc_shared_lat = sample_categorical(batch_size, self.disc_z_dim1) if self.disc_lat_flag else None
            cont_shared_lat = tf.random.normal((batch_size, self.cont_z_dim1)) if self.cont_lat_flag else None
        else:
            batch_size = d1_x.shape[0] if d1_x is not None else d2_x.shape[0]
            cont_shared_mu = tf.zeros((1, batch_size, self.cont_z_dim1), dtype=tf.float32) if self.cont_lat_flag \
                else None
            cont_shared_log_var = tf.zeros((1, batch_size, self.cont_z_dim1), dtype=tf.float32) if self.cont_lat_flag \
                else None

            if d1_x is not None:
                if self.private_lat_flag and self.cont_lat_flag and self.disc_lat_flag:
                    d1_private_lat_mu, d1_private_lat_log_var, d1_cont_shared_mu, d1_cont_shared_log_var, \
                    d1_disc_shared_logits, d1_disc_shared_probs = self.d1_encoder(d1_x)
                elif self.private_lat_flag and self.cont_lat_flag:
                    d1_private_lat_mu, d1_private_lat_log_var, d1_cont_shared_mu, d1_cont_shared_log_var = \
                        self.d1_encoder(d1_x)
                elif self.private_lat_flag and self.disc_lat_flag:
                    d1_private_lat_mu, d1_private_lat_log_var, d1_disc_shared_logits, d1_disc_shared_probs = \
                        self.d1_encoder(d1_x)
                elif self.cont_lat_flag and self.disc_lat_flag:
                    d1_cont_shared_mu, d1_cont_shared_log_var, d1_disc_shared_logits, d1_disc_shared_probs = \
                        self.d1_encoder(d1_x)
                elif self.cont_lat_flag:
                    d1_cont_shared_mu, d1_cont_shared_log_var = self.d1_encoder(d1_x)

                if self.private_lat_flag:
                    d1_private_lat = self._reparameterize_gaussian(d1_private_lat_mu, d1_private_lat_log_var)

                if self.cont_lat_flag:
                    d1_cont_shared_mu = tf.expand_dims(d1_cont_shared_mu, axis=0)
                    d1_cont_shared_log_var = tf.expand_dims(d1_cont_shared_log_var, axis=0)

                    cont_shared_mu = tf.concat((cont_shared_mu, d1_cont_shared_mu), axis=0)
                    cont_shared_log_var = tf.concat((cont_shared_log_var, d1_cont_shared_log_var), axis=0)

            else:
                d1_private_lat = tf.random.normal((batch_size, self.private_z_dim1)) if self.private_lat_flag else None

            if d2_x is not None:
                if self.private_lat_flag and self.cont_lat_flag and self.disc_lat_flag:
                    d2_private_lat_mu, d2_private_lat_log_var, d2_cont_shared_mu, d2_cont_shared_log_var, \
                    d2_disc_shared_logits, d2_disc_shared_probs = self.d2_encoder(d2_x)
                elif self.private_lat_flag and self.cont_lat_flag:
                    d2_private_lat_mu, d2_private_lat_log_var, d2_cont_shared_mu, d2_cont_shared_log_var = \
                        self.d2_encoder(d2_x)
                # elif self.d2_name == 'mnist_text' or self.d2_name == 'dsprite_text':
                #     d2_disc_shared_logits, d2_disc_shared_probs = self.d2_encoder(d2_x)
                elif self.private_lat_flag and self.disc_lat_flag:
                    d2_private_lat_mu, d2_private_lat_log_var, d2_disc_shared_logits, d2_disc_shared_probs = \
                        self.d2_encoder(d2_x)
                elif self.cont_lat_flag and self.disc_lat_flag:
                    d2_cont_shared_mu, d2_cont_shared_log_var, d2_disc_shared_logits, d2_disc_shared_probs = \
                        self.d2_encoder(d2_x)
                elif self.cont_lat_flag:
                    d2_cont_shared_mu, d2_cont_shared_log_var = self.d2_encoder(d2_x)

                if self.private_lat_flag:
                    d2_private_lat = d2_private_lat_mu

                elif self.cont_lat_flag:
                    d2_cont_shared_mu = tf.expand_dims(d2_cont_shared_mu, axis=0)
                    d2_cont_shared_log_var = tf.expand_dims(d2_cont_shared_log_var, axis=0)

                    cont_shared_mu = tf.concat((cont_shared_mu, d2_cont_shared_mu), axis=0)
                    cont_shared_log_var = tf.concat((cont_shared_log_var, d2_cont_shared_log_var), axis=0)
            else:
                d2_private_lat = tf.random.normal(
                    (batch_size, self.private_z_dim2)) if self.private_lat_flag else None

            if self.disc_lat_flag:
                if d1_x is not None and d2_x is not None:
                    disc_shared_lat = 0.5 * (d1_disc_shared_probs + d2_disc_shared_probs)
                elif d1_x is None:
                    disc_shared_lat = d2_disc_shared_probs
                else:
                    disc_shared_lat = d1_disc_shared_probs

            if self.cont_lat_flag:
                cont_shared_mu, cont_shared_log_var = self._product_of_gaussian_expert(cont_shared_mu,
                                                                                       cont_shared_log_var)
                cont_shared_lat = cont_shared_mu

        if self.private_lat_flag and self.disc_lat_flag and self.cont_lat_flag:
            d1_z = tf.concat((d1_private_lat, cont_shared_lat, disc_shared_lat), axis=1)
            d2_z = tf.concat((d2_private_lat, cont_shared_lat, disc_shared_lat), axis=1)
        elif self.private_lat_flag and self.disc_lat_flag:
            d1_z = tf.concat((d1_private_lat, disc_shared_lat), axis=1)
            # if self.d2_name == 'mnist_text' or self.d2_name == 'dsprite_text':
            #     d2_z = disc_shared_lat
            # else:
            d2_z = tf.concat((d2_private_lat, disc_shared_lat), axis=1)
        elif self.private_lat_flag and self.cont_lat_flag:
            d1_z = tf.concat((d1_private_lat, cont_shared_lat), axis=1)
            d2_z = tf.concat((d2_private_lat, cont_shared_lat), axis=1)
        elif self.cont_lat_flag and self.disc_lat_flag:
            d1_z = tf.concat((cont_shared_lat, disc_shared_lat), axis=1)
            d2_z = tf.concat((cont_shared_lat, disc_shared_lat), axis=1)
        elif self.cont_lat_flag:
            d1_z = cont_shared_lat
            d2_z = cont_shared_lat

        d1_x_hat = self.d1_decoder(d1_z)
        d2_x_hat = self.d2_decoder(d2_z)

        return d1_x_hat, d2_x_hat

    def train(self):
        d1_jrl_buf = []
        d1_srl_buf = []
        d1_crl_buf = []
        d2_jrl_buf = []
        d2_srl_buf = []
        d2_crl_buf = []
        d1_pkl_buf = []
        d2_pkl_buf = []
        joint_classification_loss_buf = []
        d1_classification_loss_buf = []
        d2_classification_loss_buf = []
        cont_jskl_buf = []
        cont_d1skl_buf = []
        cont_d2skl_buf = []
        steps_buf = []

        # data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        #     featurewise_center=False,
        #     samplewise_center=False,
        #     featurewise_std_normalization=False,
        #     samplewise_std_normalization=False,
        #     zca_whitening=False,
        #     rotation_range=15,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1,
        #     horizontal_flip=True,
        #     vertical_flip=True
        # )
        # data_gen.fit(self.d1_train_images[:self.n_supervision])
        # self.d1_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # self.d1_classifier.fit(data_gen.flow(self.d1_train_images[:self.n_supervision],
        #                                      to_one_hot(
        #                                          self.d1_train_labels[:self.n_supervision], len(np.unique(self.d1_train_labels))),
        #                                      batch_size=32),
        #                        epochs=50)
        # del data_gen

        d1_clf_epoch = 50 if self.d1_name == 'mpi3d_toy' or self.d1_name == 'smallnorb' else 10

        self.d1_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        self.d1_classifier.fit(x=self.d1_train_images[:self.n_supervision],
                               y=to_one_hot(
                                   self.d1_train_labels[:self.n_supervision], len(np.unique(self.d1_train_labels))),
                               batch_size=32,
                               epochs=d1_clf_epoch)

        self.d2_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        self.d2_classifier.fit(x=self.d2_train_text[:self.n_supervision],
                               y=to_one_hot(
                                   self.d2_train_labels[:self.n_supervision], len(np.unique(self.d2_train_labels))),
                               batch_size=32,
                               epochs=10)

        for step in range(self.training_steps):
            if step < int(3 * self.sup_tr_frac * self.training_steps) and step % 3 == 0:
                d1_mb, d2_mb, t_mb = get_paired_sample_with_labels(
                    self.d1_train_images[:self.n_supervision], self.d2_train_text[:self.n_supervision],
                    self.d1_train_labels[:self.n_supervision], self.bs)
                t_mb = to_one_hot(t_mb, len(np.unique(self.d1_train_labels)))
            else:
                d1_mb, d2_mb = get_paired_sample(self.d1_train_images, self.d2_train_text, self.bs)
                t_mb = None

            private_cap_current = (private_max - private_min) * step / np.float32(private_num_iters) + private_min
            private_cap_current = min(private_cap_current, private_max)

            disc_cap_current = (disc_max - disc_min) * step / np.float32(disc_num_iters) + disc_min
            disc_cap_current = min(disc_cap_current, disc_max)
            disc_theoretical_max = np.log(max(self.disc_z_dim1, self.disc_z_dim2)) if self.disc_lat_flag else 0
            disc_cap_current = min(disc_cap_current, disc_theoretical_max)

            cont_cap_current = (cont_max - cont_min) * step / np.float32(cont_num_iters) + cont_min
            cont_cap_current = min(cont_cap_current, cont_max)

            d1_jrl, d2_jrl, d1_pkl, d2_pkl, j_cl, cont_jskl, total_loss = self.train_step(
                d1_x=d1_mb, d2_x=d2_mb, d1_y=d1_mb, d2_y=d2_mb, target=t_mb,
                private_cap=private_cap_current, disc_cap=disc_cap_current, cont_cap=cont_cap_current)
            d1_srl, d2_crl, d1_pkl, _, d1_cl, cont_d1skl, _ = self.train_step(
                d1_x=d1_mb, d2_x=None, d1_y=d1_mb, d2_y=d2_mb, target=t_mb,
                private_cap=private_cap_current, disc_cap=disc_cap_current, cont_cap=cont_cap_current)
            d1_crl, d2_srl, _, d2_pkl, d2_cl, cont_d2skl, _ = self.train_step(
                d1_x=None, d2_x=d2_mb, d1_y=d1_mb, d2_y=d2_mb, target=t_mb,
                private_cap=private_cap_current, disc_cap=disc_cap_current, cont_cap=cont_cap_current)

            if step % (self.plot_interval // 10) == 0:
                d1_mb, d2_mb = get_paired_sample(self.d1_test_images, self.d2_test_text, 100)

                r_d1_j, r_d2_j = self.reconstruct_or_generate(d1_x=d1_mb, d2_x=d2_mb)
                r_d1_d1, r_d2_d1 = self.reconstruct_or_generate(d1_x=d1_mb, d2_x=None)
                r_d1_d2, r_d2_d2 = self.reconstruct_or_generate(d1_x=None, d2_x=d2_mb)
                g_d1, g_d2 = self.reconstruct_or_generate(d1_x=None, d2_x=None)

                image_grid(images=d1_mb[0:8], texts=r_d1_j[0:8], height=8.0, width=32.0,
                           n_row=2, n_col=8, sv_path=f'{self.samples_dir}r_{self.d1_name}_j_{step}.png', dpi=None,
                           is_text1=False, is_text2=False)
                image_grid(images=d1_mb[0:8], texts=r_d1_d1[0:8], height=8.0, width=32.0,
                           n_row=2, n_col=8, sv_path=f'{self.samples_dir}r_{self.d1_name}_s_{step}.png', dpi=None,
                           is_text1=False, is_text2=False)
                image_grid(images=d1_mb[0:8], texts=r_d1_d2[0:8], height=8.0, width=32.0,
                           n_row=2, n_col=8, sv_path=f'{self.samples_dir}r_{self.d1_name}_c_{step}.png', dpi=None,
                           is_text1=False, is_text2=False)
                image_grid(images=d2_mb[0:8], texts=r_d2_j[0:8], height=8.0, width=32.0,
                           n_row=2, n_col=8, sv_path=f'{self.samples_dir}r_{self.d2_name}_j_{step}.png', dpi=None,
                           is_text1=True, is_text2=True)
                image_grid(images=d2_mb[0:8], texts=r_d2_d2[0:8], height=8.0, width=32.0,
                           n_row=2, n_col=8, sv_path=f'{self.samples_dir}r_{self.d2_name}_s_{step}.png', dpi=None,
                           is_text1=True, is_text2=True)
                image_grid(images=d2_mb[0:8], texts=r_d2_d1[0:8], height=8.0, width=32.0,
                           n_row=2, n_col=8, sv_path=f'{self.samples_dir}r_{self.d2_name}_c_{step}.png', dpi=None,
                           is_text1=True, is_text2=True)
                image_grid(images=g_d1[0:8], texts=g_d1[8:], height=8.0, width=32.0, n_row=2, n_col=8,
                           sv_path=f'{self.samples_dir}g_{self.d1_name}_{step}.png', dpi=None, is_text1=False,
                           is_text2=False)
                image_grid(images=g_d2[0:8], texts=g_d2[8:], height=8.0, width=32.0, n_row=2, n_col=8,
                           sv_path=f'{self.samples_dir}g_{self.d2_name}_{step}.png', dpi=None, is_text1=True,
                           is_text2=True)

                d1_jrl_buf.append(d1_jrl)
                d1_srl_buf.append(d1_srl)
                d1_crl_buf.append(d1_crl)
                d2_jrl_buf.append(d2_jrl)
                d2_srl_buf.append(d2_srl)
                d2_crl_buf.append(d2_crl)

                d1_pkl_buf.append(d1_pkl)
                d2_pkl_buf.append(d2_pkl)

                d1_classification_loss_buf.append(d1_cl)
                d2_classification_loss_buf.append(d2_cl)
                joint_classification_loss_buf.append(j_cl)

                cont_jskl_buf.append(cont_jskl)
                cont_d1skl_buf.append(cont_d1skl)
                cont_d2skl_buf.append(cont_d2skl)

                steps_buf.append(step)

                print(HEADER + f'Training SSDMMVAE on {self.d1_name.upper()} and {self.d2_name.upper()}, '
                               f'LR: {self.opt.lr.numpy()}\n' + END_C)
                print(WARNING + f'Step: {steps_buf[-1]}/{self.training_steps}, ' + END_C + OK_BLUE +
                      f'D1 Joint Recon. Loss: {d1_jrl_buf[-1]}, D1 Self Recon. Loss: {d1_srl_buf[-1]}, '
                      f'D1 Cross Recon. Loss: {d1_crl_buf[-1]}, '
                      f'D2 Joint Recon. Loss: {d2_jrl_buf[-1]}, D2 Self Recon. Loss: {d2_srl_buf[-1]}, '
                      f'D2 Cross Recon. Loss: {d2_crl_buf[-1]}, '
                      f'D1 Private KL: {d1_pkl_buf[-1]}, D2 Private KL: {d2_pkl_buf[-1]}, '
                      f'D1 Classification Loss: {d1_classification_loss_buf[-1]}, D2 Classification Loss:'
                      f' {d2_classification_loss_buf[-1]}, Joint Classification Loss:'
                      f' {joint_classification_loss_buf[-1]}, '
                      f'D1 Cont. Shared KL: {cont_d1skl_buf[-1]}, D2 Cont. Shared KL: {cont_d2skl_buf[-1]}, '
                      f'Joint Cont. Shared KL: {cont_jskl_buf[-1]}, '
                      f'Total Loss: {total_loss}\n'
                      + END_C)

            if step % self.plot_interval == 0 and step > 0:
                plot_graph(x=steps_buf, y=d1_jrl_buf, x_label='Steps', y_label='D1 Joint Recon Loss',
                           samples_dir=self.samples_dir, img_name='loss_d1_joint_recon.png')
                plot_graph(x=steps_buf, y=d1_srl_buf, x_label='Steps', y_label='D1 Self Recon Loss',
                           samples_dir=self.samples_dir, img_name='loss_d1_self_recon.png')
                plot_graph(x=steps_buf, y=d1_crl_buf, x_label='Steps', y_label='D1 Cross Recon Loss',
                           samples_dir=self.samples_dir, img_name='loss_d1_cross_recon.png')

                plot_graph(x=steps_buf, y=d2_jrl_buf, x_label='Steps', y_label='D2 Joint Recon Loss',
                           samples_dir=self.samples_dir, img_name='loss_d2_joint_recon.png')
                plot_graph(x=steps_buf, y=d2_srl_buf, x_label='Steps', y_label='D2 Self Recon Loss',
                           samples_dir=self.samples_dir, img_name='loss_d2_self_recon.png')
                plot_graph(x=steps_buf, y=d2_crl_buf, x_label='Steps', y_label='D2 Cross Recon Loss',
                           samples_dir=self.samples_dir, img_name='loss_d2_cross_recon.png')

                plot_graph(x=steps_buf, y=d1_pkl_buf, x_label='Steps', y_label='D1 Private KL',
                           samples_dir=self.samples_dir, img_name='loss_d1_pkl.png')
                plot_graph(x=steps_buf, y=d2_pkl_buf, x_label='Steps', y_label='D2 Private KL',
                           samples_dir=self.samples_dir, img_name='loss_d2_pkl.png')

                plot_graph(x=steps_buf, y=d1_classification_loss_buf, x_label='Steps', y_label='D1 Classification Loss',
                           samples_dir=self.samples_dir, img_name='loss_d1_classification.png')
                plot_graph(x=steps_buf, y=d2_classification_loss_buf, x_label='Steps', y_label='D2 Classification Loss',
                           samples_dir=self.samples_dir, img_name='loss_d2_classification.png')
                plot_graph(x=steps_buf, y=joint_classification_loss_buf, x_label='Steps',
                           y_label='Joint Classification Loss',
                           samples_dir=self.samples_dir, img_name='loss_joint_classification.png')

                plot_graph(x=steps_buf, y=cont_d1skl_buf, x_label='Steps', y_label='D1 Continuous Shared KL',
                           samples_dir=self.samples_dir, img_name='loss_d1_cont_skl.png')
                plot_graph(x=steps_buf, y=cont_d2skl_buf, x_label='Steps', y_label='D2 Continuous Shared KL',
                           samples_dir=self.samples_dir, img_name='loss_d2_cont_skl.png')
                plot_graph(x=steps_buf, y=cont_jskl_buf, x_label='Steps', y_label='Joint Continuous Shared KL',
                           samples_dir=self.samples_dir, img_name='loss_exp_cont_skl.png')

            if step % self.save_interval == 0 and step > THRESHOLD - 1:

                tf.keras.models.save_model(self.d1_encoder, f'{self.model_dir}/d1_encoder_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf')
                tf.keras.models.save_model(self.d1_decoder, f'{self.model_dir}/d1_decoder_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf')
                tf.keras.models.save_model(self.d2_encoder, f'{self.model_dir}/d2_encoder_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf')
                tf.keras.models.save_model(self.d2_decoder, f'{self.model_dir}/d2_decoder_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf')

                d1_gen_img = np.ndarray(shape=(10000, self.d1_side, self.d1_side, self.d1_channels), dtype=np.float32)
                if self.d2_name == 'mnist_text' or self.d2_name == 'dsprite_text':
                    d2_gen_img = np.ndarray(shape=(10000, 8, 70), dtype=np.float32)
                elif self.d2_name == 'shapes3d_text' or self.d2_name == 'mpi3d_toy_text' or self.d2_name == 'smallnorb_text':
                    d2_gen_img = np.ndarray(shape=(10000, 10, 70), dtype=np.float32)
                for j in range(10000 // 100):
                    d1_gen_batch, d2_gen_batch = self.reconstruct_or_generate(d1_x=None, d2_x=None, bs=100)
                    d1_gen_img[(j * 100):((j + 1) * 100)] = d1_gen_batch
                    d2_gen_img[(j * 100):((j + 1) * 100)] = d2_gen_batch
                np.save('{}generated_{}_images_{}.npy'.format(self.gen_dir, self.d1_name, step), d1_gen_img)
                np.save('{}generated_{}_images_{}.npy'.format(self.gen_dir, self.d2_name, step), d2_gen_img)

                recon_d1_joint = np.ndarray(shape=(10000, self.d1_side, self.d1_side, self.d1_channels),
                                            dtype=np.float32)
                recon_d1_d1 = np.ndarray(shape=(10000, self.d1_side, self.d1_side, self.d1_channels), dtype=np.float32)
                recon_d1_d2 = np.ndarray(shape=(10000, self.d1_side, self.d1_side, self.d1_channels), dtype=np.float32)
                
                if self.d2_name == 'mnist_text' or self.d2_name == 'dsprite_text':
                    recon_d2_joint = np.ndarray(shape=(10000, 8, 70),
                                            dtype=np.float32)
                    recon_d2_d2 = np.ndarray(shape=(10000, 8, 70), dtype=np.float32)
                    recon_d2_d1 = np.ndarray(shape=(10000, 8, 70), dtype=np.float32)
                elif self.d2_name == 'shapes3d_text' or self.d2_name == 'mpi3d_toy_text' or self.d2_name == 'smallnorb_text':
                    recon_d2_joint = np.ndarray(shape=(10000, 10, 70),
                                            dtype=np.float32)
                    recon_d2_d2 = np.ndarray(shape=(10000, 10, 70), dtype=np.float32)
                    recon_d2_d1 = np.ndarray(shape=(10000, 10, 70), dtype=np.float32)
                for j in range(10000 // 100):
                    if self.d2_name == 'mnist_text' or self.d2_name == 'dsprite_text' or self.d2_name == 'shapes3d_text' \
                        or self.d2_name == 'mpi3d_toy_text' or self.d2_name == 'smallnorb_text':
                        d1_mb = self.d1_test_images[(j * 100):((j + 1) * 100)].astype(np.float32)
                        d2_mb = self.d2_test_text[(j * 100):((j + 1) * 100)]
                    else:
                        d1_mb = self.d1_test_images[self.d1_test_idx[(j * 100):((j + 1) * 100)]].astype(np.float32)
                        d2_mb = self.d2_test_text[self.d2_test_idx[(j * 100):((j + 1) * 100)]]

                    r_d1_j, r_d2_j = self.reconstruct_or_generate(d1_x=d1_mb, d2_x=d2_mb)
                    r_d1_d1, r_d2_d1 = self.reconstruct_or_generate(d1_x=d1_mb, d2_x=None)
                    r_d1_d2, r_d2_d2 = self.reconstruct_or_generate(d1_x=None, d2_x=d2_mb)

                    recon_d1_joint[(j * 100):((j + 1) * 100)] = r_d1_j
                    recon_d1_d1[(j * 100):((j + 1) * 100)] = r_d1_d1
                    recon_d1_d2[(j * 100):((j + 1) * 100)] = r_d1_d2
                    recon_d2_joint[(j * 100):((j + 1) * 100)] = r_d2_j
                    recon_d2_d2[(j * 100):((j + 1) * 100)] = r_d2_d2
                    recon_d2_d1[(j * 100):((j + 1) * 100)] = r_d2_d1

                np.save('{}recon_{}_joint_{}.npy'.format(self.recon_dir, self.d1_name, step), recon_d1_joint)
                np.save('{}recon_{}_self_{}.npy'.format(self.recon_dir, self.d1_name, step), recon_d1_d1)
                np.save('{}recon_{}_cross_{}.npy'.format(self.recon_dir, self.d1_name, step), recon_d1_d2)
                np.save('{}recon_{}_joint_{}.npy'.format(self.recon_dir, self.d2_name, step), recon_d2_joint)
                np.save('{}recon_{}_self_{}.npy'.format(self.recon_dir, self.d2_name, step), recon_d2_d2)
                np.save('{}recon_{}_cross_{}.npy'.format(self.recon_dir, self.d2_name, step), recon_d2_d1)

        # np.save('{}true_{}_test_img.npy'.format(self.recon_dir, self.d1_name), self.d1_test_images[:10000])
        # np.save('{}true_{}_test_img.npy'.format(self.recon_dir, self.d2_name), self.d2_test_text[:10000])
        # np.save('{}test_img_target.npy'.format(self.recon_dir), self.test_target_label[:10000])

        with open(self.samples_dir + 'd1_joint_recon_loss.txt', 'wb') as fp:
            pickle.dump(d1_jrl_buf, fp)
        with open(self.samples_dir + 'd1_self_recon_loss.txt', 'wb') as fp:
            pickle.dump(d1_srl_buf, fp)
        with open(self.samples_dir + 'd1_cross_recon_loss.txt', 'wb') as fp:
            pickle.dump(d1_crl_buf, fp)

        with open(self.samples_dir + 'd2_joint_recon_loss.txt', 'wb') as fp:
            pickle.dump(d2_jrl_buf, fp)
        with open(self.samples_dir + 'd2_self_recon_loss.txt', 'wb') as fp:
            pickle.dump(d2_srl_buf, fp)
        with open(self.samples_dir + 'd2_cross_recon_loss.txt', 'wb') as fp:
            pickle.dump(d2_crl_buf, fp)

        with open(self.samples_dir + 'd1_cont_private_kl.txt', 'wb') as fp:
            pickle.dump(d1_pkl_buf, fp)
        with open(self.samples_dir + 'd2_cont_private_kl.txt', 'wb') as fp:
            pickle.dump(d2_pkl_buf, fp)

        with open(self.samples_dir + 'd1_classification_loss.txt', 'wb') as fp:
            pickle.dump(d1_classification_loss_buf, fp)
        with open(self.samples_dir + 'd2_classification_loss.txt', 'wb') as fp:
            pickle.dump(d2_classification_loss_buf, fp)
        with open(self.samples_dir + 'joint_classification_loss.txt', 'wb') as fp:
            pickle.dump(joint_classification_loss_buf, fp)

        with open(self.samples_dir + 'd1_cont_shared_kl.txt', 'wb') as fp:
            pickle.dump(cont_d1skl_buf, fp)
        with open(self.samples_dir + 'd2_cont_shared_kl.txt', 'wb') as fp:
            pickle.dump(cont_d2skl_buf, fp)
        with open(self.samples_dir + 'joint_cont_shared_kl.txt', 'wb') as fp:
            pickle.dump(cont_jskl_buf, fp)

        with open(self.samples_dir + 'plot_steps.txt', 'wb') as fp:
            pickle.dump(steps_buf, fp)

        return


parser = argparse.ArgumentParser()
parser.add_argument('--d1', type=str, default='shapes3d')
parser.add_argument('--d2', type=str, default='shapes3d_text')
parser.add_argument('--private_lat_flag', type=int, default=1, choices=[0, 1])
parser.add_argument('--cont_lat_flag', type=int, default=0, choices=[0, 1])
parser.add_argument('--disc_lat_flag', type=int, default=1, choices=[0, 1])
parser.add_argument('--private_z_dim1', type=int, default=10)
parser.add_argument('--cont_z_dim1', type=int, default=0)
parser.add_argument('--disc_z_dim1', type=int, default=4)
parser.add_argument('--private_z_dim2', type=int, default=1)
parser.add_argument('--cont_z_dim2', type=int, default=0)
parser.add_argument('--disc_z_dim2', type=int, default=4)
parser.add_argument('--lat_sampler', type=str, default='normal',
                    choices=['one_hot', 'uniform', 'normal', 'mix_gauss'])
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--sup_percent', type=float, default=1.0)
parser.add_argument('--sup_tr_frac', type=float, default=0.35)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_interval', type=int, default=5000)
parser.add_argument('--plot_interval', type=int, default=1000)
parser.add_argument('--training_steps', type=int, default=130001)
parser.add_argument('--expt_name', type=str, default='run')

args = parser.parse_args()

if args.private_lat_flag and args.disc_lat_flag and args.cont_lat_flag:
    if args.cont_z_dim1 == -1:
        args.cont_z_dim1 = 5
    if args.disc_z_dim1 == -1:
        args.disc_z_dim1 = 10
    if args.private_z_dim1 == -1:
        args.private_z_dim1 = 5
    if args.cont_z_dim2 == -1:
        args.cont_z_dim2 = 5
    if args.disc_z_dim2 == -1:
        args.disc_z_dim2 = 10
    if args.private_z_dim2 == -1:
        args.private_z_dim2 = 5
elif args.private_lat_flag and args.cont_lat_flag:
    if args.cont_z_dim1 == -1:
        args.cont_z_dim1 = 10
    args.disc_z_dim1 = 0
    if args.private_z_dim1 == -1:
        args.private_z_dim1 = 10
    if args.cont_z_dim2 == -1:
        args.cont_z_dim2 = 10
    args.disc_z_dim2 = 0
    if args.private_z_dim2 == -1:
        args.private_z_dim2 = 10
elif args.private_lat_flag and args.disc_lat_flag:
    args.cont_z_dim1 = 0
    if args.disc_z_dim1 == -1:
        args.disc_z_dim1 = 10
    if args.private_z_dim1 == -1:
        args.private_z_dim1 = 10
    args.cont_z_dim2 = 0
    if args.disc_z_dim2 == -1:
        args.disc_z_dim2 = 10
    if args.private_z_dim2 == -1:
        args.private_z_dim2 = 2
elif args.cont_lat_flag and args.disc_lat_flag:
    args.private_z_dim1 = 0
    if args.cont_z_dim1 == -1:
        args.cont_z_dim1 = 10
    if args.disc_z_dim1 == -1:
        args.disc_z_dim1 = 10
    args.private_z_dim2 = 0
    if args.cont_z_dim2 == -1:
        args.cont_z_dim2 = 10
    if args.disc_z_dim2 == -1:
        args.disc_z_dim2 = 10
elif args.cont_lat_flag:
    args.private_z_dim1 = 0
    if args.cont_z_dim1 == -1:
        args.cont_z_dim1 = 20
    args.disc_z_dim1 = 0
    args.private_z_dim2 = 0
    if args.cont_z_dim2 == -1:
        args.cont_z_dim2 = 20
    args.disc_z_dim2 = 0

prefix = f'./{args.expt_name}_sup{args.sup_percent}_priv{args.private_lat_flag}_cont{args.cont_lat_flag}_disc' \
         f'{args.disc_lat_flag}/'
trained_models_dir = f'{prefix}{args.d1}_{args.d2}/Models/'
training_data_dir = f'{prefix}{args.d1}_{args.d2}/Samples/'
reconstructed_data_dir = f'{prefix}{args.d1}_{args.d2}/Reconstructed/'
generated_data_dir = f'{prefix}{args.d1}_{args.d2}/Generated/'
create_directory(trained_models_dir)
create_directory(training_data_dir)
create_directory(reconstructed_data_dir)
create_directory(generated_data_dir)

with open(f'{prefix}config_{args.d1}_{args.d2}.txt', 'w') as f:
    f.write(f'Private Latent: {args.private_lat_flag}, Continuous Latent: {args.cont_lat_flag}, Discrete Latent:'
            f' {args.disc_lat_flag}\n')
    f.write(
        f'Private Latent Dim: {args.private_z_dim1},{args.private_z_dim2}, Shared Continuous Latent Dim: {args.cont_z_dim1},{args.cont_z_dim2}, '
        f'Shared Discrete Latent Dim: {args.disc_z_dim1},{args.disc_z_dim2}\n'
        f'LR: {args.lr}, Supervision Threshold: {args.sup_percent}, Batch Size: {args.batch_size}\n')
    f.write(f'Private Cap Min: {private_min}, Private Cap Max: {private_max}, Private Iterations: '
            f'{private_num_iters}, Private Gamma: {private_gamma}\n')
    f.write(f'Discrete Cap Min: {disc_min}, Discrete Cap Max: {disc_max}, Discrete Iterations: '
            f'{disc_num_iters}, Discrete Gamma: {disc_gamma}\n')
    f.write(f'Continuous Cap Min: {cont_min}, Continuous Cap Max: {cont_max}, Continuous Iterations: '
            f'{cont_num_iters}, Continuous Gamma: {cont_gamma}\n')
    f.write(f'Discrete Logit Loss Gamma: {dll_gamma}, Temperature: {tau}\n')

tf.compat.v1.reset_default_graph()
model = SSDMMVAE(d1_name=args.d1, d2_name=args.d2, bs=args.batch_size, private_lat_flag=args.private_lat_flag,
                 cont_lat_flag=args.cont_lat_flag, disc_lat_flag=args.disc_lat_flag, cont_z_dim1=args.cont_z_dim1,
                 disc_z_dim1=args.disc_z_dim1, private_z_dim1=args.private_z_dim1, cont_z_dim2=args.cont_z_dim2,
                 disc_z_dim2=args.disc_z_dim2, private_z_dim2=args.private_z_dim2, lat_sampler=args.lat_sampler,
                 lr=args.lr, supervision_percent=args.sup_percent, sup_tr_frac=args.sup_tr_frac,
                 models_dir=trained_models_dir, samples_dir=training_data_dir,
                 gen_dir=generated_data_dir, recon_dir=reconstructed_data_dir, training_steps=args.training_steps,
                 save_interval=args.save_interval, plot_interval=args.plot_interval
                )
model.train()
