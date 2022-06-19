import argparse
from email.mime import image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import random


SEED = 1237
DPI = None
POWER_ITERATIONS = 5
HEADER = '\033[95m'
OK_BLUE = '\033[94m'
OK_GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END_C = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

digit_text_english = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
shapes3d_text_english = ['cube', 'cylinder', 'sphere', 'capsule']
mpi3d_toy_text_english = ['cone', 'cube', 'cylinder', 'hexagonal', 'pyramid', 'sphere']
smallnorb_text_english = ['animal', 'human', 'airplane', 'truck', 'car']
shape_text = ['square', 'ellipse', 'heart']


def char2Index(alphabet, character):
    return alphabet.find(character)


def text_one_hot_encode(seq, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]',
                        len_seq=9):
    X = np.zeros((len_seq, len(alphabet)))
    if len(seq) > len_seq:
        seq = seq[:len_seq]
    for index_char, char in enumerate(seq):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
    return X


def text_one_hot_decode(X, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\\/|_@#$%^&*~`+-=<>()[]',
                        len_seq=9):
    idxs = np.nonzero(X == 1)[1]
    text = []
    for elt in idxs:
        text.append(alphabet[elt])
    return text


def dsprite_gen_cond(batch_size, shape=1):
    npz_path = './data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    dataset_zip = np.load(npz_path, encoding='latin1')
    imgs = dataset_zip['imgs']
    imgs = imgs[shape * 245760:(shape + 1) * 245760]
    im_shape = imgs.shape
    imgs = np.reshape(imgs, (im_shape[0], im_shape[1], im_shape[2], 1))
    imgs = imgs.astype(float)
    np.random.shuffle(imgs)
    ind_1 = 0
    while (1):
        yield imgs[ind_1 * batch_size:(ind_1 + 1) * batch_size]
        ind_1 += 1
        if ind_1 * batch_size >= im_shape[0]:
            ind_1 = 0
            np.random.shuffle(imgs)


def get_dsprite_dataset(split=0.8):
    npz_path = './data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    dataset_zip = np.load(npz_path, encoding='latin1')
    imgs = dataset_zip['imgs']
    del dataset_zip
    num_examples = 737280
    np.random.seed(0)
    indices = np.random.permutation(num_examples)
    part = int(split * num_examples)
    training_idx, test_idx = indices[:part], indices[part:]
    train_imgs = np.expand_dims(imgs[training_idx, :], axis=-1)
    test_imgs = np.expand_dims(imgs[test_idx, :], axis=-1)
    del imgs
    labels = np.array([0] * 245760 + [1] * 245760 + [2] * 245760)
    train_labels = labels[training_idx]
    test_labels = labels[test_idx]
    del labels
    np.random.seed(None)
    return train_imgs, test_imgs, train_labels, test_labels


def get_dsprite_text_dataset(len_seq, split=0.8):
    num_examples = 737280
    labels = np.array([0] * 245760 + [1] * 245760 + [2] * 245760)
    np.random.seed(0)
    indices = np.random.permutation(num_examples)
    part = int(split * num_examples)
    training_idx, test_idx = indices[:part], indices[part:]
    text = create_text_arr_from_label_dsprite(len_seq, labels)
    train_text = text[training_idx, :]
    test_text = text[test_idx, :]
    train_labels = labels[training_idx]
    test_labels = labels[test_idx]
    np.random.seed(None)
    return train_text, test_text, train_labels, test_labels


def get_mpi3d_toy_dataset(split=0.8):
    npz_path = './data/mpi3d_toy.npz'
    dataset_zip = np.load(npz_path)
    imgs = dataset_zip['images']
    del dataset_zip
    num_examples = imgs.shape[0]
    np.random.seed(0)
    indices = np.random.permutation(num_examples)
    part = int(split * num_examples)
    training_idx, test_idx = indices[:part], indices[part:]
    train_imgs = imgs[training_idx]
    test_imgs = imgs[test_idx]
    # labels = imgs.reshape([6,6,2,3,3,40,40,64,64,3])
    del imgs
    labels = np.array(([0] * (num_examples // 36) + 
                      [1] * (num_examples // 36) + 
                      [2] * (num_examples // 36) + 
                      [3] * (num_examples // 36) + 
                      [4] * (num_examples // 36) + 
                      [5] * (num_examples // 36)) * 6)
    train_labels = labels[training_idx]
    test_labels = labels[test_idx]
    del labels
    np.random.seed(None)
    return train_imgs, test_imgs, train_labels, test_labels


def get_mpi3d_toy_text_dataset(len_seq, split=0.8):
    npz_path = './data/mpi3d_toy.npz'
    dataset_zip = np.load(npz_path)
    imgs = dataset_zip['images']
    del dataset_zip
    num_examples = imgs.shape[0]
    del imgs
    labels = np.array(([0] * (num_examples // 36) + 
                      [1] * (num_examples // 36) + 
                      [2] * (num_examples // 36) + 
                      [3] * (num_examples // 36) + 
                      [4] * (num_examples // 36) + 
                      [5] * (num_examples // 36)) * 6)
    np.random.seed(0)
    indices = np.random.permutation(num_examples)
    part = int(split * num_examples)
    training_idx, test_idx = indices[:part], indices[part:]
    text = create_text_arr_from_label_mpi3d_toy(len_seq, labels)
    train_text = text[training_idx, :]
    test_text = text[test_idx, :]
    train_labels = labels[training_idx]
    test_labels = labels[test_idx]
    np.random.seed(None)
    return train_text, test_text, train_labels, test_labels


def create_text_from_label_mnist(
        len_seq, label, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text = digit_text_english[label]
    sequence = len_seq * [' ']
    start_index = random.randint(0, len_seq - 1 - len(text))
    sequence[start_index:start_index + len(text)] = text
    sequence_one_hot = text_one_hot_encode(sequence, alphabet, len_seq)
    return sequence_one_hot


def create_text_from_label_shapes3d(
        len_seq, label, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text = shapes3d_text_english[label]
    sequence = len_seq * [' ']
    start_index = random.randint(0, len_seq - 1 - len(text))
    sequence[start_index:start_index + len(text)] = text
    sequence_one_hot = text_one_hot_encode(sequence, alphabet, len_seq)
    return sequence_one_hot


def create_text_from_label_mpi3d_toy(
        len_seq, label, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text = mpi3d_toy_text_english[label]
    sequence = len_seq * [' ']
    start_index = random.randint(0, len_seq - 1 - len(text))
    sequence[start_index:start_index + len(text)] = text
    sequence_one_hot = text_one_hot_encode(sequence, alphabet, len_seq)
    return sequence_one_hot


def create_text_from_label_smallnorb(
        len_seq, label, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text = smallnorb_text_english[label]
    sequence = len_seq * [' ']
    start_index = random.randint(0, len_seq - 1 - len(text))
    sequence[start_index:start_index + len(text)] = text
    sequence_one_hot = text_one_hot_encode(sequence, alphabet, len_seq)
    return sequence_one_hot


def create_text_from_label_dsprite(
        len_seq, label, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text = shape_text[label]
    sequence = len_seq * [' ']
    start_index = random.randint(0, len_seq - 1 - len(text))
    sequence[start_index:start_index + len(text)] = text
    sequence_one_hot = text_one_hot_encode(sequence, alphabet, len_seq)
    return sequence_one_hot


def create_text_arr_from_label_mnist(
        len_seq, label_arr, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text_arr = []
    for elt in list(label_arr):
        text_arr.append(np.expand_dims(create_text_from_label_mnist(len_seq=len_seq, label=elt, alphabet=alphabet),
                                       axis=0))
    return np.concatenate(text_arr, axis=0)


def create_text_arr_from_label_shapes3d(
        len_seq, label_arr, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text_arr = []
    for elt in list(label_arr):
        text_arr.append(np.expand_dims(create_text_from_label_shapes3d(len_seq=len_seq, label=elt, alphabet=alphabet),
                                       axis=0))
    return np.concatenate(text_arr, axis=0)


def create_text_arr_from_label_smallnorb(
        len_seq, label_arr, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text_arr = []
    for elt in list(label_arr):
        text_arr.append(np.expand_dims(create_text_from_label_smallnorb(len_seq=len_seq, label=elt, alphabet=alphabet),
                                       axis=0))
    return np.concatenate(text_arr, axis=0)


def create_text_arr_from_label_mpi3d_toy(
        len_seq, label_arr, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text_arr = []
    for elt in list(label_arr):
        text_arr.append(np.expand_dims(create_text_from_label_mpi3d_toy(len_seq=len_seq, label=elt, alphabet=alphabet),
                                       axis=0))
    return np.concatenate(text_arr, axis=0)


def create_text_arr_from_label_dsprite(
        len_seq, label_arr, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    text_arr = []
    for elt in list(label_arr):
        text_arr.append(np.expand_dims(create_text_from_label_dsprite(len_seq=len_seq, label=elt, alphabet=alphabet),
                                       axis=0))
    return np.concatenate(text_arr, axis=0)


def seq2text(seq, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    decoded = []
    for j in range(len(seq)):
        decoded.append(alphabet[seq[j]])
    return decoded


def tensor_to_text(gen_t, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=-1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(gen_t[i], alphabet)
        decoded_samples.append(decoded)
    return decoded_samples


def numpy_to_text(gen_t, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    gen_t = np.argmax(gen_t, axis=-1)
    # print(gen_t)
    decoded = seq2text(gen_t, alphabet)
    return decoded


def get_text(one_hot_text, alphabet=' {}abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"\/|_@#$%^&*~`+-=<>()[]'):
    idx = np.nonzero(np.array(one_hot_text) == 1)
    return alphabet[idx]


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def image_grid(images, texts, height, width, n_row, n_col, sv_path, dpi, is_text1, is_text2):
    # images = (images - np.min(images)) / (np.max(images) - np.min(images))
    # fig = plt.gcf()
    # fig.set_size_inches(width, height)
    f, axarr = plt.subplots(n_row, n_col, figsize=(width, height))
    for i in range(n_col):
        if images is not None:
            if is_text1 == False:
                image = images[i]
                if image.shape[-1] == 1:
                    image = np.reshape(image, [image.shape[0], image.shape[1]])
                    axarr[0, i].imshow(image, cmap='gray')
                elif image.shape[-1] == 3:
                    axarr[0, i].imshow(image)
                axarr[0, i].set_axis_off()
            else:
                text = numpy_to_text(images[i])
                text = ''.join(text)
                # print(text)
                axarr[0, i].text(0.5, 0.5, text, horizontalalignment='center',
                                 verticalalignment='center')
                axarr[0, i].set_axis_off()

        if texts is not None:
            if is_text2 == False:
                image = texts[i]
                if image.shape[-1] == 1:
                    image = np.reshape(image, [image.shape[0], image.shape[1]])
                    axarr[1, i].imshow(image, cmap='gray')
                elif image.shape[-1] == 3:
                    axarr[1, i].imshow(image)
                axarr[1, i].set_axis_off()
            else:
                text = numpy_to_text(texts[i])
                text = ''.join(text)
                axarr[1, i].text(0.5, 0.5, text, horizontalalignment='center',
                                 verticalalignment='center')
                axarr[1, i].set_axis_off()
    plt.savefig(sv_path, dpi=dpi)
    plt.close('all')
    return


def image_grid_text(texts, height, width, n_row, n_col, sv_path, dpi):
    f, axarr = plt.subplots(n_row, n_col, figsize=(width, height))
    for row in range(n_row):
        for col in range(n_col):
            text = numpy_to_text(texts[row * n_col + col])
            text = ''.join(text)
            axarr[row, col].text(0.5, 0.5, text, horizontalalignment='center',
                          verticalalignment='center')
            # axarr[i].set_axis_off()
            axarr[row, col].grid(True, which='both', axis='x', color='gray', linestyle='-', linewidth=0.5)
            axarr[row, col].set_xticklabels([])
            axarr[row, col].set_yticklabels([])
            axarr[row, col].set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            axarr[row, col].tick_params(axis='x', colors='white')
            axarr[row, col].set_yticks([])
    plt.savefig(sv_path, dpi=dpi)
    plt.close('all')
    return


def image_grid_orig(images, height, width, n_row, n_col, sv_path, dpi):
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        image = images[i]
        if image.shape[-1] == 1:
            image = np.reshape(image, [image.shape[0], image.shape[1]])
            plt.imshow(image, cmap='gray')
        elif image.shape[-1] == 3:
            plt.imshow(image)
        plt.axis('off')
    # fig.subplots_adjust(wspace=0.07, hspace=0.02)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(sv_path, dpi=dpi)
    plt.close('all')
    return


def plot_graph(x, y, x_label, y_label, samples_dir, img_name):
    plt.close('all')
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(b=True, which='both')
    plt.savefig(samples_dir + img_name, dpi=DPI)


def to_one_hot(label_index, num_class):
    return np.eye(num_class)[label_index]


def get_text_arr(num_arr):
    num_dict = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight',
                9: 'Nine'}
    text_arr = np.array([num_dict[elt] for elt in list(num_arr)])
    return text_arr


def load_data(data_name):
    if data_name == 'mnist':
        tfds_train, tfds_test = tfds.load('mnist', split=['train', 'test'], batch_size=-1, as_supervised=True)
        train_images, train_labels = tfds.as_numpy(tfds_train)
        test_images, test_labels = tfds.as_numpy(tfds_test)
        train_images = np.array(train_images).reshape(60000, 28, 28, 1) / np.float32(255.0)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images).reshape(10000, 28, 28, 1) / np.float32(255.0)
        test_labels = np.array(test_labels)
        side = 28
        channels = 1
    elif data_name == 'svhn':
        tfds_train, tfds_test = tfds.load('svhn_cropped', split=['train', 'test'], batch_size=-1, as_supervised=True)
        train_images, train_labels = tfds.as_numpy(tfds_train)
        test_images, test_labels = tfds.as_numpy(tfds_test)
        train_images = np.array(train_images) / np.float32(255.0)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images) / np.float32(255.0)
        test_labels = np.array(test_labels)
        side = 32
        channels = 3
    elif data_name == 'mad_base':
        train_images = pd.read_csv(
            "./Arabic Handwritten Digits Dataset CSV/csvTrainImages 60k x 784.csv",
            header=None).values.reshape(60000, 28, 28, 1)
        train_images = train_images / np.float32(255.0)
        train_labels = pd.read_csv(
            "./Arabic Handwritten Digits Dataset CSV/csvTrainLabel 60k x 1.csv", header=None).values.reshape(60000, )
        test_images = pd.read_csv(
            "./Arabic Handwritten Digits Dataset CSV/csvTestImages 10k x 784.csv",
            header=None).values.reshape(10000, 28, 28, 1)
        test_images = test_images / np.float32(255.0)
        test_labels = pd.read_csv(
            "./Arabic Handwritten Digits Dataset CSV/csvTestLabel 10k x 1.csv", header=None).values.reshape(10000, )
        side = 28
        channels = 1
    elif data_name == 'visda17_syn':
        x = np.load('./syn2real/train_images.npy') / np.float32(255.0)
        np.random.seed(SEED)
        np.random.shuffle(x)
        y = np.load('./syn2real/train_labels.npy')
        np.random.seed(SEED)
        np.random.shuffle(y)
        train_images = x[:-10000]
        train_labels = y[:-10000]
        test_images = x[-10000:]
        test_labels = y[-10000:]
        side = 64
        channels = 3
    elif data_name == 'visda17_real':
        x = np.load('./syn2real/val_images.npy') / np.float32(255.0)
        np.random.seed(SEED)
        np.random.shuffle(x)
        y = np.load('./syn2real/val_labels.npy')
        np.random.seed(SEED)
        np.random.shuffle(y)
        train_images = x[:-10000]
        train_labels = y[:-10000]
        test_images = x[-10000:]
        test_labels = y[-10000:]
        side = 64
        channels = 3
    elif data_name == 'mnist_text':
        tfds_train, tfds_test = tfds.load('mnist', split=['train', 'test'], batch_size=-1, as_supervised=True)
        _, train_labels = tfds.as_numpy(tfds_train)
        _, test_labels = tfds.as_numpy(tfds_test)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        train_images = create_text_arr_from_label_mnist(8, train_labels)
        test_images = create_text_arr_from_label_mnist(8, test_labels)
        side = 28
        channels = 1
    elif data_name == 'dsprite':
        train_images, test_images, train_labels, test_labels = get_dsprite_dataset(0.8)
        side = 64
        channels = 1
    elif data_name == 'dsprite_text':
        train_images, test_images, train_labels, test_labels = get_dsprite_text_dataset(8, 0.8)
        side = 64
        channels = 1
    
    elif data_name == 'mpi3d_toy':
        train_images, test_images, train_labels, test_labels = get_mpi3d_toy_dataset(0.8)
        train_images = train_images / np.float32(255.0)
        test_images = test_images / np.float32(255.0)
        side = 64
        channels = 3
    elif data_name == 'mpi3d_toy_text':
        train_images, test_images, train_labels, test_labels = get_mpi3d_toy_text_dataset(10, 0.8)
        side = 64
        channels = 3
    elif data_name == 'shapes3d':
        tfds_train = tfds.load('shapes3d', split=['train'], batch_size=-1)
        split = 0.8
        num_examples = 480000
        np.random.seed(0)
        indices = np.random.permutation(num_examples)
        part = int(split * num_examples)
        training_idx, test_idx = indices[:part], indices[part:]
        images = tfds.as_numpy(tfds_train)[0]['image']
        labels = tfds.as_numpy(tfds_train)[0]['label_shape']
        train_images, train_labels = images[training_idx], labels[training_idx]
        test_images, test_labels = images[test_idx], labels[test_idx]
        train_images = train_images / np.float32(255.0)
        test_images = test_images / np.float32(255.0)
        side = 64
        channels = 3
        del images
        del labels
        np.random.seed(None)
    elif data_name == 'shapes3d_text':
        tfds_train = tfds.load('shapes3d', split=['train'], batch_size=-1)
        split = 0.8
        num_examples = 480000
        np.random.seed(0)
        indices = np.random.permutation(num_examples)
        part = int(split * num_examples)
        training_idx, test_idx = indices[:part], indices[part:]
        labels = tfds.as_numpy(tfds_train)[0]['label_shape']
        train_labels = labels[training_idx]
        test_labels = labels[test_idx]
        side = 64
        channels = 3
        del labels
        np.random.seed(None)
        train_images = create_text_arr_from_label_shapes3d(10, train_labels)
        test_images = create_text_arr_from_label_shapes3d(10, test_labels)
    elif data_name == 'smallnorb':
        tfds_train, tfds_test = tfds.load('smallnorb', split=['train', 'test'], batch_size=-1)
        train_images = tfds.as_numpy(tfds_train)['image']
        train_labels = tfds.as_numpy(tfds_train)['label_category']
        test_images = tfds.as_numpy(tfds_test)['image']
        test_labels = tfds.as_numpy(tfds_test)['label_category']
        train_images = tf.image.resize(train_images, [64, 64]).numpy() / np.float32(255.0)
        test_images = tf.image.resize(test_images, [64, 64]).numpy() / np.float32(255.0)
        side = 64
        channels = 1
    elif data_name == 'smallnorb_text':
        tfds_train, tfds_test = tfds.load('smallnorb', split=['train', 'test'], batch_size=-1)
        train_labels = tfds.as_numpy(tfds_train)['label_category']
        test_labels = tfds.as_numpy(tfds_test)['label_category']
        train_images = create_text_arr_from_label_smallnorb(10, train_labels)
        test_images = create_text_arr_from_label_smallnorb(10, test_labels)
        side = 64
        channels = 1
    else:
        raise Exception('Define load_data for dataset `{}`.'.format(data_name))
    np.random.seed(None)
    return train_images, train_labels, test_images, test_labels, side, channels


def get_paired_sample(d1_images, d2_images, batch_size):
    num_images = d1_images.shape[0]
    ridx = np.random.randint(low=0, high=num_images, size=batch_size)
    return d1_images[ridx].astype(np.float32), d2_images[ridx].astype(np.float32)


def get_paired_sample_with_labels(d1_images, d2_images, target_label, batch_size):
    num_images = d1_images.shape[0]
    ridx = np.random.randint(low=0, high=num_images, size=batch_size)
    return d1_images[ridx].astype(np.float32), d2_images[ridx].astype(np.float32), target_label[ridx].astype(np.int)


def get_random_sample(images, labels, batch_size, n_class=10):
    ridx = np.random.randint(low=0, high=len(labels), size=batch_size)
    return images[ridx], to_one_hot(labels[ridx], n_class)


def sample_z(batch, z_dim, sampler='one_hot', num_class=10, label_index=None):
    if sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return np.hstack((np.random.randn(batch, z_dim), np.eye(num_class)[label_index]))
    elif sampler == 'uniform':
        return np.random.uniform(-1., 1., size=[batch, z_dim])
    elif sampler == 'normal':
        return np.random.randn(batch, z_dim)
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index]


def sample_categorical(batch, z_dim):
    label_index = np.random.randint(low=0, high=z_dim, size=batch)
    return np.eye(z_dim)[label_index]


def ssim_loss(target, predicted):
    return - tf.reduce_mean(tf.image.ssim(target, predicted, max_val=1.0))


def kld_loss(mu, log_var):
    return tf.reduce_mean(tf.square(mu) + tf.math.exp(log_var) - log_var - 1) / 2.0


def kl_discrete_loss(probs, eps=1e-20):
    k = tf.cast(tf.shape(probs)[-1], tf.float32)
    kl_loss = tf.reduce_mean(
        tf.reduce_sum(probs * (tf.math.log(probs + eps) - tf.math.log(1.0 / k)), [1]))
    return kl_loss
