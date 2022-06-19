import tensorflow as tf


class ImageClassifier(tf.keras.Model):
    def __init__(self, n_class):
        super(ImageClassifier, self).__init__()
        self.n_class = n_class
        self.lam = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype=tf.float32))
        self.conv1 = tf.keras.layers.Conv2D(32, 5)
        self.conv2 = tf.keras.layers.Conv2D(64, 5)
        self.drop = tf.keras.layers.Dropout(0.2)
        self.flat = tf.keras.layers.Flatten()
        self.linear1 = tf.keras.layers.Dense(50)
        self.linear2 = tf.keras.layers.Dense(self.n_class, activation='softmax')

    def call(self, x, training=True):
        x = self.conv1(self.lam(x))
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.activations.relu(x)
        x = self.drop(x, training)
        x = self.conv2(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.activations.relu(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = tf.keras.activations.relu(x)
        x = self.drop(x, training)
        x = self.linear2(x)
        return x


class TextClassifier(tf.keras.Model):
    def __init__(self, n_class):
        super(TextClassifier, self).__init__()
        self.n_class = n_class
        self.conv1 = tf.keras.layers.Conv1D(32, 2)
        self.conv2 = tf.keras.layers.Conv1D(64, 2)
        self.drop = tf.keras.layers.Dropout(0.2)
        self.flat = tf.keras.layers.Flatten()
        self.linear1 = tf.keras.layers.Dense(50)
        self.linear2 = tf.keras.layers.Dense(self.n_class, activation='softmax')

    def call(self, x, training=True):
        x = self.conv1(x)
        x = tf.keras.layers.MaxPool1D()(x)
        x = tf.keras.activations.relu(x)
        x = self.drop(x, training)
        x = self.conv2(x)
        x = tf.keras.layers.MaxPool1D()(x)
        x = tf.keras.activations.relu(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = tf.keras.activations.relu(x)
        x = self.drop(x, training)
        x = self.linear2(x)
        return x
