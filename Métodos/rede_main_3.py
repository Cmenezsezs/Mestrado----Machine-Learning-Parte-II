import tensorflow as tf


class RedeTop(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation=None)
        self.dense2 = tf.keras.layers.Dense(64, activation=None)
        self.dense3 = tf.keras.layers.Dense(32, activation=None)
        self.dense4 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x
