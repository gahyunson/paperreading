import tensorflow as tf
from keras.applications import vgg16
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization
from keras import Model
from keras import callbacks
from keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import vgg16 as vgg

class VggCustom(tf.keras.Model):
    def __init__(self, num_classes=100):
        super().__init__()

        # VGG16 base model
        self.base_model = vgg.VGG16(weights='imagenet', include_top=False,
                                    pooling='avg')

        # Classification layers
        # self.global_average_pooling2d = tf.keras.layers.GlobalAveragePooling2D()
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.predictions = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        # x = self.global_average_pooling2d(x)
        x = self.batch_normalization(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.predictions(x)
