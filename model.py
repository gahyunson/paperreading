import tensorflow as tf
from keras.applications import vgg16
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization
from keras import Model
from keras import callbacks
from keras import optimizers

class VGG16(tf.keras.Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
        for layer in self.base_model.layers:
          layer.trainable = False
        self.vgg_layer = self.base_model.get_layer('block5_pool')
        
    def call(self, inputs):
        x = self.vgg_layer(inputs)
        return x

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.NUM_CLASSES = 100
    self.LEARNING_RATE = 1e-3
    self.LOSS = "categorical_crossentropy"
    self.OPTIMIZER = optimizers.Adam(lr=self.LEARNING_RATE)
    self.METRICS = ['accuracy']
    
    self.vgg_layer = VGG16()
    self.pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.dense1 = tf.keras.layers.Dense(256, activation='relu')
    self.dense2 = tf.keras.layers.Dense(256, activation='relu')
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.class_layer = tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax')
    
  def call(self, inputs):
    x = self.vgg_layer(inputs)
    x = self.pooling_layer(x)
    x = self.batch_norm(x)
    x = self.dense1(x)
    x = self.dropout(x)
    x = self.dense2(x)
    output = self.class_layer(x)
    return output

def model_compile():
    model = MyModel()   
    model.compile(loss=MyModel.LOSS,
                optimizer=MyModel.OPTIMIZER,
                metrics=MyModel.METRICS)
    return model
