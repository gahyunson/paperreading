
# class VGG16(Model):
#     def __init__(self, num_classes=100):
#         super(VGG16, self).__init__()
#         self.base_model = vgg16.VGG16(weights='imagenet', include_top=False)

#     def call(self, inputs):
#         x = self.base_model(inputs)
#         return x

# vgg = VGG16()
# vgg.build(input_shape=(None, 32, 32, 3))


# class MyModel(tf.keras.Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.NUM_CLASSES = 100
#     self.LEARNING_RATE = 1e-3
#     self.LOSS = "categorical_crossentropy"
#     self.OPTIMIZER = optimizers.Adam(learning_rate=self.LEARNING_RATE)
#     self.METRICS = ['accuracy']
    
#     self.vgg_layer = VGG16().call()
#     # self.pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
#     self.batch_norm = tf.keras.layers.BatchNormalization()
#     self.dense1 = tf.keras.layers.Dense(256, activation='relu')
#     self.dense2 = tf.keras.layers.Dense(256, activation='relu')
#     self.dropout = tf.keras.layers.Dropout(0.5)
#     self.class_layer = tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax')
    
#   def call(self, inputs):
#     x = self.vgg_layer(inputs)
#     x = self.batch_norm(x)
#     x = self.dense1(x)
#     x = self.dropout(x)
#     x = self.dense2(x)
#     # model = Model(self.vgg_model.base_model.input, output)
#     return self.class_layer(x)

# model = MyModel()
# model.build(input_shape=(None, 32, 32, 3))
# print(model.summary())
  
# def model_compile():
#   mymodel = MyModel()
#   model = mymodel.call()
#   model.build((None, 32, 32, 3))
#   model.compile(loss=MyModel().LOSS,
#                 optimizer=MyModel().OPTIMIZER,
#                 metrics=MyModel().METRICS)
#   return model
