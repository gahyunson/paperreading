import tensorflow as tf 
from data_preprocess import DataLoader
import numpy as np

data_loader = DataLoader()
_, _, test_generator = data_loader.call()

model = tf.keras.models.load_model('model.h5')

predict = model.predict()
label = np.argmax(predict, axis=1)
