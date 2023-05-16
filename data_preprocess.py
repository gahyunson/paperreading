from keras.datasets import cifar100
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self):
        self.batch_size = 32
        self.NUM_CLASSES = 100
    
    def load_cifar(self):
        (x_data, y_data), (x_test, y_test) = cifar100.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=42)
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def data_len(self):
        x_train, x_val, x_test, _, _, _ = self.load_cifar()
        train_len = x_train.shape[0]
        val_len = x_val.shape[0]
        test_len = x_test.shape[0]
        data_dim = x_train.shape[2]
        return train_len, val_len, test_len, data_dim
    
    def call(self):
        x_train, x_val, x_test, y_train, y_val, y_test = self.load_cifar()
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.NUM_CLASSES)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=self.NUM_CLASSES)
        
        train_scale = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=False
        )
        train_scale.fit(x_train)
        train_generator = train_scale.flow(x_train, y_train,
                                           batch_size=self.batch_size)
        
        val_scale = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=False
        )
        val_scale.fit(x_val)
        val_generator = val_scale.flow(
            x_val, y_val,
            batch_size=self.batch_size
        )
        
        test_scale = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=False
        )
        test_scale.fit(x_test)
        test_generator = test_scale.flow(x_test, y_test,
                                       batch_size=self.batch_size)
                
        return train_generator, val_generator, test_generator
    
# data_loader = DataLoader()

    # def get_train_data(self):
    #     train_generator, _, _ = self.load_data()
    #     train_ds = tf.data.Dataset.from_tensor_slices(train_generator)
    #     train_ds = train_ds.shuffle(10000).batch(self.batch_size)
    #     return train_ds
    
    # def get_val_data(self):
    #     _, val_generator, _ = self.load_data()
    #     val_ds = tf.data.Dataset.from_tensor_slices(val_generator)
    #     val_ds = val_ds.shuffle(10000).batch(self.batch_size)
    #     return val_ds
    
    # def get_test_data(self):
    #     _, _, x_test, y_test = self.load_data()
    #     test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    #     test_ds = test_ds.batch(self.batch_size)
    #     return test_ds
    