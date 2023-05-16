from model import VggCustom
from data_preprocess import DataLoader
from keras import optimizers

EPOCHS = 1
BATCH_SIZE = 32
NUM_CLASSES = 100
LEARNING_RATE = 1e-2

data_loader = DataLoader()
train_generator, val_generator, test_generator = data_loader.call()
train_len, val_len, _, data_dim = data_loader.data_len()
# x_train, x_val, x_test, y_train, y_val, y_test = data_loader.load_cifar()

model = VggCustom(NUM_CLASSES)
for layer in model.base_model.layers[:8]:
    layer.trainable = False
model.build(input_shape=(None, 32, 32, 3))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])

model.fit(train_generator, epochs=EPOCHS, batch_size=BATCH_SIZE)
model.evaluate(test_generator, batch_size=BATCH_SIZE)

model.save('model.h5')