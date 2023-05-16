from model import VggCustom
from data_preprocess import DataLoader
from keras import optimizers
from keras import callbacks

EPOCHS = 1
BATCH_SIZE = 32
NUM_CLASSES = 100
LEARNING_RATE = 1e-2

data_loader = DataLoader()
train_generator, val_generator, test_generator = data_loader.call()
# train_len, val_len, _, data_dim = data_loader.data_len()
# x_train, x_val, x_test, y_train, y_val, y_test = data_loader.load_cifar()

lr_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=5, mode='min', verbose=1)
early_stop = callbacks.EarlyStopping(monitor="val_loss",
                                           min_delta=0,
                                           patience=5,
                                           restore_best_weights=True)


model = VggCustom(NUM_CLASSES)
for layer in model.base_model.layers[:8]:
    layer.trainable = False
model.build(input_shape=(None, 32, 32, 3))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=['accuracy'])

model.fit(train_generator, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=val_generator,
          callbacks=[lr_cb, early_stop],)
model.evaluate(test_generator, batch_size=BATCH_SIZE)

model.save('model.h5')