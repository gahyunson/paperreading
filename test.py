from keras.applications import vgg16
from keras import Model
from keras.models import Model

class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.base_model = vgg16.VGG16(weights='imagenet', include_top=False)

    def call(self, inputs):
        x = self.base_model(inputs)
        return x

vgg = VGG16()
vgg.build(input_shape=(None, 32, 32, 3))
print(vgg.summary())