import tensorflow as tf
from tensorflow.keras.layers import Lambda, Dense, Flatten, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

from numpy import load
import h5py



#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


data = load('Vehicles_training_features_newer.npz')
training_features = data['arr_0'][0]
data = load('Vehicles_training_labels_newer.npz')
training_labels = data['arr_0'][0]
training_labels = tf.keras.utils.to_categorical(training_labels, 8)

vgg = VGG16(input_tensor=Input(shape=(60,60,3)), include_top=False,input_shape=(len(training_features),60,60,3)[1:])
for layer in vgg.layers:
    layer.trainable = False
flatten_layer = Flatten()
x = flatten_layer(vgg.output)

prediction = Dense(8, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)
print(model)
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit([training_features], training_labels, batch_size=8, epochs=5, validation_split=0.1)
model.save('VehicleClassifierVGG16New.h5')

