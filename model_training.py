import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

base_dir = 'C:\\Users\\gabri\\Desktop\\Code\\NUT-Carcinoma-Classification\\NUT-Carcinoma-Pathology-Classification\\NMC Pathology'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 2, class_mode = 'binary', target_size = (224, 224))
validation_generator = test_datagen.flow_from_directory( validation_dir, batch_size = 2, class_mode = 'binary', target_size = (224, 224))

model = tf.keras.models.Sequential([
      ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='max'),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data = validation_generator, epochs=8, verbose=1)

model.save('resnet_model')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()