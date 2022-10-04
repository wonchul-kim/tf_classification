import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers


### configurate GPUs settings
physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_device, True)
    except:
        pass

### logs    
output_dir = './results'
if not osp.exists(output_dir):
    os.mkdir(output_dir)
vis_results = osp.join(output_dir, 'figs')
if not osp.exists(vis_results):
    os.mkdir(vis_results)
ckpt_results = osp.join(output_dir, 'checkpoints')
if not osp.exists(ckpt_results):
    os.mkdir(ckpt_results)

### define dataset & dataloader
train_dir = '/home/wonchul/HDD/datasets/rps/train'
val_dir = '/home/wonchul/HDD/datasets/rps/test'
test_dir = '/home/wonchul/HDD/datasets/rps/test'

width = 150
height = 150
input_shape = (height, width, 3)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width), ### resize to
        batch_size=batch_size,
        class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')

print(f"* class information: {train_generator.class_indices}")

NUM_TRAIN = sum([len(files) for r, d, files in os.walk(train_dir)])
NUM_TEST = sum([len(files) for r, d, files in os.walk(val_dir)])

### training params.
epochs = 20

### Define model && strategy for multi-gpu
strategy = tf.distribute.MirroredStrategy()
print('* Number of devices: {}'.format(strategy.num_replicas_in_sync))

### define MODEL
TF_APP = False
FINETUNE = False
dropout_rate = 0.2

num_classes = len(os.listdir(train_dir))
if TF_APP: 
    from tensorflow.keras.applications import * 
else:
    from models.efficientnet import EfficientNetB0 
    from models.efficientnet import center_crop_and_resize, preprocess_input

with strategy.scope():
    model = models.Sequential()

    conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    # # model.add(layers.Flatten(name="flatten"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_out"))

    # model.add(layers.Dense(256, activation='relu', name="fc1"))
    model.add(layers.Dense(3, activation='softmax', name="fc_out"))
    model.summary()

    print(f'The number of trainable layers before freezing backbone: {len(model.trainable_weights)}')

    if FINETUNE:
        conv_base.trainable = True

        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'multiply_16': ### last layer of backbone
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        print('This is the number of trainable layers '
            'after freezing the conv base:', len(model.trainable_weights))
    else:
        conv_base.trainable = True

        print('This is the number of trainable layers '
            'after freezing the conv base:', len(model.trainable_weights))

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(learning_rate=2e-5),
                metrics=['acc'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(osp.join(ckpt_results, 'best_model.h5'), save_weights_only=True, \
                                    save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(),
]
history = model.fit(
      train_generator,
      steps_per_epoch= NUM_TRAIN //batch_size,
      epochs=epochs,
      callbacks=callbacks,
      validation_data=validation_generator,
      validation_steps= NUM_TEST //batch_size,
      verbose=1,
      use_multiprocessing=True,
      workers=4
      )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_x = range(len(acc))

plt.plot(epochs_x, acc, 'bo', label='Training acc')
plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_x, loss, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig(osp.join(vis_results, 'res.png'))