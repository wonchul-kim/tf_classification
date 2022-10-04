import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from glob import glob 

import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from src.datasets import DirectoryDataset, Dataloader
from utils.augment import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils.helpers import denormalize

### configurate GPUs settings
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    os.mkdir(osp.join(vis_results, 'debug'))
ckpt_results = osp.join(output_dir, 'checkpoints')
if not osp.exists(ckpt_results):
    os.mkdir(ckpt_results)

### define dataset & dataloader
input_dir = '/home/wonchul/HDD/datasets/rps'
TF_APP = False
FINETUNE = False

dropout_rate = 0.2

model = models.Sequential()
if TF_APP: 
    from tensorflow.keras.applications import * 
else:
    from models.efficientnet import EfficientNetB0 
    from models.efficientnet import center_crop_and_resize, preprocess_input

test_dataset = DirectoryDataset(input_dir, 'test', preprocessing=preprocess_input)
label2class = test_dataset.get_label2class()

## To check the dataset is working well
idxes = [10, 150, 250]
for idx in idxes:
    batch = test_dataset[idx]
    print(idx, batch[0].shape, batch[1].shape, np.mean(batch[0])) 
    fig = plt.figure()
    plt.imshow(batch[0])
    plt.title(label2class[np.argmax(np.array(batch[1]))])
    plt.savefig("results/figs/debug/test_batch_{}".format(idx))
    plt.close()

width = 300
height = 300
input_shape = (height, width, 3)

### define MODEL

conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
model.add(layers.Dense(3, activation='softmax', name="fc_out"))
model.summary()

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


model.load_weights("./results/checkpoints/best_loss.h5")

for idx in range(len(test_dataset)):
    if idx%10 == 0:
        batch = test_dataset[idx]
        img = batch[0]
        img = np.expand_dims(img, axis=0)
        label = batch[1]

        preds = model(img)

        fig = plt.figure()
        # plt.imshow(denormalize(batch[0]))
        plt.imshow(batch[0])
        plt.title(label2class[np.argmax(np.array(preds), axis=1)[0]])
        plt.savefig("results/figs/preds_batch_{}".format(idx))
        plt.close()


# def predict_image(model, img_path, class_lookup):
#     # Read the image and resize it
#     img = image.load_img(img_path, target_size=(height, width))
#     # Convert it to a Numpy array with target shape.
#     x = image.img_to_array(img)
#     # Reshape
#     x = x.reshape((1,) + x.shape)
#     # x /= 255.
#     result_verbose = model.predict([x])

#     print(result_verbose)

#     predicted_class = class_lookup[np.argmax(result_verbose, axis=1)[0]]
#     predicted_probability = result_verbose[0][np.argmax(result_verbose, axis=1)[0]]

#     return predicted_class ,predicted_probability, result_verbose


