import os 

import segmentation_models as sm 
import tensorflow as tf 
import numpy as np 
import tf2onnx
from tensorflow.keras import models
from tensorflow.keras import layers

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

TF_APP = False
FINETUNE = False

if TF_APP: 
    from tensorflow.keras.applications import * 
else:
    from models.efficientnet import EfficientNetB0 
    from models.efficientnet import center_crop_and_resize, preprocess_input

dropout_rate = 0.2

width = 300
height = 300
input_shape = (height, width, 3)

model = models.Sequential()

conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))

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



model.load_weights("./results/checkpoints/best_acc.h5")

in_shape = model.inputs[0].shape.as_list()
in_shape[0] = 1
in_shape[1] = 300
in_shape[2] = 300
spec = (tf.TensorSpec(in_shape, tf.float32, name="data"),)        
output_path = './results/best_model.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=12, output_path=output_path) 