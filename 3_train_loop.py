import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from src.datasets import DirectoryDataset, Dataloader
from utils.augment import get_training_augmentation, get_preprocessing, get_validation_augmentation

### configurate GPUs settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

if TF_APP: 
    from tensorflow.keras.applications import * 
else:
    from models.efficientnet import EfficientNetB0 
    from models.efficientnet import center_crop_and_resize, preprocess_input

# train_dataset = DirectoryDataset(input_dir, 'train', augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocess_input))
# val_dataset = DirectoryDataset(input_dir, 'test', augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocess_input))
train_dataset = DirectoryDataset(input_dir, 'train', preprocessing=preprocess_input)
val_dataset = DirectoryDataset(input_dir, 'test', preprocessing=preprocess_input)

label2class = train_dataset.get_label2class()

## To check the dataset is working well
idxes = [10, 1500, 2500]
for idx in idxes:
    batch = train_dataset[idx]
    print(idx, batch[0].shape, batch[1].shape, np.mean(batch[0])) 
    fig = plt.figure()
    plt.imshow(batch[0])
    plt.title(label2class[np.argmax(np.array(batch[1]))])
    plt.savefig("results/figs/debug/train_batch_{}".format(idx))
    plt.close()

idxes = [10, 150, 250]
for idx in idxes:
    batch = val_dataset[idx]
    print(idx, batch[0].shape, batch[1].shape, np.mean(batch[0])) 
    fig = plt.figure()
    plt.imshow(batch[0])
    plt.title(label2class[np.argmax(np.array(batch[1]))])
    plt.savefig("results/figs/debug/val_batch_{}".format(idx))
    plt.close()

width = 300
height = 300
input_shape = (height, width, 3)
batch_size = 32


train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = Dataloader(val_dataset, batch_size=1, shuffle=True)

### training params.
epochs = 50
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0, reduction=tf.keras.losses.Reduction.AUTO,
    name='categorical_crossentropy'
)
acc_fn = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)

# optimizer = optimizers.RMSprop(learning_rate=2e-5)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
### declare train/val step functions 
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        preds = model(x)
        # preds = tf.cast(preds, tf.float32)
        y = tf.cast(y, tf.float32)
        y = tf.squeeze(y, axis=1)
        loss = loss_fn(y, preds)
        acc = acc_fn(y, preds)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss, acc

@tf.function
def validation_step(x, y):
    preds = model(x)
    y = tf.cast(y, tf.float32)

    y = tf.squeeze(y, axis=1)
    loss = loss_fn(y, preds)
    acc = acc_fn(y, preds)

    return loss, acc

train_steps = len(train_dataset)//batch_size
valid_steps = len(val_dataset)//1

if len(train_dataset) % batch_size != 0:
    train_steps += 1
if len(val_dataset) % 1 != 0:
    valid_steps += 1

### define MODEL

dropout_rate = 0.2

model = models.Sequential()

conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dropout(0.2, name="dropout_out"))

model.add(layers.Dense(train_dataset.get_num_classes(), activation='softmax', name="fc_out"))
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


best_acc = 0.0
best_loss = 999
for epoch in range(epochs):
    # initializes training losses and iou_scores lists for the epoch
    losses = []
    accuracies = []

    for step, (batch) in enumerate(train_dataloader):

        x, y = batch[0], batch[1]
        # run one training step for the current batch
        loss, acc = train_step(x, y)

        # Save current batch loss and iou-score
        losses.append(float(loss))
        accuracies.append(float(acc))

        # print("loss: ", sum(losses), len(losses))
        # print("acc: ", sum(accuracies), len(accuracies))
        print("\r Epoch: {} >> step: {}/{} >> train-loss: {} >> train-acc: {}".format(epoch, step + 1, train_steps, \
                                np.round(sum(losses) / len(losses), 6), np.round(sum(accuracies) / len(accuracies), 4)), end="")
    print()

    if epoch % 1 == 0 and epoch != 0:
        val_losses = []
        val_accuracies = []
        for val_step, val_batch in enumerate(val_dataloader):
            x_val, y_val = val_batch[0], val_batch[1]

            val_loss, val_acc = validation_step(x_val, y_val)


            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print("\r** Epoch: {} >> step: {}/{} >> Val_Loss: {:.8f} >> val-acc: {} ".format(epoch, val_step,\
                                                    valid_steps, np.round(sum(val_losses) / len(val_losses), 8), \
                                                    np.round(sum(val_accuracies) / len(val_accuracies), 4)), end="")
        print()
                
        if sum(val_accuracies) / len(val_accuracies) > best_acc:
            best_acc = sum(val_accuracies) / len(val_accuracies)
            model.save_weights(osp.join(ckpt_results, 'best_acc.h5'))
            print(f"Saved {osp.join(ckpt_results, 'best_acc.h5')} ......!")

        if sum(val_losses) / len(val_losses) < best_loss:
            best_loss = sum(val_losses) / len(val_losses)
            model.save_weights(osp.join(ckpt_results, 'best_loss.h5'))
            print(f"Saved {osp.join(ckpt_results, 'best_loss.h5')} ......!")




# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_x = range(len(acc))

# plt.plot(epochs_x, acc, 'bo', label='Training acc')
# plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs_x, loss, 'bo', label='Training loss')
# plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.savefig(osp.join(vis_results, 'res.png'))