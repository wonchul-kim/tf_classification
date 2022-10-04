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
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
val_dataloader = Dataloader(val_dataset, batch_size=2, shuffle=True)

### training params.
epochs = 50
dropout_rate = 0.2
### Define model && strategy for multi-gpu
strategy = tf.distribute.MirroredStrategy()
print('* Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    ### define MODEL
    model = models.Sequential()

    conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_out"))

    model.add(layers.Dense(train_dataset.get_num_classes(), activation='softmax', name="fc_out"))
    model.summary()

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0, \
                        reduction=tf.keras.losses.Reduction.NONE, name='categorical_crossentropy')



    def compute_loss(labels, predictions):
        per_example_loss = loss_fn(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')


    optimizer = optimizers.RMSprop(learning_rate=2e-5)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)

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


def train_generator():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_dataloader, use_multiprocessing=False)
    multi_enqueuer.start(workers=4, max_queue_size=10)
    for _ in range(len(train_dataloader)):
        batch_xs, batch_ys = next(multi_enqueuer.get())
        yield batch_xs, batch_ys

_train_dataset = tf.data.Dataset.from_generator(train_generator,
                                         output_types=(tf.float64, tf.float32),
                                        #  output_shapes=(tf.TensorShape([None, None, None, None]),
                                        #                 tf.TensorShape([None, None, None]))
                                        )

def valid_generator():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(val_dataloader, use_multiprocessing=False)
    multi_enqueuer.start(workers=4, max_queue_size=10)
    for _ in range(len(val_dataloader)):
        batch_xs, batch_ys = next(multi_enqueuer.get()) # I have three outputs
        yield batch_xs, batch_ys

_valid_dataset = tf.data.Dataset.from_generator(valid_generator,
                                         output_types=(tf.float64, tf.float32),
                                        #  output_shapes=(tf.TensorShape([None, None, None, None]),
                                        #                 tf.TensorShape([None, None]))
                                        )

train_dist_dataset = strategy.experimental_distribute_dataset(_train_dataset)
valid_dist_dataset = strategy.experimental_distribute_dataset(_valid_dataset)

### declare train/val step functions 
@tf.function
def train_step(batch):
    image, label = batch 
    label = tf.cast(label, tf.float32)
    label = tf.squeeze(label, axis=1)
    with tf.GradientTape() as tape:
        preds = model(image)
        loss = compute_loss(label, preds)

    train_accuracy.update_state(label, preds)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss

@tf.function
def distributed_train_epoch(ds):
    total_loss = 0.
    num_train_batches = 0.
    for batch in ds:
        per_replica_loss = strategy.run(train_step, args=(batch,))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        total_loss += loss
        num_train_batches += 1

    return total_loss, num_train_batches

@tf.function
def val_step(batch):
    image, label = batch 
    preds = model(image)
    label = tf.cast(label, tf.float32)
    label = tf.squeeze(label, axis=1)
    val_loss.update_state(loss_fn(label, preds)[0])
    val_accuracy.update_state(label, preds)


@tf.function
def distributed_val_epoch(ds):
    for batch in ds:
        strategy.run(val_step, args=(batch,))

train_steps = len(train_dataset)//batch_size
valid_steps = len(val_dataset)//1

if len(train_dataset) % batch_size != 0:
    train_steps += 1
if len(val_dataset) % 1 != 0:
    valid_steps += 1


best_acc = -1
best_loss = 999
for epoch in range(epochs):
    # initializes training losses and acc_scores lists for the epoch
    train_total_loss, num_train_batches = distributed_train_epoch(train_dist_dataset)
    train_avg_loss = train_total_loss / num_train_batches 
    train_avg_acc = train_accuracy.result()*100
    
    distributed_val_epoch(valid_dist_dataset)
    val_avg_loss = val_loss.result()
    val_avg_acc = val_accuracy.result()*100

    print('>> Epoch: {}, Train Loss: {}, Train acc: {}, Val Loss: {}, Val acc: {}'.format(epoch, train_avg_loss, \
                                                                            train_avg_acc, val_avg_loss, val_avg_acc))

    if val_avg_acc > best_acc:
        best_acc = val_avg_acc
        model.save_weights(osp.join(ckpt_results, 'best_acc.h5'))
        print(f"Saved {osp.join(ckpt_results, 'best_acc.h5')} ......!")

    if val_avg_loss < best_loss:
        best_loss = val_avg_loss
        model.save_weights(osp.join(ckpt_results, 'best_loss.h5'))
        print(f"Saved {osp.join(ckpt_results, 'best_loss.h5')} ......!")

    val_loss.reset_states()
    train_accuracy.reset_states()
    val_accuracy.reset_states()


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