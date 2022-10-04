## Tensorflow for semantic segmentation using custom dataset

### Contents

#### TRAIN

- 1. <a href="https://github.com/wonchul-kim/tf_segmentation/blob/master/1_train_fit.py">train with `fit` function of `tf.keras.model` module using single GPU and ImageDataGenerator</a>

- 2. <a href="https://github.com/wonchul-kim/tf_segmentation/blob/master/2_train_loop.py">custom train loop using single GPU and ImageDataGenerator</a>

- 3. <a href="https://github.com/wonchul-kim/tf_segmentation/blob/master/3_train_loop_multigpu.py">custom train loop using single GPU and custom dataset</a>

- 4. <a href="https://github.com/wonchul-kim/tf_segmentation/blob/master/3_train_loop_multigpu.py">custom train loop using multiple GPUs and custom dataset</a>

#### TEST

- <a href="https://github.com/wonchul-kim/tf_segmentation/blob/master/test.py">test trained model</a>

#### EXPORT to ONNX

- <a href="https://github.com/wonchul-kim/tf_segmentation/blob/master/export2onnx.py">export trained model into ONNX</a>

### ToDo

- [ ] `EfficientNet from official EfficientNet source` is better than `EfficientNet from tensorflow.keras.applications`, Need to figure out.

## References

- Used Efficient model from https://github.com/qubvel/efficientnet
- another models from https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
- https://www.tensorflow.org/tutorials/images/transfer_learning?hl=ko
