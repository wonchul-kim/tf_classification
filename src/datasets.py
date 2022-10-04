import os 
import os.path as osp
from glob import glob
import cv2 
import numpy as np 
import tensorflow as tf 

class DirectoryDataset: 
    def __init__(self, input_dir, mode, input_formats=['bmp', 'png'], augmentation=None, preprocessing=None):
        _file_path = osp.join(input_dir, mode)
        assert osp.exists(_file_path), f"There is no such file: f{_file_path}"
        
        self._classes = [_dir for _dir in os.listdir(_file_path) if os.path.isdir(os.path.join(_file_path, _dir))]
        print(f"There are {self._classes} classes for {mode} datasets")

        self._class2label = {val: idx for idx, val in enumerate(self._classes)}
        self._label2class = {idx: val for idx, val in enumerate(self._classes)}

        self._num_classes = len(self._classes)
        self._data = []
        for input_format in input_formats:
            for idx, _class in enumerate(self._classes):
                img_files = glob(osp.join(_file_path, _class, '*.{}'.format(input_format)))
                labels = [idx]*len(img_files)
                self._data += [[img_file, label] for img_file, label in zip(img_files, labels)]
        
        print(f"There are {len(self._data)} images in {_file_path}")
        assert len(self._data) != 0, f"There is no image and label data: {_file_path}"  

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def get_classes(self):
        return self._classes

    def get_num_classes(self):
        return self._num_classes

    def get_class2label(self):
        return self._class2label 

    def get_label2class(self):
        return self._label2class

    def __getitem__(self, idx):
        _img_fp, _label = self._data[idx]
        img = cv2.imread(_img_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_label = np.eye(self._num_classes)[[_label]]
        
        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=img)
        #     img = sample['image']

        if self.preprocessing:
            img = self.preprocessing(img)

            # MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
            # STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

            # img = img - np.array(MEAN_RGB)
            # img = img / np.array(STDDEV_RGB)

        return img, encoded_label
        
    def __len__(self):
        return len(self._data)  

class Dataloader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[self.indexes[j]])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   