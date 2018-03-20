import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm


# 把传入的value转化为整数型的属性，int64_list对应着 tf.train.Example 的定义
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 把传入的value转化为字符串型的属性，bytes_list对应着 tf.train.Example 的定义
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_tfrecord(tensor, file_name, output_dir, resize=False, shape=(-1, -1)):
    """
    :param  tensor:         two numpy arrays (images, labels)
    :param  file_name:      *.tfrecord file name
    :param  output_dir:     output dictionary
    :param  resize:         resize or not
    :param  shape:          resize shape
    """

    if not os.path.exists(output_dir, ):
        os.mkdir(output_dir)
    _examples, _labels = tensor
    save_path = os.path.join(output_dir, file_name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(save_path)
    n_classes = _labels[np.argmax(_labels)] + 1
    print(n_classes)
    for i, (example, label) in tqdm(enumerate(zip(_examples, _labels))):
        if resize:
            image = tf.image.resize_images(example, (shape[0], shape[1]))
        else:
            image = example

        image_raw = image.tostring()

        if len(image.shape) <= 2:
            depth = 1
        else:
            depth = image.shape[2]

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(depth),
            'label': _int64_feature(label),
            'num_classes': _int64_feature(n_classes)
        }))

        writer.write(example.SerializeToString())
    writer.close()

    pass

