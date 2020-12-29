from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', r'e:/python/tf/raccoon_dataset/images', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', r'e:/python/tf/raccoon_dataset/train.txt', 'train')
flags.DEFINE_string('annotations_dir', r'e:/python/tf/raccoon_dataset/annotations', 'annotations')
flags.DEFINE_string('year', 'VOC2007', 'Desired challange year.')
flags.DEFINE_string('output_path', r'e:/python/tf/raccoon_dataset/output/train.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', r'e:/python/tf/raccoon_dataset/pacal_label_map.pbtxt', 'Path to label map photo')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficutl instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

def dict_to_tf_example(data,
                        dataset_directory, 
                        label_map_dict,
                        ignore_difficult_instances=False,
                        image_subdirectory='JPEGImages'):
    img_path = os.path.join(dataset_directory, data['filename'].replace('.png', '.jpg').replace('.PNG', '.jpg'))
    full_path = img_path
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue
        difficult_obj.append(int(difficult))
        xmin.append(float(obj['bndbox']['xmin'])/ width)
        ymin.append(float(obj['bndbox']['ymin'])/ height)
        xmax.append(float(obj['bndbox']['xmax'])/ width)
        ymax.append(float(obj['bndbox']['ymax'])/ height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))

    return example

def main(_):
    data_dir = FLAGS.data_dir
    years = ['VOC2007', 'VOC2012']
    if FLAGS.year != 'merged':
        years = [FLAGS.year]
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    for year in years:
        logging.info('Reading from PASCAL %s dataset', year)
        examples_path = FLAGS.set
        annotations_dir = FLAGS.annotations_dir
        print(examples_path)
        examples_list = dataset_util.read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
            tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict, FLAGS.ignore_difficult_instances)
            writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    tf.app.run()
