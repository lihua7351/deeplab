# -*- coding: utf-8 -*-
# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts define data to TFRecord file format with Example protos.

define dataset is expected to have the following directory structure:

  - build_data.py
  - build_define_data.py (current working directory).
  + define_data
    + images
    + labels
    + vis
    + Segmentation
    + tfrecord

Image folder:
  ./define_data/images

Semantic segmentation annotations:
  ./define_data/labels

list folder:
  ./define_data/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os.path
import sys
import build_data
from six.moves import range
import tensorflow as tf
import cv2
import numpy as np
# import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
FLAGS = tf.compat.v1.app.flags.FLAGS



tf.compat.v1.app.flags.DEFINE_string('image_folder',
                           './define_data/images',
                           'Folder containing images.')

tf.compat.v1.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './define_data/labels',
    'Folder containing semantic segmentation annotations.')

tf.compat.v1.app.flags.DEFINE_string(
    'list_folder',
    './define_data/Segmentation',
    'Folder containing lists for training and validation')

tf.compat.v1.app.flags.DEFINE_string(
    'output_dir',
    './define_data/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')

tf.compat.v1.app.flags.DEFINE_list('eval_crop_size', '513,513',
                  'Image crop size [height, width] during eval.')

_NUM_SHARDS = 4


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

  image_reader = build_data.ImageReader('png', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
        image_data = tf.io.gfile.GFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i] + '.' + FLAGS.label_format)
        seg_data = tf.io.gfile.GFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def _convert_dataset_with_crop(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))
  crop_size = [int(sz) for sz in FLAGS.eval_crop_size]
  crop_height = crop_size[0]
  crop_width = crop_size[1]

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
        # image_data = tf.gfile.GFile(image_filename, 'rb').read()
        # height, width = image_reader.read_image_dims(image_data)
        image_data = cv2.imdecode(np.fromfile(image_filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i] + '.' + FLAGS.label_format)
        seg_data = cv2.imdecode(np.fromfile(seg_filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        height = image_data.shape[0]
        width = image_data.shape[1]
        # if img_size is less than cropsize
        if (height <= crop_height) :
          crop_height = height
        if (width <= crop_width):
          crop_width = width
        col = int(width/crop_width)
        row = int(height/crop_height)
        for r in range(0,row):
          for c in range(0,col):
            crop_img = image_data[r*crop_height:(r+1)*crop_height,c*crop_width:(c+1)*crop_width]
            str_crop_img = encode_image(crop_img, FLAGS.image_format)
            crop_label = seg_data[r*crop_height:(r+1)*crop_height,c*crop_width:(c+1)*crop_width]
            str_crop_label = encode_image(crop_label, FLAGS.label_format)
            file_name = filenames[i] + "_r" + str(r) + "_c" + str(c)
            # Convert to tf example.
            example = build_data.image_seg_to_tfexample(
                str_crop_img, file_name, crop_height, crop_width, str_crop_label)
            tfrecord_writer.write(example.SerializeToString())
            if c+1==col and r+1==row:
              crop_img = image_data[height-crop_height:height,width-crop_width:width]
              str_crop_img = encode_image(crop_img, FLAGS.image_format)
              crop_label = seg_data[height-crop_height:height,width-crop_width:width]
              str_crop_label = encode_image(crop_label, FLAGS.label_format)
              file_name = filenames[i] + "_r" + str(r) + "_c" + str(c)
              # Convert to tf example.
              example = build_data.image_seg_to_tfexample(
                  str_crop_img, file_name, crop_height, crop_width, str_crop_label)
              tfrecord_writer.write(example.SerializeToString())
            if c+1==col:
              crop_img = image_data[r*crop_height:(r+1)*crop_height,width-crop_width:width]
              str_crop_img = encode_image(crop_img, FLAGS.image_format)
              crop_label = seg_data[r*crop_height:(r+1)*crop_height,width-crop_width:width]
              str_crop_label = encode_image(crop_label, FLAGS.label_format)
              file_name = filenames[i] + "_r" + str(r) + "_c" + str(c)
              # Convert to tf example.
              example = build_data.image_seg_to_tfexample(
                  str_crop_img, file_name, crop_height, crop_width, str_crop_label)
              tfrecord_writer.write(example.SerializeToString())
            if r+1==row:
              crop_img = image_data[height-crop_height:height,c*crop_width:(c+1)*crop_width]
              str_crop_img = encode_image(crop_img, FLAGS.image_format)
              crop_label = seg_data[height-crop_height:height,c*crop_width:(c+1)*crop_width]
              str_crop_label = encode_image(crop_label, FLAGS.label_format)
              file_name = filenames[i] + "_r" + str(r) + "_c" + str(c)
              # Convert to tf example.
              example = build_data.image_seg_to_tfexample(
                  str_crop_img, file_name, crop_height, crop_width, str_crop_label)
              tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def encode_image(img, format):
  img_encode = cv2.imencode('.'+format,img)[1]
  data_encode = np.array(img_encode)
  str_encode = data_encode.tobytes()
  return str_encode

def main(unused_argv):
  dataset_splits = tf.io.gfile.glob(os.path.join(FLAGS.list_folder, '*.txt'))
  output_dir = FLAGS.output_dir
  if os.path.exists(output_dir) == False:
    os.makedirs(output_dir)
  if os.listdir(output_dir):
    print('*'*50)
    print('File already exists in {}, please delete it and try again.'.format(output_dir))
    return
  for dataset_split in dataset_splits:
    dataset_name = os.path.basename(dataset_split)[:-4]
    if 'val'==dataset_name.strip():
      _convert_dataset_with_crop(dataset_split)
    else:
      _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.compat.v1.app.run()
  # tf.app.run()
