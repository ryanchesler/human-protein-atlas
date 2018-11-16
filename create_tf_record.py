#!/usr/bin/python
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Converts HPA data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import pandas as pd
import numpy as np

import tensorflow as tf

FLAGS = None


def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  #images = data_set.images
  #labels = data_set.labels
  num_examples = len(data_set)

  #rows = images.shape[1]
  #cols = images.shape[2]
  #depth = images.shape[3]

  channels = ['green', 'red', 'blue', 'yellow']
  filename = os.path.join(FLAGS.output_dir, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for r in data_set.itertuples():
      imagenames=[]
      images=[]
      features={}
      for c in channels:
        fname="{Id}_{Colour}.png".format(Id=r.Id, Colour=c)
        with open(os.path.join(FLAGS.input_dir, fname), 'rb') as f:
          features['image/%s/filename'%c] = _bytes_feature(fname.encode('utf8'))
          features['image/%s/encoded'%c]  = _bytes_feature(f.read())
      features['image/format']        = _bytes_feature('PNG'.encode('utf8'))
      label=np.zeros(28, dtype='float32')
      for lbl in r.Target.split(' '):
        label[int(lbl)]=1
      features['image/label']=_float32_feature(label)
      features['image/id']=_bytes_feature(r.Id.encode('utf8'))

      tf_example=tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())


def main(unused_argv):
  data = pd.read_csv(os.path.join(FLAGS.input_dir, 'train.csv'))
  shard_size=int(len(data)/FLAGS.num_shards)
  shards=[]
  for i in range(FLAGS.num_shards-1):
    shard=data.sample(shard_size)
    data.drop(shard.index, inplace=True)
    shards.append(shard)
  shards.append(data)
  for i, shard in enumerate(shards):
    print("Shard size {size}".format(size=len(shard)))
    convert_to(shard, 'hpa_{dim}_{num}'.format(dim=FLAGS.dimensions, num=i))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--output_dir',
      type=str,
      default='/tmp/data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--input_dir',
      type=str,
      default='/tmp/data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--dimensions',
      type=str,
      default='512x512',
      help="image dimensions (WxH)"
  )
  parser.add_argument(
      '--num_shards',
      type=int,
      default=10,
      help="number of shards to make"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
