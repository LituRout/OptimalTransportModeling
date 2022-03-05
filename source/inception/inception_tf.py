# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

from tqdm import tqdm

import pdb

MODEL_DIR = 'inception_cache/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10, batch_size=8, mem_fraction=1):
  assert(len(images.shape) == 4, 'Input should be 4 dim')
  assert(np.max(images[0]) > 10, 'Input should be 0 to 255')
  assert(np.min(images[0]) >= 0.0, 'Input should be greater than 0 always')
  bs = batch_size
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    preds = []
    n_batches = int(math.ceil(float(images.shape[0]) / float(bs)))
    for i in tqdm(range(n_batches), desc="IS"):
        pred = sess.run(softmax, {'InputTensor:0': images[(i * bs):min((i + 1) * bs, images.shape[0])]})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                  name='InputTensor')
    _ = tf.import_graph_def(graph_def, name='',
                            input_map={'ExpandDims:0':input_tensor})

  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape(tf.TensorShape(new_shape))
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

def initialize_inception():
  res = 0
  while res == 0:
    try:
      _init_inception()
      res = 1
      print('Inception graph successfully initialized')
    except:
      pass
