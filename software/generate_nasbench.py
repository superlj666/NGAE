from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import sys

from absl import app
from absl import flags
from absl import logging
from nasbench import api
from nasbench.lib import graph_util
import numpy as np
import tensorflow as tf   # For gfile


label_list = ['output', 'input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']

NASBENCH_TFRECORD = './data/nasbench_only108.tfrecord'
nasbench = api.NASBench(NASBENCH_TFRECORD)

with tf.gfile.Open('./data/generated_graphs.json', 'r') as f:
    hash_set = json.load(f)

model_lines = []
for (key, value) in hash_set.items():
  ad_matrix = value[0]
  labeling = value[1]

  model_spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix=ad_matrix,   # output layer
        # Operations at the vertices of the module, matches order of matrix
        ops=[label_list[index+2] for index in labeling])
  data = nasbench.query(model_spec)
  if not data == {}:
    line = str(data['module_adjacency'].tolist()) + ', ' + str(labeling) + ', ' + str(data['trainable_parameters']) + ', ' + str(data['test_accuracy'])
  model_lines.append(line)

with open('./data/nasbench_only108.txt', 'w') as f:
    for line in model_lines:
      f.write(line)
      f.write('\n')