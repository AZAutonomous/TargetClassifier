# File: sparsenet_model.py
# Author: Arizona Autonomous
# Description: This contains the base SparseNet model, originally
#                built for CIFAR-10. It is designed for the AUVSI
#                SUAS 2017 competition

""" SparseNet model with Fractional Max Pooling

Related papers:
https://arxiv.org/pdf/1412.6071.pdf
https://arxiv.org/pdf/1409.6070.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import tensorflow as tf

# TEMP CIFAR10 STUFF (TESTING ONLY) -- DELETEME
import cifar10_input
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

FLAGS = tf.app.flags.FLAGS

# Basic model parameters
tf.app.flags.DEFINE_integer('batch_size', 64, 
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing dataset
# TODO

# Constants describing the training process
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

# Utilities
def _activation_summary(x): # TODO
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


# Submodules 
def _relu(input, leakiness=0.333):
  """Relu, with optional (default) leaky support"""
  return tf.where(tf.less(input, 0.0), leakiness*input, input, name='leaky_relu')
      # Alternatively:
      # return tf.maximum(leakiness * input, input)

def _conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           activation_fn=_relu,
           stddev=0.01,
           bias=0.0,
           weight_decay=0,
           batch_norm_params=None,
           is_training=True,
           trainable=True,
           scope=None,
           reuse=None):
  """Adds a 2D convolution followed by an optional batch_norm layer.

  conv2d creates a variable called 'weights', representing the convolutional
  kernel, that is convolved with the input. If `batch_norm_params` is None, a
  second variable called 'biases' is added to the result of the convolution
  operation.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_filters_out: the number of output filters (depth).
    kernel_size: an int representing the dimensions of a (square) filter
    stride: an int representing stride in both x and y
    padding: one of 'VALID' or 'SAME'.
    activation: activation function.
    stddev: standard deviation of the truncated guassian weight distribution.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    batch_norm_params: parameters for the batch_norm. If is None don't use it.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
  Returns:
    a tensor representing the output of the operation.

  """
  # Reuse by default if scope is provided
  if scope is not None and reuse is None:
    reuse = True

  with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
    num_filters_in = inputs.get_shape()[-1]
    weights_shape = [kernel_size, kernel_size,
                     num_filters_in, num_filters_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    
    weights = tf.get_variable('weights', shape=weights_shape,
                              dtype=tf.float32, initializer=weights_initializer,
                              regularizer=l2_regularizer, trainable=trainable,
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES]) # TODO -- what are collections?
    # Add convolution to graph -- y_int = (w*x)
    conv = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1],
                        padding=padding)
                        
    if batch_norm_params is not None:
      print('warning: batch_norm disabled')
      assert False # Force an error like this because I'm dumb
      # with scopes.arg_scope([batch_norm], is_training=is_training,
                            # trainable=trainable, restore=restore):
        # outputs = batch_norm(conv, **batch_norm_params)
    else:
      bias_shape = [num_filters_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = tf.get_variable('biases', shape=bias_shape,
                               dtype=tf.float32, initializer=bias_initializer,
                               trainable=trainable,
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES]) # But really what are collections?
      # Add biases to graph -- y = y_int + b = (w*x + b) 
      outputs = tf.nn.bias_add(conv, biases)
    if activation:
      # Get activiation -- a = f(y), f = activation_fn
      outputs = activation_fn(outputs)
    return outputs

def _fmp(inputs,
         pooling_ratio,
         pseudo_random=True,
         overlapping=True,
         deterministic=None,
         seed=None,
         seed2=None,
         name=None,
         scope=None,
         reuse=None):
  """Adds a fractional max pooling layer

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    pooling_ratio: a float >= 1 (square pooling is assumed)
    pseudo_random: a bool enabling pseudorandom pooling sequence generation
    overlapping: a bool enabling overlapping pooling regions
    deterministic: a bool enabling a fixed pooling region
    seed/seed2: an int used to as the random number generator seed

    name: a name for the operation
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.

  Returns:
    a tuple of Tensor objects [output, row_pooling_seq, col_pooling_seq]
    output: A Tensor, output tensor after fractional max pooling
    row_pooling_seq: A Tensor of type int64. Needed to calculate gradient
    col_pooling_seq: A Tensor of type int64. Needed to calculate gradient

  """
  # Reuse by default if scope is provided
  if scope is not None and reuse is None:
    reuse = True

  with tf.variable_scope(scope, 'FMP', [inputs], reuse=reuse):
    fmp = tf.nn.fractional_max_pool(inputs, [1,pooling_ratio,pooling_ratio,1],
                                             pseudo_random=pseudo_random, 
                                             overlapping=overlapping,
                                             deterministic=deterministic,
                                             seed=seed, seed2=seed2, name=name)
    return fmp
    
# Main model functions
def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)

  # TODO: Resize images to 94x94

  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)

  # TODO: Resize images to 94x94

  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build SparseConvNet with Fractional Max Pooling model architecture.
  
  Related papers:
  https://arxiv.org/pdf/1412.6071.pdf
  https://arxiv.org/pdf/1409.6070.pdf
  
  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifier

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # BatchNorm -- TODO
  
  # Weight decay -- TODO
  # Collect relevant activations for external use, e.g. summaries or losses
  end_points = {}
  # TODO: Dropout, auxiliary nodes, stuff
  # TODO: BatchNorm params (from Inception: batch_norm_decay=0.9997, batch_norm_epsilon=0.001)
  with tf.name_scope(scope, 'sparseconvnet', [images]):
    # 94 x 94 x 3
    end_points['conv0'] = _conv2d(images, num_filters_out=160, kernel_size=2, 
                                  scope='conv0',
                                  weight_decay=0.00004, stddev=0.1)
    # 93 x 93 x 160
    end_points['fmp0'] = _fmp(end_points['conv0'], tf.pow(2, 0.333), 
                              pseudo_random=True, overlapping=True, 
                              scope='fmp0')
    # 74 x 74 x 160
    end_points['conv1'] = _conv2d(end_points['fmp0'], num_filters_out=160, 
                                  kernel_size=2, scope='conv1',
                                  weight_decay=0.00004, stddev=0.1)
    # 73 x 73 x 320
    end_points['fmp1'] = _fmp(end_points['conv1'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp1')
    # 58 x 58 x 320
    end_points['conv2'] = _conv2d(end_points['fmp1'], num_filters_out=320, 
                                  kernel_size=2, scope='conv2',
                                  weight_decay=0.00004, stddev=0.1)
    # 57 x 57 x 480
    end_points['fmp2'] = _fmp(end_points['conv2'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp2')
    # 45 x 45 x 480
    end_points['conv3'] = _conv2d(end_points['fmp2'], num_filters_out=480, 
                                  kernel_size=2, scope='conv3',
                                  weight_decay=0.00004, stddev=0.1)
    # 44 x 44 x 640
    end_points['fmp3'] = _fmp(end_points['conv3'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp3')
    # 35 x 35 x 640
    end_points['conv4'] = _conv2d(end_points['fmp3'], num_filters_out=640, 
                                  kernel_size=2, scope='conv4',
                                  weight_decay=0.00004, stddev=0.1)
    # 34 x 34 x 800
    end_points['fmp4'] = _fmp(end_points['conv4'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp4')
    # 27 x 27 x 800
    end_points['conv5'] = _conv2d(end_points['fmp4'], num_filters_out=800, 
                                  kernel_size=2, scope='conv5',
                                  weight_decay=0.00004, stddev=0.1)
    # 26 x 26 x 960
    end_points['fmp5'] = _fmp(end_points['conv5'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp5')
    # 21 x 21 x 960
    end_points['conv6'] = _conv2d(end_points['fmp5'], num_filters_out=960, 
                                  kernel_size=2, scope='conv6',
                                  weight_decay=0.00004, stddev=0.1)
    # 20 x 20 x 1120
    end_points['fmp6'] = _fmp(end_points['conv6'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp6')
    # 16 x 16 x 1120
    end_points['conv7'] = _conv2d(end_points['fmp6'], num_filters_out=1120, 
                                  kernel_size=2, scope='conv7',
                                  weight_decay=0.00004, stddev=0.1)
    # 15 x 15 x 1280
    end_points['fmp7'] = _fmp(end_points['conv7'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp7')
    # 12 x 12 x 1280
    end_points['conv8'] = _conv2d(end_points['fmp7'], num_filters_out=1280, 
                                  kernel_size=2, scope='conv8',
                                  weight_decay=0.00004, stddev=0.1)
    # Apply dropout if training
    if for_training:
      end_points['conv8'] = tf.nn.dropout(end_points['conv8'], 0.1)
    # 11 x 11 x 1440
    end_points['fmp8'] = _fmp(end_points['conv8'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp8')
    # 9 x 9 x 1440
    end_points['conv9'] = _conv2d(end_points['fmp8'], num_filters_out=1440, 
                                  kernel_size=2, scope='conv9',
                                  weight_decay=0.00004, stddev=0.1)
    # Apply dropout if training
    if for_training:
      end_points['conv9'] = tf.nn.dropout(end_points['conv9'], 0.2)
    # 8 x 8 x 1600
    end_points['fmp9'] = _fmp(end_points['conv9'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp9')
    # 6 x 6 x 1600
    end_points['conv10'] = _conv2d(end_points['fmp9'], num_filters_out=1600, 
                                  kernel_size=2, scope='conv10',
                                   weight_decay=0.00004, stddev=0.1)
    # Apply dropout if training
    if for_training:
      end_points['conv10'] = tf.nn.dropout(end_points['conv10'], 0.3)
    # 5 x 5 x 1760
    end_points['fmp10'] = _fmp(end_points['conv10'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp10')
    # 4 x 4 x 1760
    end_points['conv11'] = _conv2d(end_points['fmp10'], num_filters_out=1760, 
                                  kernel_size=2, scope='conv11',
                                   weight_decay=0.00004, stddev=0.1)
    # Apply dropout if training
    if for_training:
      end_points['conv11'] = tf.nn.dropout(end_points['conv11'], 0.4)
    # 3 x 3 x 1920
    end_points['fmp11'] = _fmp(end_points['conv11'], tf.pow(2, 0.333),
                              pseudo_random=True, overlapping=True,
                              scope='fmp11')
    # 2 x 2 x 1920
    end_points['conv12'] = _conv2d(end_points['fmp11'], num_filters_out=1920, 
                                  kernel_size=2, scope='conv12',
                                   weight_decay=0.00004, stddev=0.1)
    # Apply dropout if training
    if for_training:
      end_points['conv12'] = tf.nn.dropout(end_points['conv12'], 0.5)
    # 1 x 1 x 2080
    end_points['fc0'] = _conv2d(end_points['conv12'], num_filters_out=2080, 
                                  kernel_size=2, scope='fc0',
                                weight_decay=0.00004, stddev=0.1)
    # 1 x 1 x 2240
    # TODO: flatten to 2240 (1D)
    logits = _conv2d(end_points['fc0'], num_filters_out=num_classes,
                     kernel_size=1, scope='fc1') # ??? TODO
    # num_classes
    end_points['logits'] = logits
    end_points['predictions'] = tf.nn.softmax('logits', name='predictions')
    # Outputs 1 x 1 x num_classes
    
  
    # Add summaries for TensorBoard visualization
  
    # Grab logits associated with the side head -- TODO
    auxiliary_logits = endpoints['aux_logits']
  
    # TODO: Finalize return statement
    # return logits, auxiliary_logits
    return logits
 

def loss(logits, labels, batch_size=None):
  """Adds all losses for the model.

  Note the final loss is not returned. Instead, the list of losses are collected
  by slim.losses. The losses are accumulated in tower_loss() and summed to
  calculate the total loss.

  Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    batch_size: integer

  Returns:
    Loss tensor of type float.
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  ''' From ConvNet Tutorial '''
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

  ''' From Inception -- MODIFY!
  # Reshape the labels into a dense Tensor of
  # shape [FLAGS.batch_size, num_classes].
  sparse_labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
  concated = tf.concat(1, [indices, sparse_labels])
  num_classes = logits[0].get_shape()[-1].value
  dense_labels = tf.sparse_to_dense(concated,
                                    [batch_size, num_classes],
                                    1.0, 0.0)

  # Cross entropy loss for the main softmax prediction.
  slim.losses.cross_entropy_loss(logits[0],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=1.0)

  # Cross entropy loss for the auxiliary softmax head.
  slim.losses.cross_entropy_loss(logits[1],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=0.4,
                                 scope='aux_loss')
  '''

# TODO
def train():
  """Train model (based on CIFAR-10 tutorial)

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


