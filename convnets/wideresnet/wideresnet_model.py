# File: wideresnet_model.py
# Author: Arizona Autonomous
# Description: This contains the base WideResNet model, originally
#                built for CIFAR-10. It is designed for the AUVSI
#                SUAS 2017 competition

""" Wide ResNet

Related papers:
https://arxiv.org/pdf/1605.07146.pdf -- Wide ResNet
https://arxiv.org/pdf/1512.03385.pdf -- ResNet (Original)
https://arxiv.org/pdf/1603.05027.pdf -- ResNet (Follow-Up)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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
# MOVING_AVERAGE_DECAY = 0.9999
MOMENTUM = 0.9
MOMENTUM_TYPE = 'Nesterov'
MINIBATCH_SIZE = 128
DAMPENING_RATIO = 0
NUM_EPOCHS_PER_DECAY = 60
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.2
WEIGHT_DECAY = 0.0005

# TODO: This whole function
def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build WRN-28-10 model

  See here for reference: https://arxiv.org/pdf/1605.07146

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Set hyperparameters (optional, may hardcode)
  
  # Build graph
  # -- create global_step (??)
  # -- _build_model
  # ---- _residual
  # ------ orig_x = x
  # ------ x = _batch_norm -> _relu -> _conv
  # -------- _batch_norm: set-up and use tf.nn.batch_normalization
  # -------- _relu: tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu') (or tf.maximum?)
  # -------- _conv: set up weights, use tf.nn.conv2d
  # ------ NOTE: Check for dimension reduction (how to handle?)
  # ------ x = x + orig_x (residual + shortcut)
  
  # Add summaries
  
  # Return logits (and auxiliary logits?)
  
# TODO: This whole function
def loss(logits, labels, batch_size=None):
  """Adds all losses for the model.

  Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    batch_size: integer

  Returns:
    loss
  """
  
  # Reshape labels into dense Tensor, e.g.
  # [0, 5, 3, 4] (batch of 4)
  #      ||
  #      \/
  # [1 0 0 0 0 0 0 0 0 0]
  # [0 0 0 0 0 1 0 0 0 0]
  # [1 0 0 1 0 0 0 0 0 0]
  # [1 0 0 0 1 0 0 0 0 0]
  
  # Calculate cross entropy loss
  # NOTE: Decay?
  
  # Calculate cross entropy loss of auxiliary softmax? (wth is this?)
  
# Submodules
# TODO: Batch norm. Like the whole thing.
def _batch_norm(inputs, scope):
  """Batch normalization"""
  pass

def _relu(inputs, leakiness=0.333):
  """Relu, with optional (default) leaky support"""
  return tf.where(tf.less(inputs, 0.0), leakiness * inputs,
                  inputs, name='leaky_relu')
  # Alternatively:
  # return tf.maximum(leakiness * input, input)

# TODO: Change weight initializer and std_dev (see resnet_model from tf models)
def _conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           stddev=0.01,
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

    l2_regularizer = losses.l2_regularizer(WEIGHT_DECAY)
    
    weights = tf.get_variable('weights', shape=weights_shape,
                              dtype=tf.float32, initializer=weights_initializer,
                              regularizer=l2_regularizer, trainable=trainable,
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES]) # TODO -- what are collections?
    # Add convolution to graph -- y_int = (w*x)
    return = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1],
                        padding=padding)
