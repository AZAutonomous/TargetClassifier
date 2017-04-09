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
from tensorflow.python.training import moving_averages

FLAGS = tf.app.flags.FLAGS

# Basic model parameters
tf.app.flags.DEFINE_integer('batch_size', 128, 
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Constants describing the training process
MOVING_AVERAGE_DECAY = 0.9999
MINIBATCH_SIZE = 128
DAMPENING_RATIO = 0
NUM_EPOCHS_PER_DECAY = 60
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.2
WEIGHT_DECAY = 0.0005

# Other global flags
VARIABLES_TO_RESTORE = '_variables_to_restore_'
UPDATE_OPS_COLLECTION = '_update_ops_'

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
  with tf.variable_scope(scope, 'WideResNet', [images]):
    # Block 1:  (32x32x3)   -> (32x32x16)  -> (32x32x160)
    block1 = _residual(images, 
              16, 3, 1,
              160, 3, 1,
              is_training=for_training,
              scope='residual_block1')
    # Block 2:  (32x32x160) -> (32x32x160) -> (32x32x160)
    block2 = _residual(block1, 
              160, 3, 1,
              160, 3, 1,
              is_training=for_training,
              scope='residual_block2')
    # Block 3:  (32x32x160) -> (32x32x160) -> (32x32x160)
    block3 = _residual(block2, 
              160, 3, 1,
              160, 3, 1,
              is_training=for_training,
              scope='residual_block3')
    # Block 4:  (32x32x160) -> (32x32x160) -> (32x32x160)
    block4 = _residual(block3, 
              160, 3, 1,
              160, 3, 1,
              is_training=for_training,
              scope='residual_block4')
    # Block 5:  (32x32x160) -> (32x32x160) -> (32x32x160)
    block5 = _residual(block4, 
              160, 3, 1,
              160, 3, 1,
              is_training=for_training,
              scope='residual_block5')
    # Block 6:  (32x32x320) -> (16x16x320) -> (16x16x320)
    block6 = _residual(block5, 
              320, 3, 2,
              320, 3, 1,
              is_training=for_training,
              scope='residual_block6')
    # Block 7:  (16x16x320) -> (16x16x320) -> (16x16x320)
    block7 = _residual(block6, 
              320, 3, 1,
              320, 3, 1,
              is_training=for_training,
              scope='residual_block7')
    # Block 8:  (16x16x320) -> (16x16x320) -> (16x16x320)
    block8 = _residual(block7, 
              320, 3, 1,
              320, 3, 1,
              is_training=for_training,
              scope='residual_block8')
    # Block 9:  (16x16x320) -> (16x16x320) -> (16x16x320)
    block9 = _residual(block8, 
              320, 3, 1,
              320, 3, 1,
              is_training=for_training,
              scope='residual_block9')
    # Block 10: (16x16x320) -> (16x16x320) -> (8x8x640)
    block10 = _residual(block9, 
              320, 3, 1,
              640, 3, 2,
              is_training=for_training,
              scope='residual_block10')
    # Block 11: (8x8x640)   -> (8x8x640)   -> (8x8x640)
    block11 = _residual(block10, 
              640, 3, 1,
              640, 3, 1,
              is_training=for_training,
              scope='residual_block11')
    # Block 12: (8x8x640)   -> (8x8x640)   -> (8x8x640)
    block12 = _residual(block11, 
              640, 3, 1,
              640, 3, 1,
              is_training=for_training,
              scope='residual_block12')
    # Block 13: (8x8x640)   -> (8x8x640)   -> (8x8x640)
    block13 = _residual(block12, 
              640, 3, 1,
              640, 3, 1,
              is_training=for_training,
              scope='residual_block13')
    # Block 14: (8x8x640)   -> (8x8x640)   -> (8x8x640)
    block14 = _residual(block13, 
              640, 3, 1,
              640, 3, 1,
              is_training=for_training,
              scope='residual_block14')
    
    # Average Pool: (8x8x640) -> (1x1x640)
    pool = _avg_pool(block14, filter_size=8, stride=1)
    
    # Fully connected: (1x1x640) -> (1x1xNUM_CLASSES)
    logits = _conv(pool, num_classes, 1, is_training=for_training,
                   restore=restore_logits, scope='fc1')
                   
    # Flatten logits
    logits = tf.squeeze(logits)

    # Optionally softmax for predictions, but NOT for training
    # since the loss function internally runs softmax!
    predictions = tf.nn.softmax(logits, name='softmax')
    if for_training:
      output = logits
    else:
      output = predictions
    
    return output
  
def loss(logits, labels, batch_size=None, scope=None):
  """Adds all losses for the model.

  Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    batch_size: integer

  Returns:
    loss
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Convert labels to one-hot encoding
  num_classes = logits[0].get_shape()[-1].value
  one_hot = tf.one_hot(labels, num_classes)
  logits.get_shape().assert_is_compatible_with(one_hot.get_shape())
  
  with tf.variable_scope(scope, 'cross_entropy_loss', [logits, one_hot]):
    # NOTE: Consider label smoothing (S7, https://arxiv.org/pdf/1512.00567.pdf)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                          logits=logits, labels=one_hot, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='avg_xentropy')
    
    # TODO: Calculate cross entropy loss of auxiliary softmax? (wth is this?)

    # Compute the total loss
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + regularization_loss, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)
    
    # Compute the moving average of all individual losses and the total loss.
    loss_average = tf.train.ExponentialMovingAverage(0.999, name='moving_avg_xentropy')
    loss_average_op = loss_average.apply([loss, total_loss])
    
    # Return total loss
    with tf.control_dependencies([loss_average_op]):
      total_loss = tf.identity(total_loss)
    return total_loss
  
# Submodules
def _residual(inputs, 
              num_filters_out_1, kernel_size_1, stride_1,
              num_filters_out_2, kernel_size_2, stride_2,
              is_training=True, restore=True, scope=None, reuse=None):
  """Residual unit, with 2 stacked (3x3) conv layers
  
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_filters_out_1: the number of output filters for conv layer 1
    kernel_size_1: an int representing the square filter size for conv layer 1
    stride_1: an int representing stride in both x and y for conv layer 1
    num_filters_out_2: the number of output filters for conv layer 2
    kernel_size_2: an int representing the square filter size for conv layer 2
    stride_2: an int representing stride in both x and y for conv layer 2
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
  Returns:
    a tensor representing the output of the operation.

  """
  with tf.variable_scope(scope, 'residual_block', [inputs], reuse=reuse):
    x_orig = inputs
    
    # Convolution Layer 1
    x = _batch_norm(inputs, scope='batchnorm1', restore=restore)
    x = _relu(x, leakiness=0.0)
    x = _conv(x, num_filters_out_1, kernel_size_1, stride_1,
              scope='conv1', restore=restore)
    
    # Convolution Layer 2
    x = _batch_norm(x, scope='batchnorm2')
    x = _relu(x, leakiness=0.0)
    x = _conv(x, num_filters_out_2, kernel_size_2, stride_2,
              scope='conv2', restore=restore)
   
    # TODO: Handle dimension reduction & depth changes!
    # Reduce dimensions using strided avg_pool TODO: play with params
    reduction = stride_1 * stride_2
    x_orig = _avg_pool(x_orig, filter_size=reduction, stride=reduction)
    # Pad difference in depth with zeros
    filter_diff = tf.shape(x)[-1] - tf.shape(x_orig)[-1]
    padding = tf.zeros([tf.shape(x_orig)[0], tf.shape(x_orig)[1], 
                        tf.shape(x_orig)[2], filter_diff],
                       dtype=tf.float32)
    x_orig = tf.concat([x_orig, padding], 3)
    # Output: y = x + F(W, x)
    outputs = tf.add(x_orig, x)
  
    return outputs

def _batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               moving_vars='moving_vars',
               is_training=True,
               trainable=True,
               restore=True,
               scope=None,
               reuse=None):
  """Adds a Batch Normalization layer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels]
            or [batch_size, channels].
    decay: decay for the moving average.
    center: If True, subtract beta. If False, beta is not created and ignored.
    scale: If True, multiply by gamma. If False, gamma is
      not used. When the next layer is linear (also e.g. ReLU), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    moving_vars: collection to store the moving_mean and moving_variance.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.

  Returns:
    a Tensor representing the output of the operation.
  """
  inputs_shape = inputs.get_shape()
  with tf.variable_scope(scope, 'batchnorm', [inputs], reuse=reuse):
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    
    # Manage collections
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if restore:
      collections.append(VARIABLES_TO_RESTORE)
      
    if center:
      beta = tf.get_variable('beta',
                             shape=params_shape, dtype=tf.float32,
                             initializer=tf.zeros_initializer(),
                             trainable=trainable, collections=collections)
    if scale:
      gamma = tf.get_variable('gamma',
                              shape=params_shape, dtype=tf.float32,
                              initializer=tf.zeros_initializer(),
                              trainable=trainable, collections=collections)
    # Create moving_mean and moving_variance add them to
    # GraphKeys.MOVING_AVERAGE_VARIABLES collections.
    moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
    if restore:
      moving_collections.append(VARIABLES_TO_RESTORE)
    moving_mean = tf.get_variable('moving_mean',
                                  shape=params_shape, dtype=tf.float32,
                                  initializer=tf.zeros_initializer(),
                                  trainable=False,
                                  collections=moving_collections)
    moving_variance = tf.get_variable('moving_variance',
                                      shape=params_shape, dtype=tf.float32,
                                      initializer=tf.ones_initializer(),
                                      trainable=False,
                                      collections=moving_collections)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, variance = tf.nn.moments(inputs, axis)

      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean, mean, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance, variance, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    else:
      # Just use the moving_mean and moving_variance.
      mean = moving_mean
      variance = moving_variance
    # Normalize the activations.
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon)
    outputs.set_shape(inputs.get_shape())
    return outputs

def _relu(inputs, leakiness=0.0):
  """Relu, with optional (default) leaky support"""
  if leakiness == 0.0:
    return tf.nn.relu(inputs, name='relu')
  else:
    return tf.where(tf.less(inputs, 0.0), leakiness * inputs,
                  inputs, name='leaky_relu')
  # Alternatively:
  # return tf.maximum(leakiness * input, input)

# TODO: Change weight initializer and std_dev (see resnet_model from tf models)
def _conv(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           stddev=0.01,
           is_training=True,
           trainable=True,
           restore=True,
           scope=None,
           reuse=None):
  """Adds a 2D convolution followed by an optional batch_norm layer.

  _conv creates a variable called 'weights', representing the convolutional
  kernel, that is convolved with the input.

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
  with tf.variable_scope(scope, 'conv', [inputs], reuse=reuse):
    num_filters_in = inputs.get_shape()[-1]
    weights_shape = [kernel_size, kernel_size,
                     num_filters_in, num_filters_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = _l2_regularizer(WEIGHT_DECAY)
    
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if restore:
      collections.append(VARIABLES_TO_RESTORE)
    
    weights = tf.get_variable('weights', shape=weights_shape,
                              dtype=tf.float32, initializer=weights_initializer,
                              regularizer=l2_regularizer, trainable=trainable,
                              collections=collections)
    # Add convolution to graph -- y_int = (w*x)
    return tf.nn.conv2d(inputs, weights, [1, stride, stride, 1],
                        padding=padding)

def _avg_pool(inputs, filter_size, stride, name='avg_pool'):
  """ Average Pooling Layer """
  pool = tf.nn.avg_pool(inputs, ksize=[1, filter_size, filter_size, 1], 
                        strides=[1, stride, stride, 1],
                        padding='VALID', name=name)        
  return pool

  
# Utility functions
def _l2_regularizer(weight=1.0, scope=None):
  """Define a L2 regularizer.

  Args:
    weight: scale the loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    a regularizer function.
  """
  def regularizer(tensor):
    with tf.name_scope(scope, 'L2Regularizer', [tensor]):
      l2_weight = tf.convert_to_tensor(weight,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight')
      return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
  return regularizer
