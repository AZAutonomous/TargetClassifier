# File: mlp_train.py
# Author: Arizona Autonomous
# Description: This contains the trainer for the WideResNet model.
#							All code has been simplified to use only 1 GPU

""" Multilayer Perceptron Trainer """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf

import image_processing
import mlp_model as mlp

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/aza_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('checkpoint_path', '',
                           """If specified, restore this model """
                           """before beginning any training.""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 60.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.2,
                          """Learning rate decay factor.""")

# Constants for learning
NESTEROV_MOMENTUM = 0.9

def train(dataset, classname=None, preserve_view=False, scope=None):
	"""Train on dataset for a number of steps."""
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		# Create a variable to count the number of train() calls. This equals the
		# number of batches processed * FLAGS.num_gpus.
		global_step = tf.get_variable(
				'global_step', [],
				initializer=tf.constant_initializer(0), trainable=False)

		# Calculate the learning rate schedule.
		num_batches_per_epoch = (dataset.num_examples_per_epoch() /
														FLAGS.batch_size)
		decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
		lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
										global_step,
										decay_steps,
										FLAGS.learning_rate_decay_factor,
										staircase=True)


		# Create an optimizer that performs gradient descent.
		opt = tf.train.MomentumOptimizer(lr, momentum=NESTEROV_MOMENTUM,
											use_nesterov=True)

		images, labels = image_processing.distorted_inputs(dataset, classname=classname, preserve_view=preserve_view)

		input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

		# Number of classes in the Dataset label.
		num_classes = dataset.num_classes()
		
		# When fine-tuning model, do not restore logits and randomly initialize
		restore_logits = not FLAGS.fine_tune

		# Calculate the gradients for each model tower.
		with tf.device('/gpu:0'):
			with tf.variable_scope(tf.get_variable_scope()):
				logits = mlp.inference(images, num_classes, for_training=True,
												restore_logits=restore_logits, scope=scope)
				loss = mlp.loss(logits, labels, batch_size=FLAGS.batch_size, scope=scope)

				# Retain the Batch Normalization updates operations.
				batchnorm_updates = tf.get_collection(mlp.UPDATE_OPS_COLLECTION,
														scope)

				# Calculate the gradients for the batch of data
				grads = opt.compute_gradients(loss)
				
		# Add a summary to track the learning rate.
		tf.summary.scalar('learning_rate', lr)

		# Add histograms for gradients.
		for grad, var in grads:
			if grad is not None:
				tf.summary.histogram(var.op.name + '/gradients', grad)

		# Apply the gradients to adjust the shared variables.
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

		# Add histograms for trainable variables.
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)

		# Track the moving averages of all trainable variables.
		# Note that we maintain a "double-average" of the BatchNormalization
		# global statistics. This is more complicated then need be but we employ
		# this for backward-compatibility with inception models (original source).
		variable_averages = tf.train.ExponentialMovingAverage(
				mlp.MOVING_AVERAGE_DECAY, global_step)
		variables_to_average = (tf.trainable_variables() +
								tf.moving_average_variables())
		variables_averages_op = variable_averages.apply(variables_to_average)

		# Group all updates to into a single train op.
		batchnorm_updates_op = tf.group(*batchnorm_updates)
		train_op = tf.group(apply_gradient_op, variables_averages_op,
												batchnorm_updates_op)

		# Create a saver.
		saver = tf.train.Saver(tf.global_variables())

		# Build the summary operation.
		summary_op = tf.summary.merge_all()

		# Build an initialization operation to run below.
		init = tf.global_variables_initializer()

		# Start running operations on the Graph. allow_soft_placement allows
		# support for ops without GPU support
		sess = tf.Session(config=tf.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=FLAGS.log_device_placement))
		sess.run(init)

		if FLAGS.checkpoint_path:
			variables_to_restore = tf.get_collection(
					mlp.VARIABLES_TO_RESTORE)
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, FLAGS.checkpoint_path)
			print('%s: Pre-trained model restored from %s' %
					(datetime.now(), FLAGS.checkpoint_path))

		# Start the queue runners.
		tf.train.start_queue_runners(sess=sess)

		summary_writer = tf.summary.FileWriter(
				FLAGS.train_dir,
				graph=sess.graph)

		for step in range(FLAGS.max_steps):
			start_time = time.time()
			_, loss_value = sess.run([train_op, loss])
			duration = time.time() - start_time

			assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

			if step % 10 == 0:
				examples_per_sec = FLAGS.batch_size / float(duration)
				format_str = ('%s: step %d (epoch %d), loss = %.2f (%.1f examples/sec; %.3f '
											'sec/batch)')
				print(format_str % (datetime.now(), step, int(step/num_batches_per_epoch),
                                            loss_value,	examples_per_sec, duration))

			if step % 100 == 0:
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)

			# Save the model checkpoint periodically.
			if step % 500 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)
