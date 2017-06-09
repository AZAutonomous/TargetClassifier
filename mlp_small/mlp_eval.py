"""Library to evaluate WideResNet on a single GPU."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time

import numpy as np
import tensorflow as tf

import image_processing
import mlp_model as mlp

FLAGS = tf.app.flags.FLAGS

# Eval directory/pathing flags
tf.app.flags.DEFINE_string('eval_dir', '/tmp/aza_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/aza_train',
                          """Directory where to read model checkpoints.""")

# Eval frequency flags
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Eval data flags
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
	"""Run Eval once.

	Args:
		saver: Saver.
		summary_writer: Summary writer.
		top_k_op: Top K op.
		summary_op: Summary op.
	"""
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint with absolute path.
			if os.path.isabs(ckpt.model_checkpoint_path):
				saver.restore(sess, ckpt.model_checkpoint_path)
			# Restores from checkpoint with relative path.
			else:
				saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
											ckpt.model_checkpoint_path))
			# Assuming model_checkpoint_path looks something like:
			#	 /my-favorite-path/imagenet_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return

		# Start the queue runners.
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
																	start=True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0	# Counts the number of correct predictions.
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			
			print('%s: Starting evaluation on subset: %s.' % (datetime.now(), FLAGS.subset))
			start_time = time.time()
			while step < num_iter and not coord.should_stop():
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1
				if step % 20 == 0:
					duration = time.time() - start_time
					sec_per_batch = duration / 20.0
					examples_per_sec = FLAGS.batch_size / sec_per_batch
					print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
					      'sec/batch)' % (datetime.now(), step, num_iter,
					      examples_per_sec, sec_per_batch))
					start_time = time.time()

			# Compute precision @ 1.
			precision = true_count / total_sample_count
			print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Precision @ 1', simple_value=precision)
			summary_writer.add_summary(summary, global_step)
			
		except Exception as e:	# pylint: disable=broad-except
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset, classname=None, scope=None):
	"""Evaluate model on Dataset for a number of steps."""
	with tf.Graph().as_default() as g:
		# Get images and labels from the dataset.
		images, labels = image_processing.inputs(dataset, classname=classname)

		# Build a Graph that computes the logits predictions from the
		# inference model.
		logits = mlp.inference(images, dataset.num_classes(), scope=scope)

		# Calculate predictions.
		top_k_op = tf.nn.in_top_k(logits, labels, 1)

		# Restore the moving average version of the learned variables for eval.
		variable_averages = tf.train.ExponentialMovingAverage(
								mlp.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		# Build the summary operation based on the TF collection of Summaries.
		summary_op = tf.summary.merge_all()

		summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

		while True:
			eval_once(saver, summary_writer, top_k_op, summary_op)
			if FLAGS.run_once:
				break
			time.sleep(FLAGS.eval_interval_secs)
