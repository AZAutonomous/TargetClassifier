# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read and preprocess image data.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
	 of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
							"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 32,
							"""Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
							"""Number of preprocessing threads. """
							"""Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
							"""Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing. A larger shuffling queue guarantees better mixing across 
# examples within a batch and results in slightly higher predictive performance 
# in a trained model. Empirically, --input_queue_memory_factor=16 works well. 
# A value of 192 implies a queue size of 1024*192 images. Assuming RGB 32x32 
# images, this implies a queue size of 2.25GB. If the machine is memory limited,
# then decrease this factor to decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 192,
							"""Size of the queue of preprocessed images. """)

def inputs(dataset, batch_size=None, num_preprocess_threads=None):
	"""Generate batches of CIFAR images for evaluation.

	Use this function as the inputs for evaluating a network.

	Note that some (minimal) image preprocessing occurs during evaluation
	including central cropping and resizing of the image to fit the network.

	Args:
		dataset: instance of Dataset class specifying the dataset.
		batch_size: integer, number of examples in batch
		num_preprocess_threads: integer, total number of preprocessing threads but
			None defaults to FLAGS.num_preprocess_threads.

	Returns:
		images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
																			 image_size, 3].
		labels: 1-D integer Tensor of [FLAGS.batch_size].
	"""
	if not batch_size:
		batch_size = FLAGS.batch_size

	# Force all input processing onto CPU in order to reserve the GPU for
	# the forward inference and back-propagation.
	with tf.device('/cpu:0'):
		images, labels = batch_inputs(
					dataset, batch_size, train=False,
					num_preprocess_threads=num_preprocess_threads,
					num_readers=1)

	return images, labels


def distorted_inputs(dataset, batch_size=None, num_preprocess_threads=None):
	"""Generate batches of distorted versions of ImageNet images.

	Use this function as the inputs for training a network.

	Distorting images provides a useful technique for augmenting the data
	set during training in order to make the network invariant to aspects
	of the image that do not effect the label.

	Args:
		dataset: instance of Dataset class specifying the dataset.
		batch_size: integer, number of examples in batch
		num_preprocess_threads: integer, total number of preprocessing threads but
			None defaults to FLAGS.num_preprocess_threads.

	Returns:
		images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
																			 FLAGS.image_size, 3].
		labels: 1-D integer Tensor of [batch_size].
	"""
	if not batch_size:
		batch_size = FLAGS.batch_size

	# Force all input processing onto CPU in order to reserve the GPU for
	# the forward inference and back-propagation.
	with tf.device('/cpu:0'):
		images, labels = batch_inputs(
				dataset, batch_size, train=True,
				num_preprocess_threads=num_preprocess_threads,
				num_readers=FLAGS.num_readers)
	return images, labels
	
def decode_jpeg(image_buffer, scope=None):
	"""Decode a JPEG string into one 3-D float image Tensor.

	Args:
		image_buffer: scalar string Tensor.
		scope: Optional scope for name_scope.
	Returns:
		3-D float Tensor with values ranging from [0, 255).
	"""
	with tf.name_scope(values=[image_buffer], name=scope,
								default_name='decode_jpeg'):
		# Decode the string as an RGB JPEG.
		# Note that the resulting image contains an unknown height and width
		# that is set dynamically by decode_jpeg. In other words, the height
		# and width of image is unknown at compile-time.
		image = tf.image.decode_jpeg(image_buffer, channels=3)
		
		# Convert dataset to floats. Does NOT scale to [0, 1)!
		image = tf.cast(image, tf.float32)
		return image
	
# TODO: Remove bounding box
def distort_image(image, height, width, bbox, thread_id=0, scope=None):
	"""Distort one image for training a network.

	Distorting images provides a useful technique for augmenting the data
	set during training in order to make the network invariant to aspects
	of the image that do not effect the label.

	Args:
		image: 3-D float Tensor of image
		height: integer
		width: integer
		bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
			where each coordinate is [0, 1) and the coordinates are arranged
			as [ymin, xmin, ymax, xmax].
		thread_id: integer indicating the preprocessing thread.
		scope: Optional scope for name_scope.
	Returns:
		3-D float Tensor of distorted image used for training.
	"""
	with tf.name_scope(values=[image, height, width, bbox], name=scope,
										 default_name='distort_image'):
		# Each bounding box has shape [1, num_boxes, box coords] and
		# the coordinates are ordered [ymin, xmin, ymax, xmax].

		# Display the bounding box in the first thread only.
		if not thread_id:
			image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
			tf.summary.image('image_with_bounding_boxes', image_with_box)

	# A large fraction of image datasets contain a human-annotated bounding
	# box delineating the region of the image containing the object of interest.
	# We choose to create a new bounding box for the object which is a randomly
	# distorted version of the human-annotated bounding box that obeys an allowed
	# range of aspect ratios, sizes and overlap with the human-annotated
	# bounding box. If no box is supplied, then we assume the bounding box is
	# the entire image.
		sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
				tf.shape(image),
				bounding_boxes=bbox,
				min_object_covered=0.1,
				aspect_ratio_range=[0.75, 1.33],
				area_range=[0.05, 1.0],
				max_attempts=100,
				use_image_if_no_bounding_boxes=True)
		bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
		if not thread_id:
			image_with_distorted_box = tf.image.draw_bounding_boxes(
					tf.expand_dims(image, 0), distort_bbox)
			tf.summary.image('images_with_distorted_bounding_box',
											 image_with_distorted_box)

		# Crop the image to the specified bounding box.
		distorted_image = tf.slice(image, bbox_begin, bbox_size)

		# This resizing operation may distort the images because the aspect
		# ratio is not respected. We select a resize method in a round robin
		# fashion based on the thread number.
		# Note that ResizeMethod contains 4 enumerated resizing methods.
		resize_method = thread_id % 4
		distorted_image = tf.image.resize_images(distorted_image, [height, width],
																method=resize_method)
		# Restore the shape since the dynamic slice based upon the bbox_size loses
		# the third dimension.
		distorted_image.set_shape([height, width, 3])
		if not thread_id:
			tf.summary.image('cropped_resized_image',
								tf.expand_dims(distorted_image, 0))

		# Randomly flip the image horizontally.
		distorted_image = tf.image.random_flip_left_right(distorted_image)

		# Randomly distort the colors.
		distorted_image = distort_color(distorted_image, thread_id)

		if not thread_id:
			tf.summary.image('final_distorted_image',
								tf.expand_dims(distorted_image, 0))
		return distorted_image

def mean_std(image, meanstd, scope=None):
	"""Perform mean-std normalization on images

	Args:
		images: 3-D Tensor containing all images from a dataset.
	meanstd: 2-D array containing mean and std dev in format
					 [[r_mean, g_mean, b_mean],
			 [r_std, g_std, b_std]]
		scope: Optional scope for name_scope.
	Returns:
		a 3-D Tensor containing meanstd normalized images
	"""
	with tf.name_scope(values=[images], name=scope,
						default_name='calc_mean_std'):
		# Calculate meanstd
		mean = tf.constant(meanstd[0], tf.float32)
		stddev = tf.constant(meanstd[1], tf.float32)
		
		# Perform normalization
		meansub = tf.subtract(image, mean)
		normalized = tf.divide(meansub, stddev)
		
		return normalized
	
	
def preprocess_image(image_buffer, meanstd, train):
	"""Contruct batches of training or evaluation examples from the image dataset.

	Args:
		image_buffer: 3-D Tensor containing a single image
		meanstd: 2-D array containing mean and std dev in format
		         [[r_mean, g_mean, b_mean],
		          [r_std, g_std, b_std]]
		train: boolean

	Returns:
		images: 3-D float Tensor of a preprocessed image
	"""
	image_size = FLAGS.image_size
	assert image.get_shape() == [image_size, image_size, 3]
	
	image = decode_jpeg(image_buffer)
	
	# meanstd normalization against entire dataset
	preprocessed_image = mean_std(image, meanstd)
	
	if train:
		# TODO: Rotations!
	
		# randomly horizontal flip
		preprocessed_image = tf.image.random_flip_left_right(preprocessed_image)
	
		# zero pad by 4 pixels on all sides
		preprocessed_image = tf.image.resize_image_with_crop_or_pad(
														preprocessed_image,
														image_size + 2 * 4,
														image_size + 2 * 4)

		# randomly crop back to 32x32
		preprocessed_image = tf.random_crop(preprocessed_image,
												[image_size, image_size, 3])

	return preprocessed_image
	
def parse_example_proto(example_serialized):
	"""Parses an Example proto containing a training example of an image.

	The output of the build_image_data.py image preprocessing script is a dataset
	containing serialized Example protocol buffers. Each Example proto contains
	the following fields:

		image/height: 32
		image/width: 32
		image/colorspace: 'RGB'
		image/channels: 3
		image/class/label: 3
		image/class/text: 'knee pad'
		image/format: 'JPEG'
		image/filename: 'ILSVRC2012_val_00041207.JPEG'
		image/encoded: <JPEG encoded string>

	Args:
		example_serialized: scalar Tensor tf.string containing a serialized
			Example protocol buffer.

	Returns:
		image_buffer: Tensor tf.string containing the contents of a JPEG file.
		label: Tensor tf.int32 containing the label.
		text: Tensor tf.string containing the human-readable label.
	"""
	# Dense features in Example proto.
	feature_map = {
			'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
															default_value=''),
			'image/class/label': tf.FixedLenFeature([1], dtype=tf.int32,
															default_value=-1),
			'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
															default_value=''),
	}
	sparse_float32 = tf.VarLenFeature(dtype=tf.float32)

	features = tf.parse_single_example(example_serialized, feature_map)
	label = tf.cast(features['image/class/label'], dtype=tf.int32)

	return features['image/encoded'], label, bbox, features['image/class/text']

def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None,
                 num_readers=1):
  """Contruct batches of training or evaluation examples from the image dataset.

  Args:
    dataset: instance of Dataset class specifying the dataset.
      See dataset.py for details.
    batch_size: integer
    train: boolean
    num_preprocess_threads: integer, total number of preprocessing threads
    num_readers: integer, number of parallel readers

  Returns:
    images: 4-D float Tensor of a batch of images
    labels: 1-D integer Tensor of [batch_size].

  Raises:
    ValueError: if data is not found
  """
  with tf.name_scope('batch_processing'):
    data_files = dataset.data_files()
    if data_files is None:
      raise ValueError('No data files found for this dataset')

    # Create filename_queue
    if train:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=True,
                                                      capacity=16)
    else:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=False,
                                                      capacity=1)
    if num_preprocess_threads is None:
      num_preprocess_threads = FLAGS.num_preprocess_threads

    if num_preprocess_threads % 4:
      raise ValueError('Please make num_preprocess_threads a multiple '
                       'of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers is None:
      num_readers = FLAGS.num_readers

    if num_readers < 1:
      raise ValueError('Please make num_readers at least 1')

    # Approximate number of examples per shard.
    examples_per_shard = 1024
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 32*32*3*4 bytes = 12 KB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 12KB = 192 MB
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    if train:
      examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string])
    else:
      examples_queue = tf.FIFOQueue(
          capacity=examples_per_shard + 3 * batch_size,
          dtypes=[tf.string])

    # Create multiple readers to populate the queue of examples.
    if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
        reader = dataset.reader()
        _, value = reader.read(filename_queue)
        enqueue_ops.append(examples_queue.enqueue([value]))

      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
    else:
      reader = dataset.reader()
      _, example_serialized = reader.read(filename_queue)

    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata.
      image_buffer, label_index, _ = parse_example_proto(example_serialized)
	  image = preprocess_image(image_buffer, dataset.meanstd(), train)
      images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=batch_size,
        capacity=2 * num_preprocess_threads * batch_size)

    # Reshape images into these desired dimensions.
    height = FLAGS.image_size
    width = FLAGS.image_size
    depth = 3

    images = tf.cast(images, tf.float32) # FIXME: This cast is redundant
    images = tf.reshape(images, shape=[batch_size, height, width, depth])

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_index_batch, [batch_size])
