# classify_images.py
# Author: Arizona Autonomous Vehicles Club
# Task: AUVSI SUAS 2017 Image Classification
# Description: This script is the primary program for competition time.
#			  Once run, it will loop forever until terminated manually,
#			  e.g. with Ctrl+C or Ctrl+Z. The script continuously polls
#			  its current directory (or optional provided directory) for
#			  images of .jpg (or optional specified format) and classify
#			  the image(s). Results will then be transmitted to the
#			  interop server for scoring

import argparse
import os
import sys
import json

from multiprocessing import Pool
import numpy as np
import cv2
import tensorflow as tf

import convnets.wideresnet.wideresnet_model as model

# Constants
DEBUG = True
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3

class TargetClassifier():
	def __init__(self, userid, checkpoint_dir):
		# Store Look Up Tables
		self.shapes = {0 : 'n/a', 1 : 'circle', 2 : 'cross', 3 : 'heptagon', 4 : 'hexagon', 5 : 'octagon', 6 : 'pentagon', 7 : 'quartercircle', 8 : 'rectangle', 9 : 'semicircle', 10 : 'square', 11 : 'star', 12 : 'trapezoid', 13 : 'triangle'}
		self.alphanums = {0 : 'n/a',  1 : 'A',  2 : 'B',  3 : 'C',  4 : 'D',  5 : 'E',  6 : 'F',  7 : 'G',  8 : 'H',  9 : 'I',  10 : 'J', 11 : 'K', 12 : 'L', 13 : 'M', 14 : 'N', 15 : 'O', 16 : 'P', 17 : 'Q', 18 : 'R', 19 : 'S', 20 : 'T', 21 : 'U', 22 : 'V', 23 : 'W', 24 : 'X', 25 : 'Y', 26 : 'Z', 27 : '0', 28 : '1', 29 : '2', 30 : '3', 31 : '4', 32 : '5', 33 : '6', 34 : '7', 35 : '8', 36 : '9'}
		self.colors = {0 : 'n/a', 1 : 'white', 2 : 'black', 3 : 'gray', 4 : 'red', 5 : 'blue', 6 : 'green', 7 : 'yellow', 8 : 'purple', 9 : 'brown', 10 : 'orange'}
		self.orientation = {0 : 'n/a', 1 : 'N', 2 : 'NE', 3 : 'E', 4 : 'SE', 5 : 'S', 6 : 'SW', 7 : 'W', 8 : 'NW'}

		# Store userid
		self.userid = userid

		# IMPORTANT! Put updated mean standard values here (TODO)
		self.mean = np.array([62.026, 128.710, 131.434]) # R, G, B
		self.stddev = np.array([53.492, 50.432, 52.819]) # R, G, B
		
		# Counters/trackers for interop
		self.target_id = 0

		# Build TensorFlow graphs
		assert os.path.isdir(checkpoint_dir)
		# Shape graph
		self.shape_graph = tf.Graph()
		with self.shape_graph.as_default():
			self.inputs_shape = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
			self.logits_shape = model.inference(self.inputs_shape, 14, scope='shapes') # 13 shapes + background
			variable_averages = tf.train.ExponentialMovingAverage(
									model.MOVING_AVERAGE_DECAY)
			variables_to_restore = variable_averages.variables_to_restore()
			saver = tf.train.Saver(variables_to_restore)
			
			self.shape_sess = tf.Session() # graph=self.shape_graph
			#shape_saver = tf.train.Saver()
			shape_ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, 'shape'))
			if shape_ckpt and shape_ckpt.model_checkpoint_path:
				print('Reading shape model parameters from %s' % shape_ckpt.model_checkpoint_path)
				#shape_saver.restore(self.shape_sess, self.shape_ckpt.model_checkpoint_path)
				saver.restore(self.shape_sess, shape_ckpt.model_checkpoint_path)
			else:
				print('Error restoring parameters for shape. Ensure checkpoint is stored in ${checkpoint_dir}/shape/')
				# sys.exit(1)
	
		# Shape color graph
		self.shape_color_graph = tf.Graph()
		with self.shape_color_graph.as_default():
			self.inputs_shape_color = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
			self.logits_shape_color = model.inference(self.inputs_shape_color, 11, scope='shape_color') # 10 shape_colors + background
			variable_averages = tf.train.ExponentialMovingAverage(
									model.MOVING_AVERAGE_DECAY)
			variables_to_restore = variable_averages.variables_to_restore()
			saver = tf.train.Saver(variables_to_restore)
			
			self.shape_color_sess = tf.Session() # graph=self.shape_color_graph
			#shape_color_saver = tf.train.Saver()
			shape_color_ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, 'shape_color'))
			if shape_color_ckpt and shape_color_ckpt.model_checkpoint_path:
				print('Reading shape_color model parameters from %s' % shape_color_ckpt.model_checkpoint_path)
				#shape_color_saver.restore(self.shape_color_sess, self.shape_color_ckpt.model_checkpoint_path)
				saver.restore(self.shape_color_sess, shape_color_ckpt.model_checkpoint_path)
			else:
				print('Error restoring parameters for shape_color. Ensure checkpoint is stored in ${checkpoint_dir}/shape_color/')
				# sys.exit(1)
	
		# Alphanum graph
		self.alphanum_graph = tf.Graph()
		with self.alphanum_graph.as_default():
			self.inputs_alphanum = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
			self.logits_alphanum = model.inference(self.inputs_alphanum, 37, scope='alphanum') # 37 alphanums + background
			variable_averages = tf.train.ExponentialMovingAverage(
									model.MOVING_AVERAGE_DECAY)
			variables_to_restore = variable_averages.variables_to_restore()
			saver = tf.train.Saver(variables_to_restore)
			
			self.alphanum_sess = tf.Session()
			#alphanum_saver = tf.train.Saver()
			alphanum_ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, 'alphanum'))
			if alphanum_ckpt and alphanum_ckpt.model_checkpoint_path:
				print('Reading alphanum model parameters from %s' % alphanum_ckpt.model_checkpoint_path)
				#alphanum_saver.restore(self.alphanum_sess, self.alphanum_ckpt.model_checkpoint_path)
				saver.restore(self.alphanum_sess, alphanum_ckpt.model_checkpoint_path)
			else:
				print('Error restoring parameters for alphanum. Ensure checkpoint is stored in ${checkpoint_dir}/alphanum/')
				# sys.exit(1)
	
		# Alphanum color graph
		self.alphanum_color_graph = tf.Graph()
		with self.alphanum_color_graph.as_default():
			self.inputs_alphanum_color = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
			self.logits_alphanum_color = model.inference(self.inputs_alphanum_color, 11, scope='alphanum_color') # 10 alphanum_colors + background
			variable_averages = tf.train.ExponentialMovingAverage(
									model.MOVING_AVERAGE_DECAY)
			variables_to_restore = variable_averages.variables_to_restore()
			saver = tf.train.Saver(variables_to_restore)
			
			self.alphanum_color_sess = tf.Session()
			#alphanum_color_saver = tf.train.Saver()
			alphanum_color_ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, 'alphanum_color'))
			if alphanum_color_ckpt and alphanum_color_ckpt.model_checkpoint_path:
				print('Reading alphanum_color model parameters from %s' % alphanum_color_ckpt.model_checkpoint_path)
				#alphanum_color_saver.restore(self.alphanum_color_sess, self.alphanum_color_ckpt.model_checkpoint_path)
				saver.restore(self.alphanum_color_sess, alphanum_color_ckpt.model_checkpoint_path)
			else:
				print('Error restoring parameters for alphanum_color. Ensure checkpoint is stored in ${checkpoint_dir}/alphanum_color/')
				# sys.exit(1)

	def preprocess_image(self, image):
		''' Preprocess image for classification
			Args:
				image: np.array containing raw input image
			Returns:
				image: np.array of size [1, width, height, depth]
		'''
		im = image.copy()

		# Change from BGR (OpenCV) to RGB
		b = im[:,:,0].copy()
		im[:,:,0] = im[:,:,2] # Put red channel in [:,:,0]
		im[:,:,2] = b # Put blue channel in [:,:,2]

		# Resize image as necessary
		if (np.greater(im.shape[:2], [IMAGE_SIZE, IMAGE_SIZE]).any()):
			# Scale down
			im = cv2.resize(im, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
		elif (np.less(im.shape[:2], [IMAGE_SIZE, IMAGE_SIZE]).any()):
			# Scale up
			im = cv2.resize(im, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

		# MeanStd normalization
		im = np.subtract(im, self.mean)
		im = np.divide(im, self.stddev)
		im = np.expand_dims(im, axis=0)
		return im

	def classify_shape(self, image):
		''' Extract the shape of the target
				Args: The preprocessed input image, of shape 
			Returns:
				str: The classified shape, in human readable text
		'''
		try:
			predictions = self.shape_sess.run([self.logits_shape],
			                                  feed_dict={self.inputs_shape: image})
			class_out = np.argmax(predictions)
			confidence = np.max(predictions)
			# TODO: Do something with the confidence
			return self.shapes[class_out]
		# If checkpoint not loaded, ignore error and return None
		except tf.errors.FailedPreconditionError:
			return None
	
	def classify_shape_color(self, image):
		''' Extract the shape color of the target
				Args: The input image
				Returns:
					str: The classified color, in human readable text
		'''
		try:
			predictions = self.shape_color_sess.run([self.logits_shape_color],
				                                 feed_dict={self.inputs_shape_color: image})
			class_out = np.argmax(predictions)
			confidence = np.max(predictions)
			# TODO: Do something with the confidence
			return self.colors[class_out]
		# If checkpoint not loaded, ignore error and return None
		except tf.errors.FailedPreconditionError:
			return None
	
	def classify_letter(self, image):
		''' Extract the letter color of the target
				Args: The input image
				Returns: 
					str: The classified letter, in human readable text
					str: Amount rotated clockwise, in degrees (int)
		'''
		try:
			# TODO: Rotate input by some interval to detect orientation
			rot = 0
			class_out_dict = {}
			while (rot < 360):
				# TODO: Rotate image clockwise by rot degrees
				predictions = self.alphanum_sess.run([self.logits_alphanum],
				                                feed_dict={self.inputs_alphanum: image})
				class_out_dict[np.max(predictions)] = np.argmax(predictions)
				rot += 45 # 45 degree stride. If computation budget allows, consider increasing to 22.5 deg
			confidence = max(class_out_dict) # Maximum confidence from classifications
			class_out = np.argmax(predictions)
			# TODO: Do something with the confidence
			return self.alphanums[class_out], rot
		# If checkpoint not loaded, ignore error and return None
		except tf.errors.FailedPreconditionError:
			return None, None
	
	def classify_letter_color(self, image):
		''' Extract the letter color of the target
				Args: The input image
				Returns:
					str: The classified color, in human readable text
		'''
		try:
			predictions = self.alphanum_color_sess.run([self.logits_alphanum_color],
			                                     feed_dict={self.inputs_alphanum_color: image})
			class_out = np.argmax(predictions)
			confidence = np.max(predictions)
			# TODO: Do something with the confidence
			return self.colors[class_out]
		# If checkpoint not loaded, ignore error and return None
		except tf.errors.FailedPreconditionError:
			return None

	# TODO
	def extract_colors(self, image):
		''' Extract color data from image using clustering algorithm
				Args:
				  image: input image (np.array)
				Returns:
				  background_color: a string representing the target background color (i.e. shape color)
					alphanum_color: a string representing the target alphanumeric color (i.e. letter color)
		'''
		pass

	def check_valid(self, packet):
		''' Check whether the prepared output packet is valid
				Args:
					dict: dictionary (JSON) of proposed output packet
				Returns:
					bool: True if packet is valid, False if not
		'''
		for key, value in packet.iteritems():
			# Background class, flagged "n/a" in our translation key
			if (value is None or value == "n/a") and key is not "description":
				return False
			# Background and alphanumeric color should never be the same
			if packet['background_color'] == packet['alphanumeric_color']:
				return False
			# TODO: Check for valid lat/lon
	
			return True
	
	def classify_and_maybe_transmit(self, image, location=(None, None), orientation=None):
		''' Main worker function for image classification. Transmits depending on validity
			Args:
				image: np.array of size [width, height, depth]
				location: tuple of GPS coordinates as (lat, lon)
				orientation: degree value in range [-180, 180],
							 where 0 represents due north and 90 represents due east
		'''
		image = self.preprocess_image(image)
		
		# Set up multiprocessing to asynchronously do stuff on CPU
		pool = Pool()
		res = pool.apply_async(self.extract_colors, image)

		# Run respective image classifiers
		shape = self.classify_shape(image)
		#background_color = self.classify_shape_color(image)
		alphanumeric, rot = self.classify_letter(image)
		#alphanumeric_color = self.classify_letter_color(image)
		latitude, longitude = location
		# TODO: Get orientation using orientation_in + rot

		res.wait()
		background_color, alphanumeric_color = res.get(timeout=3)

		if DEBUG:
			print 'Shape =', shape
			print 'Shape Color =', background_color
			print 'Alphanumeric =', alphanumeric
			print 'Alphanum Color =', alphanumeric_color
			print 'Lat, Lon =', latitude, ',', longitude
	
		packet = {
				"user": self.userid, # TODO: What will our user id be?
				"type": "standard",
				"latitude": latitude,
				"longitude": longitude,
				"orientation": orientation,
				"shape": shape,
				"background_color": background_color,
				"alphanumeric": alphanumeric,
				"alphanumeric_color": alphanumeric_color,
				"description": None,
				"autonomous": True
			}
	
		# Check for false positives or otherwise invalid targets
		if self.check_valid(packet):
			print('INFO: Transmitting target %d' % self.target_id)
			packet["id"] = self.target_id
			json_packet = json.dumps(packet)
			# TODO: Transmit data to interop server
			# TODO (optional): build database of detected targets, correct mistakes
			self.target_id += 1
			return True
		else:
			print('INFO: An invalid target was discarded')
			return False


def main():
	# Create command line args
	parser = argparse.ArgumentParser(
						description='This program is to be run on the ground station '
									'side of the 2016-17 computer vision system. It'
									'continuously scans a directory for images and passes'
									'them to image classifier(s). Results are sent to the'
									'Interop Server')
	parser.add_argument('-u', '--userid', default='azautonomous',
							help='User ID for Interop Server.')
	parser.add_argument('-f', '--format', default='jpg', 
							help='Input image format. Suggested formats are jpg or png')
	parser.add_argument('-d', '--dir', 
							help='Directory to scan for images. If no directory provided, '
									'scans current working directory')
	parser.add_argument('-c', '--checkpoint_dir', required=True, 
								help='Path to checkpoint directories. '
								'Each classifier should be kept in a separate directory '
								'according to their name (e.g. scope). For example, '
								'checkpoints/ with subdirectories shape/, alphanum/, etc')
	
	args = parser.parse_args()

	# Process command line args
	if args.dir is not None:
		 directory = args.dir
	else:
		 directory = os.getcwd()
	ext = '.' + args.format.split('.')[-1].lower()

	# Validate arguments
	assert os.path.exists(directory)

	# Initialize classifiers
	classifier = TargetClassifier(args.userid, args.checkpoint_dir)

	print 'Running on directory:\t\t', directory
	print 'Searching for images of format:\t', ext

	print("INFO: Beginning infinite loop. To terminate, use Ctrl+C")
	while True:
		# Iterate through files in directory (NOTE: 'file' is a __builtin__)
		for f in os.listdir(directory):
			if f.lower().endswith(ext):
				image = cv2.imread(os.path.join(directory, f))
				classifier.classify_and_maybe_transmit(image)
				# Move processed image into processed_## subdir
				counter = 0
				processedDir = 'processed_' + str(counter).zfill(2)
				# Increment counter until we find unused processed_##/file location
				while os.path.exists(os.path.join(directory, processedDir, f)):
					counter += 1
					processedDir = 'processed_' + str(counter).zfill(2)
				# NOTE: Program will continue to work after counter > 99, but naming
				#	   convention will be violated (e.g. processed_101/foo.jpg)
				# Make subdirectories as necessary
				if not os.path.exists(os.path.join(directory, processedDir)):
					os.mkdir(os.path.join(directory, processedDir))
				# Move processed file to processed_##/ subdirectory
				os.rename(os.path.join(directory, f), os.path.join(directory, processedDir, f))

if __name__ == "__main__":
	print("Welcome to AZA's Image Classification Program")
	print("For options and more information, please rerun this program with the -h option")
	main()

