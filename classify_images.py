# classify_images.py
# Author: Arizona Autonomous Vehicles Club
# Task: AUVSI SUAS 2017 Image Classification
# Description: This script is the primary program for competition time.
#              Once run, it will loop forever until terminated manually,
#              e.g. with Ctrl+C or Ctrl+Z. The script continuously polls
#              its current directory (or optional provided directory) for
#              images of .jpg (or optional specified format) and classify
#              the image(s). Results will then be transmitted to the
#              interop server for scoring

import argparse
import os
import json

import numpy as np
import tensorflow as tf

import convnets.wideresnet.wideresnet_model as model

# TODO: Import image classifiers!

# Constants
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3

parser = argparse.ArgumentParser(description='This program is designed to be run on the ground station side of the 2016-17 computer vision system. It continuously scans a directory for images and passes them to image classifier(s). Results are sent to the Interop Server')
parser.add_argument('-f', '--format', default='jpg', help='Input image format. Suggested formats are jpg or png')
parser.add_argument('-d', '--dir', help='Directory to scan for images. If no directory provided, scans current working directory')

args = parser.parse_args()

# TODO: Look Up Tables for converting class indices to human readable strings
shapes = {}
colors = {}
alphanums = {}

# Special flag to track whether build_graphs() was called
graphs_built = False

# TODO
def preprocess_image(image):
	''' Preprocess image for classification
	    Args:
	        image: np.array of size [width, height, depth]
	    Returns:
		image: np.array of size [width, height, depth]
	'''
	# TODO: meanstd normalization
	pass

# TODO
def build_graphs():
    ''' Build the TensorFlow graphs as needed, store as global variables
    '''
    global logits_shape
    global inputs_shape
    inputs_shape = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    logits_shape = model.inference(inputs_shape, 14, scope='shape') # 13 shapes + background
    
    global logits_shape_color
    global inputs_shape_color
    inputs_shape_color = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    logits_shape_color = model.inference(inputs_shape_color, 11, scope='shape_color') # 10 colors + background
    
    global logits_alphanum
    global inputs_alphanum
    inputs_alphanum = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    logits_alphanum = model.inference(inputs_alphanum, 37, scope='alphanum') # 36 alphanums + background
    
    global logits_alphanum_color
    global inputs_alphanum_color
    inputs_alphanum_color = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    logits_alphanum_color = model.inference(inputs_alphanum_color, 11, scope='alphanum_color') # 10 colors + background
    
    global graphs_built
    graphs_built = True

# TODO
def classify_shape(image):
	''' Extract the shape of the target
			Args: The input image
			Returns:
				str: The classified shape, in human readable text
	'''
	# Create batch size dimension of 1
	input_image = np.expand_dims(image, axis=0)
	with tf.Session() as sess:
		predictions = sess.run([logits_shape], feed_dict={inputs_shape: input_image})
		class_out = np.argmax(predictions)
		confidence = np.max(x)
		# TODO: Do something with the confidence
		return shapes[class_out]
	return None
	pass

# TODO
def classify_shape_color(image):
	''' Extract the shape color of the target
			Args: The input image
			Returns:
				str: The classified color, in human readable text
	'''
	# Create batch size dimension of 1
	input_image = np.expand_dims(image, axis=0)
	with tf.Session() as sess:
		predictions = sess.run([logits_shape_color], feed_dict={inputs_shape_color: input_image})
		class_out = np.argmax(predictions)
		confidence = np.max(x)
		# TODO: Do something with the confidence
		return colors[class_out]
	return None
	pass

# TODO
def classify_letter(image):
	''' Extract the letter color of the target
			Args: The input image
			Returns: 
				str: The classified letter, in human readable text
				str: The orientation, in human readable text (e.g. "n")
	'''
	# NOTE: Consider extracting orientation here by running multiple
	#			 inferences with rotations of the image
	# Create batch size dimension of 1
	input_image = np.expand_dims(image, axis=0)
	with tf.Session() as sess:
		# TODO: Rotate input by some interval to detect orientation
		predictions = sess.run([logits_alphanum], feed_dict={inputs_alphanum: inputs_alphanum})
		class_out = np.argmax(predictions)
		confidence = np.max(x)
		# TODO: Do something with the confidence
		return alphanums[class_out]
	return None, None
	pass

# TODO
def classify_letter_color(image):
	''' Extract the letter color of the target
			Args: The input image
			Returns:
				str: The classified color, in human readable text
	'''
	# Create batch size dimension of 1
	input_image = np.expand_dims(image, axis=0)
	with tf.Session() as sess:
		predictions = sess.run([logits_alphanum_color], feed_dict={inputs_alphanum_color: input_image})
		class_out = np.argmax(predictions)
		confidence = np.max(x)
		# TODO: Do something with the confidence
		return colors[class_out]
	return None
	pass

# TODO
def extract_location(todo): # TODO: Decide on input format
	''' Extract the location of the target
			Args: TODO/TBD
			Returns:
				float32: latitude
				float32: longitude
	'''
	return None, None
	pass

def check_valid(packet):
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

def classify_and_maybe_transmit(image, location=None, orientation=None):
	''' Main worker function for image classification. Transmits depending on validity
		Args:
			image: np.array of size [width, height, depth]
			location: tuple of GPS coordinates as (lat, lon)
			orientation: degree value in range [-180, 180],
			             where 0 represents due north and 90 represents due east
	'''
        # Build graphs once
        if not graphs_built:
            build_graphs()

	# TODO: Run respective image classifiers
	shape = classify_shape(image)
	background_color = classify_shape_color(image)
	alphanumeric, orientation = classify_letter(image)
	alphanumeric_color = classify_letter_color(image)
	latitude, longitude = extract_location(f) # TODO - input arg?

	packet = {
			"user": 1, # TODO: What will our user id be?
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
	if check_valid(packet) is True:
		print('INFO: Transmitting target found in {}'.format(f))
		packet["id"] = target_id
		json_packet = json.dumps(packet)
		# TODO: Transmit data to interop server
		# TODO (optional): build database of detected targets, correct mistakes
		target_id += 1
	else:
		print('INFO: Processed invalid target in {}'.format(f))

def main():
	# Process command line args
	if args.dir is not None:
		 directory = args.dir
	else:
		 directory = os.getcwd()
	ext = '.' + args.format.split('.')[-1].lower()

	# Validate arguments
	assert os.path.exists(directory)

        # Build graphs
        build_graphs()

	print 'Running on directory:\t\t', directory
	print 'Searching for images of format:\t', ext

	print("INFO: Beginning infinite loop. To terminate, use Ctrl+C")
	target_id = 1 # initialize to 1
	while True:
		# Iterate through files in directory (NOTE: 'file' is a __builtin__)
		for f in os.listdir(directory):
			if f.lower().endswith(ext):
				# TODO: Load image from os.path.join(directory, f)
				image = 0 #TODO/FIXME: temporary value
				
				classify_and_maybe_transmit(image)
				
				# Move processed image into processed_## subdir
				counter = 0
				processedDir = 'processed_' + str(counter).zfill(2)
				# Increment counter until we find unused processed_##/file location
				while os.path.exists(os.path.join(directory, processedDir, f)):
					counter += 1
					processedDir = 'processed_' + str(counter).zfill(2)
				# NOTE: Program will continue to work after counter > 99, but naming
				#       convention will be violated (e.g. processed_101/foo.jpg)
				# Make subdirectories as necessary
				if not os.path.exists(os.path.join(directory, processedDir)):
					os.mkdir(os.path.join(directory, processedDir))
				# Move processed file to processed_##/ subdirectory
				os.rename(os.path.join(directory, f), os.path.join(directory, processedDir, f))

if __name__ == "__main__":
	print("Welcome to AZA's Image Classification Program")
	print("For options and more information, please rerun this program with the -h option")
	main()

