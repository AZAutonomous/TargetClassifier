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

# TODO: Import image classifiers!

parser = argparse.ArgumentParser(description='This program is designed to be run on the ground station side of the 2016-17 computer vision system. It continuously scans a directory for images and passes them to image classifier(s). Results are sent to the Interop Server')
parser.add_argument('-f', '--format', default='jpg', help='Input image format. Suggested formats are jpg or png')
parser.add_argument('-d', '--dir', help='Directory to scan for images. If no directory provided, scans current working directory')

args = parser.parse_args()

# TODO
def build_graphs():
  ''' Build the TensorFlow graphs as needed
  '''
  pass

# TODO
def classify_shape(image):
  ''' Extract the shape of the target
      Args: The input image
      Returns:
        str: The classified shape, in human readable text
  '''
  return None
  pass

# TODO
def classify_shape_color(image):
  ''' Extract the shape color of the target
      Args: The input image
      Returns:
        str: The classified color, in human readable text
  '''
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
  #       inferences with rotations of the image
  return None, None
  pass

# TODO
def classify_letter_color(image):
  ''' Extract the letter color of the target
      Args: The input image
      Returns:
        str: The classified color, in human readable text
  '''
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

def main():
  # Process command line args
  if args.dir is not None:
     directory = args.dir
  else:
     directory = os.getcwd()
  ext = '.' + args.format.split('.')[-1].lower()

  # Validate arguments
  assert os.path.exists(directory)

  print 'Running on directory:\t\t', directory
  print 'Searching for images of format:\t', ext

  build_graphs()

  print("INFO: Beginning infinite loop. To terminate, use Ctrl+C")
  target_id = 1 # initialize to 1
  while True:
    # Iterate through files in directory (NOTE: 'file' is a __builtin__)
    for f in os.listdir(directory):
      if f.lower().endswith(ext):
        # TODO: Load image from os.path.join(directory, f)
        image = 0 # TODO

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

