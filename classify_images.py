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

# TODO: Import image classifiers!

parser = argparse.ArgumentParser(description='This program is designed to be run on the ground station side of the 2016-17 computer vision system. It continuously scans a directory for images and passes them to image classifier(s). Results are sent to the Interop Server')
parser.add_argument('-f', '--format', default='jpg', help='Input image format. Suggested formats are jpg or png')
parser.add_argument('-d', '--dir', help='Directory to scan for images. If no directory provided, scans current working directory')

args = parser.parse_args()

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

  # TODO: Build TensorFlow graphs

  print("INFO: Beginning infinite loop. To terminate, use Ctrl+C")
  while True:
    # Iterate through files in directory (NOTE: 'file' is a __builtin__)
    for f in os.listdir(directory):
      if f.lower().endswith(ext):
        # TODO: Load image from os.path.join(directory, f)
        # TODO: Run respective image classifiers
        print(os.path.join(directory, f)) # DELETEME: Debugging only!
        # TODO: Send data to interop server
        # TODO (optional): build database of detected targets, correct mistakes

        # Move image into processed_## subdir
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
          print('Debug')
          os.mkdir(os.path.join(directory, processedDir))
        # Move processed file to processed_##/ subdirectory
        os.rename(os.path.join(directory, f), os.path.join(directory, processedDir, f))

if __name__ == "__main__":
  print("Welcome to AZA's Image Classification Program")
  print("For options and more information, please rerun this program with the -h option")
  main()

