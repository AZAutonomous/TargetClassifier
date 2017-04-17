# meanstd.py
# Author: dhung
# Description: Helper script to calculate the meanstd of all 
#              .JPG images in a directory

import argparse
import glob
import os

import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Helper script to calculate the meanstd of all JPG images in a directory')
parser.add_argument('-f', '--format', default='jpg', help='Input image format. Suggested formats are jpg or png')
parser.add_argument('-d', '--dir', help='Directory to scan for images. If no directory provided, scans current working directory')

args = parser.parse_args()

def main():
	# Further process args
	if args.dir is None:
		searchdir = os.getcwd()
	else:
		searchdir = args.dir
	fileext = args.format.split('.')[-1] # Strip extra periods and stuff
	assert(os.path.isdir(searchdir))

	arr = []
	searchpath = os.path.join(searchdir, '*.' + fileext)
	for f in glob.iglob(searchpath):
		im = cv2.imread(f)
		arr.append(im)

	nparr = np.array(arr)
	mean = np.mean(nparr, axis=(0,1,2))
	std = np.std(nparr, axis=(0,1,2))
	print 'Mean:', mean
	print 'Std Deviation:', std

if __name__ == '__main__':
	main()
