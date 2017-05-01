#!/usr/bin/env python
import rospy
import cv2
import argparse
import math
import utm

from obe_toolset.msg import ImageAndPose
from cv_bridge import CvBridge
from classify_images import TargetClassifier

# Create command line args
parser = argparse.ArgumentParser(
					description='This program is to be run on the ground station '
								'side of the 2016-17 computer vision system. It'
								'continuously scans a directory for images and passes'
								'them to image classifier(s). Results are sent to the'
								'Interop Server')
parser.add_argument('-u', '--userid', default='azautonomous',
						help='User ID for Interop Server.')
parser.add_argument('-c', '--checkpoint_dir', default='checkpoints', 
							help='Path to checkpoint directories. '
							'Each classifier should be kept in a separate directory '
							'according to their name (e.g. scope). For example, '
							'checkpoints/ with subdirectories shape/, alphanum/, etc')

args = parser.parse_args()

classifier = TargetClassifier(args.userid, args.checkpoint_dir)
bridge = CvBridge()

def callback(data):
	image = bridge.imgmsg_to_cv2(data.image)
	# TODO: Parse these
	data.x # UTM coordinates (Easting)
	data.y # UTM coordinates (Northing)
	data.z # AGL altitude in meters
	data.roll # Unused
	data.pitch # Unused
	data.yaw # Radians CCW from east
	location = utm.to_latlon(data.x, data.y, 12, 'S')
	orientation = math.degrees(data.yaw) + 90
	classifier.classify_and_maybe_transmit(image, location=location,
	                                       orientation=orientation)

def run():
	rospy.init_node('ROI_Listener', anonymous=True)
	rospy.Subscriber('ROIs', ImageAndPose, callback)

	rospy.spin()

if __name__ == '__main__':
	run()
