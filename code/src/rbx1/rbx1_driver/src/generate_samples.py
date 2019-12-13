#! /usr/bin/env python

import time, pickle, argparse, sys
import sys, traceback, random, math
import glob
import rospy
from std_msgs.msg import Header, String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from worldobject import RosObject
from tqdm import tqdm, trange
from tf.transformations import quaternion_from_euler


def generate(num_samples=64, mode="interactive", target_directory='./files/', image_width=128, image_height=128, image_depth=3):		# other modes are save and read

	# initialize the data and labels
	data = []
	labels = []
	cv_image = None
	
	if mode != "read":

		box_pub = rospy.Publisher('/unity_cmd/box_pose', PoseStamped, queue_size=10)
		ball_pub = rospy.Publisher('/unity_cmd/ball_pose', PoseStamped, queue_size=10)
		command_pub = rospy.Publisher('/unity_cmd/command', String, queue_size=10)

		rospy.init_node("ros_unity_sample_generator")

		unity_image = None
		last_unity_image = None

		for _ in trange(0, num_samples):		# tqdm's range

			cmd_str = String()
			cmd_str.data = "randomize"
			command_pub.publish(cmd_str)

			# put the ball and box into Unity, get back the image
			ball = RosObject()
			box = RosObject()

			hdr = Header()

			ball_pose = PoseStamped()
			hdr.stamp = rospy.Time.now()
			ball_pose.header = hdr
			# NOTE:  THese were originally in x, y, z, but ROS# is converting x->z, -(y->x) , z->y.  See Ros# TransformExtensions.cs Ros2Unity
			ball_pose.pose.position.x = ball.unity_x
			ball_pose.pose.position.y = ball.unity_y
			ball_pose.pose.position.z = ball.unity_z
			q = quaternion_from_euler(0, 0, random.random() * math.pi)
			ball_pose.pose.orientation.x = q[0]
			ball_pose.pose.orientation.y = q[1]
			ball_pose.pose.orientation.z = q[2]
			ball_pose.pose.orientation.w = q[3]

			box_pose = PoseStamped()
			hdr.stamp = rospy.Time.now()
			box_pose.header = hdr
			box_pose.pose.position.x = box.unity_x
			box_pose.pose.position.y = box.unity_y
			box_pose.pose.position.z = box.unity_z
			q = quaternion_from_euler(0, 0, random.random() * math.pi)
			box_pose.pose.orientation.x = q[0]
			box_pose.pose.orientation.y = q[1]
			box_pose.pose.orientation.z = q[2]
			box_pose.pose.orientation.w = q[3]

			box_pub.publish(box_pose)			# publish to Unity
			ball_pub.publish(ball_pose)			# publish to Unity

			while True:
				time.sleep(0.1)			# this seems to be enough time on ethernet to allow the image to come back..
				unity_image = rospy.wait_for_message('/robot/camera/compressed', CompressedImage)
				if last_unity_image is None:		# first time through
					break
				if unity_image != last_unity_image:
					break

			new_labels = [ball.x, ball.y, ball.z, box.x, box.y, box.z]

			np_arr = np.fromstring(unity_image.data, np.uint8)				# convert to np array
			cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)				# convert to openCV

			last_unity_image = unity_image

			if mode == "save":				# write image to file, and write labels to text file
				filename = target_directory + str(time.time())
				cv2.imwrite(filename + '.png', cv_image)
				with open(filename + '.txt', 'w') as f: 
					f.write(str(new_labels))
				next			# don't worry about building up the 

			image = cv2.resize(cv_image, (image_width, image_height))	# use OpenCV to resize

			# the way to interpret the image.  x = how far from bottom, y = how far from right, z = up towards camera
	#		cv2.imshow( "Display window", image )			# pop up a window to view the image
	#		cv2.waitKey(0)
	#		cv2.destroyAllWindows()

			labels.append(new_labels)

			image = img_to_array(image)			# convert to numpy using keras preprocessing
			data.append(image)						# put image and labels into their lists

	elif mode == "read":
		data, labels = read_from_directory("./files/*.png", image_width, image_height, num_samples)

	if len(labels) != len(data):
		print("[ERROR] Your data {} and labels {} are not the same length.  You idiot.  ".format(len(data), len(labels)))
	else:
		print("[INFO] Your have generated {} data and {} labels.  Good for you.".format(len(data), len(labels)))

	# smooshify image data between 0 and 1
	data = np.array(data, dtype="float") / 255.0

	return data, labels

def read_from_directory(pattern="./files/*.png", image_width=128, image_height=128, num_samples=sys.maxint):		# specify image file pattern for glob

	data = []
	labels = []
	cv_image = None

	read_counter = 0
	files = glob.glob(pattern)
	for file in files:
		if read_counter >= num_samples:
			break
		read_counter = read_counter + 1
		txtname = file[:-4]
		cv_image = cv2.imread(file)
		image = cv2.resize(cv_image, (image_width, image_height))	# use OpenCV to resize
		image = img_to_array(image)			# convert to numpy using keras preprocessing
		data.append(image)
		with open (txtname + ".txt", 'r') as f:
			temp = f.read()
			content = map(float, temp[1:-1].split(","))
			content = content[:2] + content[3:5]			# NOTE:  this line removes the Z axes
			labels.append(content)

	data = np.array(data, dtype="float") / 255.0

	return data, labels