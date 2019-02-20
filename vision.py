# import the necessary packages
from __future__ import print_function
from networktables import NetworkTables
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import itertools
import math

ROBORIO_IP = "10.6.12.2"

SENSITIVITY = 5  # Determine the sensitivity white color detection
LOCATION_OFFSET = 60  # The amount of pixel offset from the center point in the camera to verify if object is in front

# Initialize variables to store frame width and height
WIDTH = 0
HEIGHT = 0

CENTER_POINT = {"X":0, "Y":0}  # Initialize dictionary to store the alignment points

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=-1, help="Whether or not frames should be displayed")
ap.add_argument("-t", "--table", type=str, required=True, help="Determine the name of the NetworkTable to push to")
args = vars(ap.parse_args())

NetworkTables.initialize(server=ROBORIO_IP)  # Initialize NetworkTable server
sd = NetworkTables.getTable(args["table"])  # Fetch the NetworkTable table

# created a *threaded* video stream, allow the camera sensor to warmup,
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()  # Start the FPS counter
 

def get_offset(x):
	return CENTER_POINT["X"] - x


def get_frame_size(img):  # Function that will return a tuple with the width and height of the opencv frame
	return tuple(img.shape[1::-1])


def midpoint(p1, p2):  # Basic function to calculate the midpoint of two points
	return (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))


def distance(p1, p2):  # Calculates the distance between two points
	return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def order_points(box):  # Recieves dimensions of bounding box, and sorts through points to find two smallest points
	combinations = list(itertools.combinations(box, 2))  # Find all the possible combinations of points given bounding box

	points = []  # Define a new array to store all point combinations after for loop
	for point in combinations:  # Loop through all combinations
		d = distance(point[0], point[1])  # Find the distance of each combination
		points.append([point, d])  # Append the distance and actual point to multidimensional array

	points.sort(key=lambda x: x[1])  # After loop is finished sort array by second element in sub array which is distance (small to large)
	points = points[:2] # Chop everything except for the first two elements in array off
	return [points[0][0], points[1][0]]  # Return a multi dimensional array with the two points


def find_angle(p1, p2):  # Basic function to calculate the angle of two points
	x = p2[0] - p1[0]  # Find the x coordinate
	y = p2[1] - p1[1]  # Find the y coordinate
	try:
		angle = math.degrees(math.atan(y/x))  # Inverse tangent (opposite over adjacent)
	except ZeroDivisionError:  # If angle is 90 degrees inverse tangent is undefined
		angle = 90.0
	return angle  # Return the angle measure


while True:

	# Grab the frame from the threaded video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Code to update dimensions of the frame and update the alignment point
	frame_size = get_frame_size(frame)
	WIDTH = frame_size[0]
	HEIGHT = frame_size[1]
	CENTER_POINT["X"] = int(WIDTH/2)
	CENTER_POINT["Y"] = int(HEIGHT/2)

	# Step 1: HSV Filtering
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_range = np.array([0,0,255-SENSITIVITY])  # This will find objects that are white
	upper_range = np.array([255,SENSITIVITY,255])  # The sensitivity will adjust the amount of white to allow

	mask = cv2.inRange(hsv, lower_range, upper_range)  # Create a mask with the ranges

	# Step 2: Reduce mask noise
	blurred = cv2.GaussianBlur(mask, (11, 11), 0)  # Blur image to reduce noise

	# Step 3: Find contours and filter them
	_, contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find all the contours in the mask
 	
	if len(contours) > 0:  # Only run if there are contours on the screen

		c = max(contours, key=cv2.contourArea)  # Isolate the largest contour
		box = cv2.minAreaRect(c)  # Find the box dimensions of contour
		box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)  # Adjust for rotation
		box = np.array(box, dtype="int")  # Create a numpy array of box dimensions

		rect = order_points(box)  # Return the two smallest point combinations from box dimensions
		side1 = rect[0]  # Split the multidimensional array returned from "order_points" into two variables
		side2 = rect[1]
		midpoint1 = midpoint(side1[0], side1[1])  # Find the midpoint of each point combinations
		midpoint2 = midpoint(side2[0], side2[1])

		# Step 4: Calculate the angle and offsets of the contours
		angle = find_angle(midpoint1, midpoint2)  # Get the angle of the line made by the two points
		p1_offset = get_offset(midpoint1[0])
		p2_offset = get_offset(midpoint2[0])

		# Step 5: Push the data to NetworkTables
		sd.putNumber("angle", angle)  # Push data to table
		sd.putNumber("p1_offset", p1_offset)  # Push data to table
		sd.putNumber("p2_offset", p2_offset)  # Push data to table

		print([angle, p1_offset, p2_offset])

		if args["display"] > 0:
			cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)  # Draw rest of contour for fun
			cv2.line(frame, midpoint1, midpoint2, (255,0,0), 2)  # Draw line create by two for fun
			cv2.line(frame, (CENTER_POINT["X"], 0), (CENTER_POINT["X"], HEIGHT), (0,0,255), 2)  # Draw line through screen by two for fun
			cv2.line(frame, (CENTER_POINT["X"] - LOCATION_OFFSET, 0), (CENTER_POINT["X"] - LOCATION_OFFSET, HEIGHT), (255,0,0), 2)  # Draw line through screen by two for fun
			cv2.line(frame, (CENTER_POINT["X"] + LOCATION_OFFSET, 0), (CENTER_POINT["X"] + LOCATION_OFFSET, HEIGHT), (255,0,0), 2)  # Draw line through screen by two for fun

	else:
		sd.putNumber("angle", 0)  # Push data to table
		sd.putNumber("p1_offset", 0)  # Push data to table
		sd.putNumber("p2_offset", 0)  # Push data to table

	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		cv2.imshow("Mask", mask)
		key = cv2.waitKey(1) & 0xFF
		if key == 27:
			break
 	
	# update the FPS counter
	fps.update()
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()