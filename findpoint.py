import cv2
import imutils
import collections
import itertools
import math
import numpy as np
import time
import argparse
from networktables import NetworkTables

cap = cv2.VideoCapture(0)  # Create capture object for webcam
#cap.set(cv2.CV_CAP_PROP_BUFFERSIZE, 3)

cap.set(3, 640)
cap.set(4, 480)

NetworkTables.initialize(server='')  # Initialize NetworkTable server
sd = NetworkTables.getTable("VisionTable")  # Fetch the NetworkTable table

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Set the width and height from camera object
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(WIDTH, HEIGHT)
SENSITIVITY = 10  # Define the hsv sensitivity range for color filtering
ANGLE_OFFSET = 5  # Define the offset range from 90 degrees to consider a "lockon"
LOCATION_OFFSET = 60  # The amount of pixel offset from the center point in the camera to verify if object is in front
CONTOUR_LIMIT = 500  # The limit for the size of the contour

RESIZE_FACTOR = 150
WIDTH_RESIZE = 100
HEIGHT_RESIZE = 100

CENTER_POINT = {"X":int(WIDTH/2), "Y":int(HEIGHT/2)}  # Change the alignment point here!

parser = argparse.ArgumentParser(description="Settings for Team 612's Vision")
parser.add_argument("-show", "--show", type=int, help="Enable opencv imshow: 1=Enabled, (Don't enable for competition!)")
args = parser.parse_args()

vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]])

def find_angle(p1, p2):  # Basic function to calculate the angle of two points
	x = p2[0] - p1[0]  # Find the x coordinate
	y = p2[1] - p1[1]  # Find the y coordinate
	try:
		angle = math.degrees(math.atan(y/x))  # Inverse tangent (opposite over adjacent)
	except ZeroDivisionError:  # If angle is 90 degrees inverse tangent is undefined
		angle = 90.0
	return angle  # Return the angle measure

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


def verify_angle(theta):  # Check if the angle given is within the range allowed to succesfully drive straight return a boolean value
	angle_dict = {}  # Create a empty dictionary to organize data
	if abs(theta) >= 90-ANGLE_OFFSET and abs(theta) <= 90:  # If the angle is within the positive range, and if its negative absolute value
		value = True
	else:  # If not in range return False
		value = False

	angle_dict['angle'] = theta  # Assign the values from the test to keys in dictionary
	angle_dict['range'] = ANGLE_OFFSET
	angle_dict['value'] = value
	return angle_dict  # Return the dictionary


def verify_location(p1, p2):  # Check if the location of the midpoints are within the center of the screen to drive straight
	location_dict = {}
	p1_offset = CENTER_POINT["X"] - p1[0]
	p2_offset = CENTER_POINT["X"] - p2[0]

	if p1[0] > (CENTER_POINT["X"] - LOCATION_OFFSET) and p1[0] < (CENTER_POINT["X"] + LOCATION_OFFSET):  # If the first midpoint is in the range we good
		if p2[0] > (CENTER_POINT["X"] - LOCATION_OFFSET) and p2[0] < (CENTER_POINT["X"] + LOCATION_OFFSET):  # If the second midpoint is in the range we good
			value = True
		else:   # If not in range return False
			value = False
	else:
		value = False

	location_dict['p1_offset'] = p1_offset  # Assign the values from the test to keys in dictionary
	location_dict['p2_offset'] = p2_offset
	location_dict['range'] = LOCATION_OFFSET
	location_dict['value'] = value
	return location_dict  # Return the dictionary

def rescale_frame(frame, percent=75):
	width = int(frame.shape[1] * percent / 100)
	height = int(frame.shape[0] * percent / 100)
	dim = (width, height)
	return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


while True:  # Frame capture loop
	_, frame = cap.read()  # Capture the frames from the camera object

	frame = frame[HEIGHT_RESIZE:(HEIGHT - HEIGHT_RESIZE), WIDTH_RESIZE:WIDTH-WIDTH_RESIZE]
	frame = rescale_frame(frame, RESIZE_FACTOR)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_range = np.array([0,0,255-SENSITIVITY])  # This will find objects that are white
	upper_range = np.array([255,SENSITIVITY,255])  # The sensitivity will adjust the amount of white to allow

	mask = cv2.inRange(hsv, lower_range, upper_range)  # Create a mask with the ranges
	res = cv2.bitwise_and(frame, frame, mask = mask)  # Result if mask is removed from frame

	blurred = cv2.GaussianBlur(mask, (5, 5), 0)  # Blur image to reduce noise

	_, contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # Find all the contours in the mask

	if len(contours) > 0:  # Only run if there are contours on the screen

		c = max(contours, key=cv2.contourArea)  # Isolate the largest contour

		if c.size > CONTOUR_LIMIT:

			M = cv2.moments(c)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			print("Centroid of the biggest area: ({}, {})".format(cx, cy))

		else:
			sd.putNumber("angle", 0)  # Push data to table
			sd.putNumber("p1_offset", 0)  # Push data to table
			sd.putNumber("p2_offset", 0)  # Push data to table

	else:
		sd.putNumber("angle", 0)  # Push data to table
		sd.putNumber("p1_offset", 0)  # Push data to table
		sd.putNumber("p2_offset", 0)  # Push data to table

	if args.show == 1:
		cv2.imshow('frame', frame)  # Display the frames
		cv2.imshow('mask', mask)  # Display the mask

	k = cv2.waitKey(5) & 0xFF  # Check for key strokes
	if k == 27:  # If ESC is clicked close window
		break


cv2.destroyAllWindows()  # Close opencv window
cap.release()  #  Close out of webcam