import cv2
import imutils
import collections
import itertools
import math
import numpy as np
import time
from networktables import NetworkTables

cap = cv2.VideoCapture(0)  # Create capture object for webcam

NetworkTables.initialize(server='10.6.12.2')  # Initialize NetworkTable server
sd = NetworkTables.getTable("VisionTable")  # Fetch the NetworkTable table

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Set the width and height from camera object
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
SENSITIVITY = 8  # Define the hsv sensitivity range for color filtering
ANGLE_OFFSET = 5  # Define the offset range from 90 degrees to consider a "lockon"
LOCATION_OFFSET = 60  # The amount of pixel offset from the center point in the camera to verify if object is in front
CONTOUR_LIMIT = 500  # The limit for the size of the contour


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
	p1_offset = int(WIDTH/2) - p1[0]
	p2_offset = int(WIDTH/2) - p2[0]

	if p1[0] > int((WIDTH/2) - LOCATION_OFFSET) and p1[0] < int((WIDTH/2) + LOCATION_OFFSET):  # If the first midpoint is in the range we good
		if p2[0] > int((WIDTH/2) - LOCATION_OFFSET) and p2[0] < int((WIDTH/2) + LOCATION_OFFSET):  # If the second midpoint is in the range we good
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


while True:  # Frame capture loop
	_, frame = cap.read()  # Capture the frames from the camera object

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_range = np.array([0,0,255-SENSITIVITY])  # This will find objects that are white
	upper_range = np.array([255,SENSITIVITY,255])  # The sensitivity will adjust the amount of white to allow

	mask = cv2.inRange(hsv, lower_range, upper_range)  # Create a mask with the ranges
	res = cv2.bitwise_and(frame, frame, mask = mask)  # Result if mask is removed from frame

	blurred = cv2.GaussianBlur(mask, (5, 5), 0)  # Blur image to reduce noise

	_, contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # Find all the contours in the mask

	if contours:  # Only run if there are contours on the screen

		c = max(contours, key=cv2.contourArea)  # Isolate the largest contour

		if c.size > CONTOUR_LIMIT:

			box = cv2.minAreaRect(c)  # Find the box dimensions of contour
			box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)  # Adjust for rotation
			box = np.array(box, dtype="int")  # Create a numpy array of box dimensions

			rect = order_points(box)  # Return the two smallest point combinations from box dimensions
			side1 = rect[0]  # Split the multidimensional array returned from "order_points" into two variables
			side2 = rect[1]
			midpoint1 = midpoint(side1[0], side1[1])  # Find the midpoint of each point combinations
			midpoint2 = midpoint(side2[0], side2[1])

			angle = find_angle(midpoint1, midpoint2)  # Get the angle of the line made by the two points
			
			print(angle)
			#print(angle)
			angle_dict = verify_angle(angle)
			location_dict = verify_location(midpoint1, midpoint2)

			sd.putNumber("angle", angle)  # Push data to table
			sd.putNumber("p1_offset", location_dict['p1_offset'])  # Push data to table
			sd.putNumber("p2_offset", location_dict['p2_offset'])  # Push data to table

		else:
			sd.putNumber("angle", 0)  # Push data to table
			sd.putNumber("p1_offset", 0)  # Push data to table
			sd.putNumber("p2_offset", 0)  # Push data to table

	else:
		sd.putNumber("angle", 0)  # Push data to table
		sd.putNumber("p1_offset", 0)  # Push data to table
		sd.putNumber("p2_offset", 0)  # Push data to table

	k = cv2.waitKey(5) & 0xFF  # Check for key strokes
	if k == 27:  # If ESC is clicked close window
		break


cv2.destroyAllWindows()  # Close opencv window
cap.release()  #  Close out of webcam