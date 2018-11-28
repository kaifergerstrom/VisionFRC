import cv2
import json
import numpy as np

cap = cv2.VideoCapture(0)  # Define capture object
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

data = []  # Create array for hsv values parsed from json
json_path = 'data/hsv.json'

screen_width = int(cap.get(3))
screen_height = int(cap.get(4))
screen_center_x = int(screen_width / 2)
screen_center_y = int(screen_height / 2)

lockon_offset = 50

object_width = 255  # in pixels
object_height = 189  # in pixels
distance = 12  # in inches
known_width = 7  # in inches
focal_length = (object_width * distance) / known_width


def nothing(x):  # Empty void function for sliders
    pass


def open_json(url):  # Open json file parse the data and return it
    with open(url) as f:
        data = json.load(f)
        return data


def save_config(tuple1, tuple2):
    data = [tuple1, tuple2]  # Save tuples into list and push into JSON file
    print(data)
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)


def slider_init():
    data = open_json(json_path)
    lower = data[0]  # Parse data into lower half
    higher = data[1]  # Parse data into upper half

    # Set values of lower sliders
    cv2.setTrackbarPos("H", "slider", lower[0])
    cv2.setTrackbarPos("S", "slider", lower[1])
    cv2.setTrackbarPos("V", "slider", lower[2])

    # Set values of higher sliders
    cv2.setTrackbarPos("H2", "slider", higher[0])
    cv2.setTrackbarPos("S2", "slider", higher[1])
    cv2.setTrackbarPos("V2", "slider", higher[2])


def slider_frame():
    cv2.namedWindow("slider")  # Create a window for sliders

    # Lower slider values
    cv2.createTrackbar("H", "slider", 0, 255, nothing)
    cv2.createTrackbar("S", "slider", 0, 255, nothing)
    cv2.createTrackbar("V", "slider", 0, 255, nothing)

    # Upper slider values
    cv2.createTrackbar("H2", "slider", 0, 255, nothing)
    cv2.createTrackbar("S2", "slider", 0, 255, nothing)
    cv2.createTrackbar("V2", "slider", 0, 255, nothing)

    slider_init()


def tuner():
    slider_frame()

    while True:
        _, frame = cap.read()

        # Fetch values from sliders
        h1 = cv2.getTrackbarPos('H', 'slider')
        s1 = cv2.getTrackbarPos('S', 'slider')
        v1 = cv2.getTrackbarPos('V', 'slider')

        h2 = cv2.getTrackbarPos('H2', 'slider')
        s2 = cv2.getTrackbarPos('S2', 'slider')
        v2 = cv2.getTrackbarPos('V2', 'slider')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV: hue, sat, val
        lower_hsv = np.array([h1, s1, v1])

        # 111, 255, 255
        upper_hsv = np.array([h2, s2, v2])

        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Blur image to reduce noise
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)  # create a frame that filters out highs and lows
        res = cv2.bitwise_and(frame, frame, mask=mask)  # Apply mask to frame to filter out content

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=1)
        dilation = cv2.dilate(mask, kernel, iterations=1)

        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

        _, contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        height, width, _ = frame.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        lockon = False
        cv2.line(frame, (screen_center_x + lockon_offset, 0), (screen_center_x + lockon_offset, screen_height),
                 (0, 255, 0), 1)
        cv2.line(frame, (screen_center_x - lockon_offset, 0), (screen_center_x - lockon_offset, screen_height),
                 (0, 255, 0), 1)

        if not contours:
            pass
        else:
            c = max(contours, key=cv2.contourArea)
            (contour_x, contour_y, contour_width, contour_height) = cv2.boundingRect(c)
            distance = (known_width * focal_length) / contour_width
            # print("{}, {}".format(w, h))
            cv2.rectangle(frame, (contour_x, contour_y), (contour_x + contour_width, contour_y + contour_height), (255, 0, 0), 2)

            center_x = int(contour_x + (contour_width / 2))
            center_y = int(contour_y + (contour_height / 2))

            cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)

            location = ""
            if center_x >= screen_center_x + lockon_offset:
                location = "right"
            elif center_x <= screen_center_x - lockon_offset:
                location = "left"
            else:
                location = "center"
                print('Target Locked: {} in away'.format(int(round(distance))))

        cv2.imshow('frame', frame)
        # cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        cv2.imshow('mask', mask)

        k = cv2.waitKey(5) & 0xFF

        if k == ord('m'):
        	print('{}, {}'.format(contour_width, contour_height))

        if k == 32:  # If SPACE is pressed store config values
            save_config((h1, s1, v1), (h2, s2, v2))

        if k == 27:  # If ESC close windows
            break


tuner()








