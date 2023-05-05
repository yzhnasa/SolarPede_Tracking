from __future__ import print_function
from __future__ import division
import easygui
import time
import csv
import cv2 as cv
import numpy as np
import argparse
import math
from math import atan2, cos, sin, sqrt, pi

conversionFactor = float(easygui.enterbox("Please enter the numerical conversion factor, in micrometers/pixel:"))

cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('LookupTable1.wmv')

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 768)

#Fast frame rate, low accuracy
#tracker = cv.TrackerMOSSE_create()
#Slow frame rate, high accuracy
tracker = cv.TrackerCSRT_create()

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)

time.sleep(5)
success, img = cap.read()
#frame75 = rescale_frame(img, percent=25)
#img = frame75
bbox = cv.selectROI("Tracking", img, False)
tracker.init(img, bbox)

def drawBox(img, bbox):

    # Get coordinates.
    # x is the pixel value corresponding to horizontal movement of the object.
    # (i.e. x = 0 is the far left of the screen, bigger number is further to the right)
    # y is the pixel value corresponding to vertical movement of the object.
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv.putText(img, "Tracking", (25, 75), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

csv_file = open('test.csv', 'w', newline='')
csv_file_writer = csv.writer(csv_file)


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    return angle

def find_max_contour(contours):
    max_contour_index = 0
    object_max_contour_index = 0
    max_contour_length = 0
    for i, c in enumerate(contours):
        temp_contour_length = len(c)
        if temp_contour_length > max_contour_length:
            object_max_contour_index = max_contour_index
            max_contour_index = i
            max_contour_length = temp_contour_length
    return object_max_contour_index
    # return max_contour_index



while True:
    timer = cv.getTickCount()
    success, img = cap.read()
    #frame75 = rescale_frame(img, percent=25)
    #img = frame75

    xCoordinate = bbox[0] * conversionFactor
    yCoordinate = bbox[1] * conversionFactor

    xCoordinateString = "X Coordinate (micrometers): " + str("%.2f" % xCoordinate)
    yCoordinateString = "Y Coordinate (micrometers): " + str("%.2f" % yCoordinate)
    csv_file_writer.writerow([xCoordinate, yCoordinate, cv.getTickCount()/cv.getTickFrequency()])
    #csv_file_writer.writerow(str(yCoordinate))


    success, bbox = tracker.update(img)

    if success:
        drawBox(img, bbox)
    else:
        cv.putText(img, "Lost", (25, 75), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv.getTickFrequency()/(cv.getTickCount()-timer)
    cv.putText(img, str(int(fps)), (25,50), cv.FONT_HERSHEY_COMPLEX, 0.7,(0,0,255),2)
    cv.putText(img, xCoordinateString, (150, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
    cv.putText(img, yCoordinateString, (150, 75), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

    # cv.imshow("Tracking",img)

    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    box_img = img[y:y+h, x:x+w]

    # Convert image to grayscale
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Convert image to binary
    gray = cv.cvtColor(box_img , cv.COLOR_BGR2GRAY)
    # _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    bw = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    # _, bw = cv.thqreshold(gray, 50, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    max_contour_index = find_max_contour(contours)
    print(max_contour_index)
    cv.drawContours(box_img, contours, max_contour_index, (0, 0, 255), 2)
    angle = getOrientation(contours[max_contour_index], box_img)
    # cv.drawContours(img, contours, max_contour_index, (0, 0, 255), 2)
    # angle = getOrientation(contours[max_contour_index], img)
    print("angle: ", math.degrees(angle))

    cv.imshow("Tracking", img)
    # cv.imshow("Tracking", gray)
    # cv.imshow("Tracking", bw)

    if cv.waitKey(1) & 0xff == ord('q'):
        break