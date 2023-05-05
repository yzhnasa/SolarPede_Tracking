from __future__ import print_function
from __future__ import division
import cv2
# import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from math import atan2, cos, sin, sqrt, pi

# M = "m210000"
# M = "m211000"
# M = "m211100"
M = "m211110"
# P = "p1"
# P = "p2"
# P = "p4"
# P = "p6"
# P = "p8"
# P = "p10"
P = "p20"
CSV_PATH = r"C:\Users\Brooke\OneDrive - University of Louisville\NGS\Micro Robotics\Solarpede\Solarpede Tracking\CSVs\\"
VIDEO_PATH = r"C:\Users\Brooke\OneDrive - University of Louisville\NGS\Micro Robotics\Solarpede\Solarpede " \
             r"Tracking\Videos\\ "
# CONVERSION_FACTOR = float(easygui.enterbox("Please enter the numerical conversion factor, in micrometers/pixel:"))
CONVERSION_FACTOR = float(1)
CAPTURE = cv2.VideoCapture(1)
CAPTURE.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CAPTURE.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
# Fast frame rate, low accuracy
# TRACKER = cv2.TrackerMOSSE_creat1e()
# Slow frame rate, high accuracy
TRACKER = cv2.TrackerCSRT_create()
IMAGE = CAPTURE.read()[1]


def record_session():
    """
    Records the trial.
    :return: n/a
    """
    print(M + '_' + P)
    frame_width = int(CAPTURE.get(3))
    frame_height = int(CAPTURE.get(4))
    size = (frame_width, frame_height)
    recorded_video = cv2.VideoWriter(VIDEO_PATH + M + '_' + P + '.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             cv2.CAP_PROP_FPS, size)
    return recorded_video


def draw_box(image, box):
    """
    Maintains a line box around the desired object.
    :param image: The image captured by the camera.
    :param box: Details about the box around the object
    :return: n/a
    """
    # Get coordinates.
    # x is the pixel value corresponding to horizontal movement of the object.
    # (i.e. x = 0 is the far left of the screen, bigger number is further to the right)
    # y is the pixel value corresponding to vertical movement of the object.
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(image, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.putText(image, "Tracking", (25, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
    # return x, y, w, h


def plot_data():
    """
    Plots the data from the CSV file generated.
    :return:
    """
    x0 = []
    y0 = []
    time0 = []

    with open(CSV_PATH + M + '_' + P + '.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            x0.append(row[0])
            y0.append(row[1])
            time0.append(row[2])
    x0 = np.asarray(x0, dtype='float64')
    y0 = np.asarray(y0, dtype='float64')
    m0, b0 = np.polyfit(x0, y0, 1)
    print('m0:', m0)
    plt.plot(x0, y0, 'r', marker='o')
    plt.plot(x0, m0 * x0 + b0, 'r', label=M + '_' + P)
    plt.legend()
    plt.show()


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
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


def get_coordinates(result):
    """
    Handles the coordinates of the object and stores them.
    :param result: The video recorder
    :return: n/a
    """
    bbox = cv2.selectROI("Tracking", IMAGE, False)
    TRACKER.init(IMAGE, bbox)
    csv_file = open(CSV_PATH + M + '_' + P + '.csv', 'w', newline='')
    # csv_file = open('test.csv', 'w', newline='')
    csv_file_writer = csv.writer(csv_file)
    while True:
        timer = cv2.getTickCount()
        img = CAPTURE.read()[1]
        xCoordinate = bbox[0] * CONVERSION_FACTOR
        yCoordinate = bbox[1] * CONVERSION_FACTOR
        xCoordinateString = "X Coordinate (micrometers): " + str("%.2f" % xCoordinate)
        yCoordinateString = "Y Coordinate (micrometers): " + str("%.2f" % yCoordinate)

        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        box_img = img[y:y + h, x:x + w]
        gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
        # _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bw = cv2.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        max_contour_index = find_max_contour(contours)
        cv2.drawContours(box_img, contours, max_contour_index, (0, 0, 255), 2)
        angle = getOrientation(contours[max_contour_index], box_img)

        csv_file_writer.writerow([xCoordinate, yCoordinate, angle, cv2.getTickCount() / cv2.getTickFrequency()])
        result.write(img)
        success, bbox = TRACKER.update(img)

        if success:
            draw_box(img, bbox)
        else:
            cv2.putText(img, "Lost", (25, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(img, str(int(fps)), (25, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, xCoordinateString, (150, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, yCoordinateString, (150, 75), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Tracking", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    CAPTURE.release()
    result.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    csv_file.close()


if __name__ == "__main__":
    recording = record_session()
    get_coordinates(recording)
    plot_data()
