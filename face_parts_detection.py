# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from matplotlib import pyplot as plt
import csv
from csv import writer
import os
os.remove("log.csv")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the face parts individually
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 1)

        # loop over the subset of facial landmarks, drawing the
        # specific face part
        for (x, y) in shape[i:j]:
            # cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            test = str(x) + "," + str(y)
            cv2.putText(clone, test, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            row_contents = [x, y]
            # Append a list as new line to an old csv file
            if(name == "jaw"):
                append_list_as_row('log.csv', row_contents)

        # extract the ROI of the face region as a separate image
        # (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        # roi = image[y:y + h, x:x + w]
        # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        # show the particular face part
        # cv2.imshow("ROI", roi)
        # cv2.imshow("Image", clone)
        # cv2.waitKey(0)

    # visualize all facial landmarks with a transparent overlay
    # output = face_utils.visualize_facial_landmarks(image, shape)
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)

    # 		# show the particular face part
    # 	plt.imshow(roi)
    # 	plt.xticks([]);plt.yticks([])
    # 	plt.imshow(clone)
    # 	plt.xticks([]);plt.yticks([])
    # 	plt.show()
    # 	fname = "results/"+"face"+str(n)+"_"+name+".png"
    # 	# print(fname)
    # 	plt.savefig(fname)

    # # visualize all facial landmarks with a transparent overlay
    # output = face_utils.visualize_facial_landmarks(image, shape)
    # plt.imshow(output)
    # plt.xticks([]);plt.yticks([])
    # fname = "results/"+"face"+"_"+"overlay.png"
    # plt.savefig(fname)
with open('log.csv', 'r') as f:
    data = list(csv.reader(f))

    leftSide1 = abs((int(data[5][1])-int(data[4][1])) /
                    (int(data[5][0])-int(data[4][0])))

    leftSide2 = abs((int(data[6][1])-int(data[5][1])) /
                    (int(data[6][0])-int(data[5][0])))
    leftSide3 = abs((int(data[7][1])-int(data[6][1])) /
                    (int(data[7][0])-int(data[6][0])))
    totalleftSide = (abs(leftSide2-leftSide1) + abs(leftSide3-leftSide2))/2
    rightSide1 = abs((int(data[10][1])-int(data[9][1])) /
                     (int(data[10][0])-int(data[9][0])))
    rightSide2 = abs((int(data[11][1])-int(data[10][1])) /
                     (int(data[11][0])-int(data[10][0])))
    rightSide3 = abs((int(data[12][1])-int(data[11][1])) /
                     (int(data[12][0])-int(data[11][0])))
    totalrightSide = (abs(rightSide2-rightSide1) +
                      abs(rightSide3-rightSide2))/2
    total = (totalrightSide+totalleftSide)/2
    row_contents = [total]
    append_list_as_row('finallog.csv', row_contents)
