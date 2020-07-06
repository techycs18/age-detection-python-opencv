#============================SOURCE=================================
#     https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/
#===================================================================

import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help = 'Image path')
ap.add_argument('-f', '--face', required = True, help = 'face detector model path')
ap.add_argument('-a', '--age', required = True, help = 'age detector model path')
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# define the list of age buckets our age detector will predict
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)","(38-43)", "(48-53)", "(60-100)"]

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
print("weights: ", weightsPath)
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our serialized age detector model from disk
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
print("weights: ", weightsPath)
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

image = cv2.imread(args['image'])

print('image shape:{}'.format(image.shape))
(h,w) = image.shape[:2]

print('image (h,w):({},{})'.format(h, w))

blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("[INFO] computing face detections...")

faceNet.setInput(blob)
detections = faceNet.forward()

print('detection(shape) => ', detections.shape)

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		print('h:{},w:{}\ndetection:{}\nbounding_box:{}\nbox:{})'.format(h, w, detections[0, 0, i, :],detections[0, 0, i, 3:7], box))
		(startX, startY, endX, endY) = box.astype("int")
		# extract the ROI of the face and then construct a blob from
		# *only* the face ROI
		face = image[startY:endY, startX:endX]
		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
		print('Confidence => ',confidence, i)

		# make predictions on the age and find the age bucket with
		# the largest corresponding probability
		ageNet.setInput(faceBlob)
		preds = ageNet.forward()
		print('pred => ',preds)
		i = preds[0].argmax()
		print('i => ', i)
		age = AGE_BUCKETS[i]
		print('age => ', age)
		ageConfidence = preds[0][i]
		# display the predicted age to our terminal
		text = "{}: {:.2f}%".format(age, ageConfidence * 100)
		print("[INFO] {}".format(text))
		# draw the bounding box of the face along with the associated
		# predicted age
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

#===============To run code==============
#python image_age_detection.py --image images/1.png --face face_detector --age age_detector
