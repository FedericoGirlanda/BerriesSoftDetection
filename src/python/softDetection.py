from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import os
from pathlib import Path
import csv

from utils import plot_one_box, filterBox, getBoxes, nonMaxSuppress, centroid, boxesCenters, lowpass, findBunch

# Load the model
model = load_model('models/YOLOv5_5')
dim = (640,640) # input image dimension

##########################
# Detection on test Images
##########################

# Create tests result file if not present
results_csv = "results/detectionBoxes.csv"
header = ["image", "boxes0","boxes1","boxes2","boxes3","boxes4" ]
f = open(results_csv, 'w')
writer = csv.writer(f)
#if not os.path.exists(results_csv):
# write a row to the csv file
writer.writerow(header)

# Testing all the test image
test_dir = "dataset/EIdataset/testing/"
imgToShow_path = "IMG_1071.jpg.4b38bvv8.ingestion-55b478f466-cp67z.jpg"
image = cv2.imread(test_dir+imgToShow_path)
image = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
for img_path in os.listdir(test_dir):
	if Path(img_path).suffix == '.jpg': 
		# Load the image
		img = load_img(test_dir+img_path, target_size=dim)

		# Convert the image to a numpy array
		input_data = img_to_array(img)

		# Normalize the pixel values
		input_data /= 255.0

		# Add a dimension to match the model's input shape
		input_data = np.expand_dims(input_data, axis=0)

		# Make a prediction
		prediction = model.predict(input_data)[0] #(x,  y,  w,  h,  confidence,  class_n) = prediction.T
		prediction = filterBox(prediction,0.3)

		# Convert to Boxes
		boxes = getBoxes(prediction, dim)

		# NMS
		boxes = nonMaxSuppress(boxes,0.7)
		#print(f"Coarse number of berries: ", len(boxes))

		# Get centers of boxes
		centers = boxesCenters(boxes)

		# Erosion and Dilation
		xb,yb,hb,wb = findBunch(centers)

		# # Calculate the centroid of the boxes
		# c = centroid(centers)
		# print(f"Centroid: ", c)

		# Calculate the center of the bunch box
		c = (xb+wb/2,yb+hb/2)
		#print(f"Bunch center: ", c)

		# Write results in the csv file
		save_img_path = np.append(img_path,[""]*len(boxes))
		save_boxes = np.vstack((np.array([xb, yb, xb + wb - 1, yb + hb - 1,1]), boxes))
		writer.writerows(np.hstack((save_img_path[:,np.newaxis], save_boxes)))
			

		if img_path == imgToShow_path:
			# Load the input image (in OpenCV format), resize it and plot the bounding boxes
			for i in range(len(boxes)):
				plot_one_box(boxes[i], image, (0,0,255), "berry", 1)
			image = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
			cv2.arrowedLine(image, (int(dim[0]/2), int(dim[1]/2)), (int(c[0]), int(c[1])), (255,255,255), 3, 5, 0, 0.1)
			cv2.rectangle(image, (xb, yb), (xb + wb - 1, yb + hb - 1), (255,255,255), 2)
f.close()

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)

#####################
# Detection on Video
#####################

# # initialize the video stream, pointer to output video file, and
# # frame dimensions
# vs = cv2.VideoCapture("dataset/IMG_1119.MOV")
# writer = None
# (W, H) = (None, None)
# c0_prev = dim[0]/2
# c1_prev = dim[1]/2
# beta = 0.8
# # loop over frames from the video file stream
# while True:
# 	# read the next frame from the file
# 	(grabbed, frame) = vs.read()
# 	# if the frame was not grabbed, then we have reached the end of the stream
# 	if not grabbed:
# 		break
# 	# if the frame dimensions are empty, grab them
# 	if W is None or H is None:
# 		(H, W) = frame.shape[:2]
		
#     # clone the output frame, then preprocess it
# 	output = frame.copy()
# 	frame = cv2.resize(frame, dim).astype("float32")
# 	input_data = img_to_array(frame)
# 	input_data /= 255.0

#     # make predictions on the frame 
# 	prediction = model.predict(np.expand_dims(frame, axis=0))[0]
# 	prediction = filterBox(prediction,0.3)

# 	if len(prediction) > 1:
# 		boxes = getBoxes(prediction, dim)
# 		boxes = nonMaxSuppress(boxes,0.8)
# 		if len(boxes)> 1:
# 			centers = boxesCenters(boxes)
# 			xb,yb,hb,wb = findBunch(centers)
# 			c = (xb+wb/2,yb+hb/2)
# 			#c = centroid(centers)
# 			# draw on the output frame
# 			for i in range(len(boxes)):
# 				plot_one_box(boxes[i], frame, (0,0,255), "berry", 1)
# 			c0_prev = lowpass(c0_prev,c[0],beta)
# 			c1_prev = lowpass(c1_prev,c[1],beta)
# 			cv2.arrowedLine(frame, (int(dim[0]/2), int(dim[1]/2)), (int(c0_prev), int(c1_prev)), (255,255,255), 3, 5, 0, 0.1)
# 			cv2.rectangle(frame, (xb, yb), (xb + wb - 1, yb + hb - 1), (255,255,255), 2)

# 	# # check if the video writer is None
# 	# if writer is None:
# 	# 	# initialize our video writer
# 	# 	fourcc = cv2.VideoWriter_fourcc(*'XVID') #*"MJPG")
# 	# 	writer = cv2.VideoWriter("results/videoDetection.avi", fourcc, 30,
# 	# 		(W, H), True)
# 	# # write the output frame to disk
# 	# writer.write(output)
	
# 	# show the output image
# 	cv2.imshow("Output", frame)#output)
# 	key = cv2.waitKey(1) & 0xFF
# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break
# # release the file pointers
# print("[INFO] cleaning up...")
# #writer.release()
# vs.release()