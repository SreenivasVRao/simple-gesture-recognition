import cv2
import numpy as np
from numpy import linalg
cap= cv2.VideoCapture(0)
idx=0

example_contour = list()

font= cv2.FONT_HERSHEY_SIMPLEX

gestures = np.load('gestures/composite_list.npy')
labels = np.load('gestures/composite_list_labels.npy')

while (cap.isOpened()):
	ret, img= cap.read()
	img= cv2.flip(img, 1)
	cv2.rectangle(img,(300,300),(0,0),(0,255,0),0)
	roi= img[0:300, 0:300]
	gray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (35,35), 0)
	_, edges= cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	_, contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	y=len(contours)	
	area= np.zeros(y)	
	for i in range(0, y):
        	area[i] = cv2.contourArea(contours[i])
	
	index= area.argmax()
	hand = contours[index]
	x,y,w,h = cv2.boundingRect(hand)
	cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),0)
	temp = np.zeros(roi.shape, np.uint8)


	M = cv2.moments(hand)

	cv2.drawContours(temp, [hand], -1, (0, 255,0), -1)
	cv2.drawContours(img, [hand], -1, (0, 255,0), -1)

	key = cv2.waitKey(1)	
	

	if key & 0xFF== ord('q'):
			break	 
	elif key == ord('c'):
		example_contour.append(hand)
		print len(example_contour)
			
	elif key== ord('i'):
		measures = list()
		for g in gestures:
			m= cv2.matchShapes(hand, g, 1, 0.0)
			measures.append(m)

		z= measures.index(min(measures))
		result = labels[z]
		cv2.putText(img,str(result), (600, 20), font, 0.7,(255,255,255),2,cv2.LINE_AA)


	cv2.imshow('Place your hand in the rectangle', img)
	


