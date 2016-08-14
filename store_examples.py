import cv2
import numpy as np


cap= cv2.VideoCapture(0)
labels = list()
example_contour = list()
images= list()
imname=  0
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

	cv2.drawContours(img, [hand], -1, (0, 255,0), -1)
	cv2.drawContours(temp, [hand], -1, (0, 255,0), -1)

	key = cv2.waitKey(1)	
	

	if key & 0xFF== ord('q'):
		c = np.asarray(example_contour)
		print c.shape
		print labels
		np.save('gestures/composite_list.npy', c)
		np.save('gestures/composite_list_labels.npy', labels)
		for img in images:
			i = str(imname)
			cv2.imwrite('gestures/'+i.zfill(4)+'.jpg', img)
			imname+=1
		break	 
	elif key & 0xFF == ord('0'):
		example_contour.append(hand)
		images.append(temp)
		labels.append(0)
		print len(example_contour)
	elif key & 0xFF == ord('1'):
		example_contour.append(hand)
		images.append(temp)
		labels.append(1)
		print len(example_contour)
	elif key & 0xFF == ord('2'):
		example_contour.append(hand)
		images.append(temp)
		labels.append(2)
		print len(example_contour)
	elif key & 0xFF == ord('3'):
		example_contour.append(hand)
		images.append(temp)
		labels.append(3)
		print len(example_contour)
	elif key & 0xFF == ord('4'):
		example_contour.append(hand)
		images.append(temp)
		labels.append(4)
		print len(example_contour)
	elif key & 0xFF == ord('5'):
		example_contour.append(hand)
		images.append(temp)
		labels.append(5)
		print len(example_contour)
			
	cv2.imshow('Place your hand in the rectangle', img)
	cv2.imshow('Contour', temp)
	cv2.moveWindow('Contour', 600, 0)


