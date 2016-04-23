import cv2
import numpy as np

temp = np.zeros((300, 300))
cap= cv2.VideoCapture(0)
while True:
	#img= cv2.imread('handexample.png')
	ret, img= cap.read()
	img= cv2.flip(img, 1)
	roi= img[100:400, 100:400, :]

	cv2.rectangle(img, (400, 400), (100, 100), 1,0)	
	roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	temp=roi
	#edges= cv2.Canny(roi, 60, 180)
	#edges2= cv2.Canny(roi, 60, 180)
	blurred = cv2.GaussianBlur(roi, (5,5), 0)
	_, edges= cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	y=len(contours)
	area=np.zeros(y)
	for i in range(0, y):
		moments= cv2.moments(contours[i])
		area[i]= moments['m00']

	max_area= area.max()
	index= area.argmax() 	
	cnt= contours[index]
	#cv2.drawContours(img[100:400, 100:400, :],cnt,-1,(0, 255, 255),5)
	x,y,w,h = cv2.boundingRect(cnt)
	"""
	cv2.rectangle(img[100:400, 100:400, :],(x,y),(x+w,y+h),(255,255,255),0)
    	hull = cv2.convexHull(cnt)
    	drawing = np.zeros(img.shape,np.uint8)
    	cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
    	cv2.drawContours(drawing,[hull],0,(0,0,255),0)
    	hull = cv2.convexHull(cnt,returnPoints = False)
    	defects = cv2.convexityDefects(cnt,hull)
    	count_defects = 0
    	cv2.drawContours(img, contours, -1, (0,255,0), 3)
	cv2.imshow('aaa', img)

	if cv2.waitKey(1) & 0xFF== ord('q'):
		break	
	
	


