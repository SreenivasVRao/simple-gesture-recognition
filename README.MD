#Simple Hand Gesture Recognition

To build your own gesture dictionary, run

`python store_examples.py`

Currently, you can store 5 gesture classes. Pressing numbers 1 to 5 will label the class.
Press 'q' to save and quit.

To identify gestures, run

`python identify_gesture.py`

Press 'i' to identify.

#Algorithm

The system uses a Nearest Neighbour algorithm to identify gestures.

The program expects the user to place a hand in the bounding box specified. The largest contour by area is calculated, and it is assumed to be the hand/gesture. cv2.matchShapes() is called on the contour in the bounding box, and each of the contours in the gesture dictionary. The similarity measure is calculated based on one of the ShapeMatchModes given here: http://docs.opencv.org/3.1.0/df/d4e/group__imgproc__c.html#gacd971ae682604ff73cdb88645725968d

A low similarity measure corresponds to a high degree of matching between two contours. The label of the class which scores lowest on the similarity measure is displayed. 

It makes sense to understand the similarity measure as a calculation of difference between two contours. Low similarity score actually equals a small difference between two contours.

composite_list.npy contains a set of predefined gestures, and the labels are contained in composite_list_labels.npy.
The images are given along with the code.


