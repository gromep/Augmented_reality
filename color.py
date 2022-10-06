import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	lower_blue = np.array([94, 80, 2])
	higher_blue = np.array([126, 255, 255])
	blue_mask = cv2.inRange(hsv, lower_blue, higher_blue)
	blue = cv2.bitwise_and(frame,frame, mask = blue_mask)
	
	lower_red = np.array([161,155,84])
	higher_red = np.array([179, 255, 255])
	red_mask = cv2.inRange(hsv, lower_red, higher_red)
	red = cv2.bitwise_and(frame,frame, mask = red_mask)
	
	lower_green = np.array([25, 52, 72])
	higher_green = np.array([102, 255, 255])
	green_mask = cv2.inRange(hsv, lower_green, higher_green)
	green = cv2.bitwise_and(frame, frame, mask = green_mask) 
	
	cv2.imshow('blue', blue)
	cv2.imshow('red', red)
	cv2.imshow('green', green)
	cv2.imshow('frame', frame)
	
	
	cv2.waitKey(5)
cap.release
cv2.destroyAllWindows()
