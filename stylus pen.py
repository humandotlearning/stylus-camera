#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 19:25:19 2018

@author: nithin
"""

import numpy as np
import cv2
import argparse
from collections import deque

def main():
	cap=cv2.VideoCapture(0)

	pts = deque(maxlen=64)

	#normally the hsv value has h=0-360 ,s=0-255, v=0-255 but in opencv it takes h=0-180 degrees
	Lower_red = np.array([165,50,50])
	Upper_red = np.array([180,255,255])
	while True:
		ret, img=cap.read()
		img=cv2.flip(img,1)
		hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		kernel=np.ones((5,5),np.uint8)
		mask=cv2.inRange(hsv,Lower_red,Upper_red)
		mask = cv2.erode(mask, kernel, iterations=2)
		#mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
		mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
		mask = cv2.dilate(mask, kernel, iterations=1)
		res=cv2.bitwise_and(img,img,mask=mask)
		cnts,heir=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
		center = None

		if len(cnts) > 0:
			c = max(cnts, key=cv2.contourArea)
			(x, y), radius = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			if radius > 5:
				cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
				cv2.circle(img, center, 5, (0, 0, 255), -1)

		pts.appendleft(center)
		for i in range (1,len(pts)):
			if pts[i-1]is None or pts[i] is None:
				continue
			thick = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
			cv2.line(img, pts[i-1],pts[i],(0,0,225),thick)


		cv2.imshow("Frame", img)
		cv2.imshow("mask",mask)
		cv2.imshow("res",res)

		#press 'q' to exit
		if cv2.waitKey(1) & 0xFF== ord('q'):
			break
	# cleanup the camera and close any open windows
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
