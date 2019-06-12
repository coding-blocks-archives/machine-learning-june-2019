import cv2


#Read a Video Stream and Display It

#Camera Object
cam = cv2.VideoCapture(0)

while True:
	ret,frame = cam.read()
	if ret==False:
		print("Something Went Wrong!")
		continue

	key_pressed = cv2.waitKey(1)&0xFF
	if key_pressed == ord('q'):
		break

	cv2.imshow("Video Title",frame)


cam.release()
cv2.destroyAllWindows()	