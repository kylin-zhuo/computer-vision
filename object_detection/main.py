import numpy as np
import cv2
import os

WINDOW_NAME = 'Image'

def get_cascades():
	return [c[:-4] for c in os.listdir('cascades') if c[-4:] == '.xml']

def get_testfiles():
	return [c for c in os.listdir('test') if c[-4:] in ('.jpg', 'png')]

def information1():
	cs = get_cascades()
	print("The choices of objects for detection: ")
	for i, c in enumerate(cs):
		print(str(i) + ": " + c)

def information2():
	fs = get_testfiles()
	print("The test files to choose:")
	for i, f in enumerate(fs):
		print(str(i) + ": " + f)


cascades = get_cascades()
test_files = get_testfiles()
param1, param2 = [1.1], [1]

COLOR = (2,255,255)

def scale_handler(val):

	global img
	global gray
	global clf

	param1[0] = max(1.1, val / 10.0)

	objects = clf.detectMultiScale(gray, max(1.1, val / 10.0), param2[0])
	display = img.copy()

	# print objects
	for (x, y, w, h) in objects:
		cv2.rectangle(display, (x,y), (x+w, y+h), (255,255,0))
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(display, cascade, (x, y+h), font, 0.5, COLOR, 2, cv2.LINE_AA)

	print("Scale: " + str(param1[0]) + "; Neighbors: " + str(param2[0]))

	cv2.imshow(WINDOW_NAME, display)


def neighbor_handler(val):
	global img
	global gray
	global clf
	global display

	param2[0] = val

	objects = clf.detectMultiScale(gray, param1[0], val)
	display = img.copy()

	# print objects
	for (x, y, w, h) in objects:
		cv2.rectangle(display, (x,y), (x+w, y+h), (255,255,0))
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(display, cascade, (x, y+h), font, 0.5, COLOR, 2, cv2.LINE_AA)

	print("Scale: " + str(param1[0]) + "; Neighbors: " + str(param2[0]))

	cv2.imshow(WINDOW_NAME, display)


if __name__ == '__main__':

	while True:

		information1()
		i = raw_input("Select an object to begin (-1 to exit): ")
		if int(i) == -1:
			break

		cascade = cascades[int(i)]
		clf = cv2.CascadeClassifier('cascades/' + cascade + '.xml')

		while True:
			information2()
			i = raw_input("Select the test file: ")
			try:
				img_path = 'test/' + test_files[int(i)]
				img = cv2.imread(img_path)
				break
			except:
				print("Error in reading image: " + img_path)

		cv2.imshow(WINDOW_NAME, img)

		while True:	
	
			print("Press s to show detection and tune the scale parameter.")
			key = cv2.waitKey()

			if key == ord('s'):
				# convert the image to grayscale and smooth it with openCV function
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				cv2.createTrackbar('Scale:', WINDOW_NAME, 1, 50, scale_handler)
				cv2.createTrackbar('Neighbors: ', WINDOW_NAME, 1, 10, neighbor_handler)

				# display = gray

			elif key == ord('0'):
				cv2.destroyWindow(WINDOW_NAME)
				break

			# elif key == ord('o'):
			# 	cv2.imwrite("output/out.jpg", output)

		cv2.destroyWindow(WINDOW_NAME)

	cv2.destroyAllWindows()

