import cv2
import numpy as np
import sys
from scipy import signal
import math

if len(sys.argv) == 2:
	filename = sys.argv[1]
	image = cv2.imread(filename)
elif len(sys.argv) < 2:
	cap = cv2.VideoCapture(0)
	retval, image = cap.read()
	filename = "from camera"
else:
	print "error"

print "Read the image " + filename
print "Image shape:", image.shape

cv2.imshow("Lin", image)

shuffle = 0
yd = np.array([[-1,-1],[1,1]])
xd = np.array([[-1,1],[-1,1]])

def normalize(image, minv, maxv):
	maxv, minv = image.max(), image.min()
	image = (image - minv) * maxv / (maxv - minv)
	return image

def gray_scale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def image_grad(image, d):
	image_grad = signal.convolve2d(image, d, boundary='symm', mode='same')
	return image_grad

def smoothingHandler1(n):
	global display
	# n = 5
	# kernel = np.ones((n,n), np.float32) / (n*n)
	display = cv2.blur(image_g, (n,n))
	cv2.imshow("Lin", display) 

def smoothingHandler2(n):
	global display
	kernel = np.ones((n,n), np.float32) / (n*n)
	# g = gray_scale(image)
	display = cv2.filter2D(image_g, -1, kernel)
	cv2.imshow("Lin", display) 

image_g = gray_scale(image)
display = image

def drawGradHandler(N):
	global display
	# global grad_x
	# global grad_y
	grad_x = image_grad(image_g, xd)
	grad_y = image_grad(image_g, yd)
	K = 10
	W = 1
	h, w = display.shape
	i = 0
	display = gray_scale(image)
	while i < h:
		j = 0
		while j < w:
			gx, gy = grad_x[i][j], grad_y[i][j]
			if gx != 0 or gy != 0:
				lenx = gx/np.sqrt(gx*gx+gy*gy) * K
				leny = gy/np.sqrt(gx*gx+gy*gy) * K
				cv2.line(display, (j, i), (int(j+lenx), int(i+leny)), (0,0,0), W)
			j += N
		i += N
	cv2.imshow("Lin", display)


def rotate_about_center(image, deg, scale=1.):
	h, w = image.shape
	rangle = np.deg2rad(deg) 
	nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
	nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale

	rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), deg, scale)
	# calculate the move from the old center to the new center combined
	# with the rotation
	rot_move = np.dot(rot_mat, np.array([(nw-w) * 0.5, (nh-h) * 0.5, 0]))
	# the move only affects the translation, so update the translation
	# part of the transform
	rot_mat[0,2] += rot_move[0]
	rot_mat[1,2] += rot_move[1]
	return cv2.warpAffine(image, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def rotateHandler(deg):
	global display
	display = gray_scale(image)
	display = rotate_about_center(display, deg)
	cv2.imshow("Lin", display)


while True:	
	
	key = cv2.waitKey()
	
	if key == ord('i'):
		# reload the image
		image = cv2.imread(filename)
		cv2.destroyWindow("Lin")
		display = image
		image_g = gray_scale(image)
		print("Reloaded the image")
	
	if key == ord('w'):
		# save the current image into 'output.jpg'
		cv2.imwrite('../output/output.jpg', display)
		print("Saved image to ../output/output.jpg")
	
	if key == ord('g'):
		# convert tHe image to greyscale
		image_g = gray_scale(image)
		display = image_g
		print("Converted the image to greyscale")

	if key == ord('G'):
		# convert to greyscale using own function
		# 0.2989 * R + 0.5870 * G + 0.1140 * B
		image_G = image[:,:,0]*0.299 + image[:,:,1]*0.587 + image[:,:,2]*0.114
		image_G = image_G.astype(np.uint8)
		display = image_G
		print("convert to greyscale using own function")
	
	if key == ord('c'):
		# cycle through color channels
		b,g,r = cv2.split(image)
		display = (b, g, r)[shuffle]
		name = ('blue', 'green', 'red')[shuffle]
		print("Cycling on the %s channel" % name)
		shuffle = (shuffle+1) % 3

	if key == ord('s'):
		# convert the image to grayscale and smooth it with openCV function
		image_g = cv2.blur(gray_scale(image), (1,1))
		cv2.createTrackbar('Smoothing amount:', 'Lin', 1, 20, smoothingHandler1)
		display = image_g

	if key == ord('S'):
		# convert the image to grayscale and smooth it with own function
		image_g = cv2.blur(gray_scale(image), (1,1))
		cv2.createTrackbar('Smoothing amount:', 'Lin', 1, 20, smoothingHandler2)
		display = image_g

	if key == ord('d'):
		# downsample by factor of 2, without smoothing
		image_d = cv2.resize(image, (image.shape[1]/2, image.shape[0]/2))
		display = image_d

	if key == ord('D'):
		# downsample by factor of 2, with smoothing
		image_D = cv2.resize(image, (image.shape[1]/2, image.shape[0]/2))
		image_D = cv2.blur(image_D, (5,5))
		display = image_D

	if key == ord('x'):
		image_x = image_grad(image_g, xd)
		image_x = normalize(image_x, 0, 255)
		display = image_x.astype(np.uint8)

	if key == ord('y'):
		image_y = image_grad(image_g, yd)
		image_y = normalize(image_y, 0, 255)
		display = image_y.astype(np.uint8)

	if key == ord('m'):
		image_x = image_grad(image_g, xd)
		image_y = image_grad(image_g, yd)
		image_m = np.sqrt(image_x * image_x + image_y * image_y)
		image_m = normalize(image_m, 0, 255)
		display = image_m.astype(np.uint8)

	if key == ord('p'):
		# convert the image to grayscale and plot the gradient vectors every N pixels
		grad_x = image_grad(image_g, xd)
		grad_y = image_grad(image_g, yd)
		display = image_g
		cv2.createTrackbar('N:', 'Lin', 5, 8, drawGradHandler)

	if key == ord('r'):
		# rotate the image using an angle of Theta
		cv2.createTrackbar('Degree:', 'Lin', 0, 360, rotateHandler)
		display = image_g

	if key == ord('h'):
		# display short description of the program
		print("This program implement the basic manipulation of computer vision.")
		print("The keys it supports is: ")
		print("'i': reload the original image")
		print("'w': save the current image into the file 'output.jpg'")
		print("'g': convert the image to grayscale using openCV conversion function")
		print("'G': convert the image to grayscale using own implementation")
		print("'c': cycle through the color channels of the image showing a different channel every time")
		print("'s': convert the image to grayscale and smooth it using the openCV function")
		print("'S': convert the image to grayscale and smooth it using own filter")
		print("'d': downsample the image by a factor of 2 without smoothing")
		print("'D': downsample the image by a factor of 2 with smoothing")
		print("'x': convert the image to grayscale and perform convolution with an x derivative filter")
		print("'y': convert the image to grayscale and perform convolution with an y derivative filter")
		print("'m': show the magnitude of the gradient")
		print("'p': convert the image to grayscale and plot the gradient vector every N pixels")
		print("'r': convert the image to grayscale and rotate it using a specific angle")
		print("'0': exit")

	if key == ord('0'):
		cv2.destroyWindow("Input Image")
		break

	cv2.imshow("Lin", display)
	






