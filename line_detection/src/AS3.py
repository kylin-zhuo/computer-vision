import cv2 
import numpy as np 

import sys
import math

from scipy import signal
from collections import defaultdict

import random

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

dis = image

xd = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
yd = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
votes = None
binary_edge = None
canny_thres_low = 180
canny_thres_high = 220
window_name = 'AS3'
threshold_peak = 0.4
threshold_lse = 4
grid = 5


def normalize(image, minv, maxv):
	_maxv, _minv = image.max(), image.min()
	ret = (image - _minv) * (maxv - minv) / (_maxv - _minv) + minv
	return ret.astype(np.uint8)


def binary_edge(image, thres=130):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# gradientX = cv2.Sobel(image, -1, 1, 0)
	# gradientY = cv2.Sobel(image, 1, -1, 0)
	gradientX = signal.convolve2d(gray, xd)
	gradientY = signal.convolve2d(gray, yd)

	output = np.sqrt(gradientX ** 2 + gradientY ** 2)
	output = 255 * (normalize(output, 0, 255) // thres)
	return output.astype(np.uint8)


# The function for applying the Hough transform to detect straight lines
def hough(image, bin_theta = 1):
	# set the rho and theta ranges
	thetas = np.deg2rad(np.arange(0, 180)[::bin_theta])
	# height corresponds to y direction and width to x direction
	heihgt, width = image.shape

	diag_len = int(np.ceil(np.sqrt(heihgt ** 2 + width ** 2)))
	rhos = np.linspace(-diag_len, diag_len, 2 * diag_len + 1).astype(int)

	cos_t, sin_t = np.cos(thetas), np.sin(thetas)
	n_thetas, n_rhos = len(thetas), len(rhos)

	# define the votes
	# mapping:
	# votes[0] -> rho == -diag_len
	# votes[1] -> rho == -(diag_len-1)
 	votes = np.zeros((n_rhos, n_thetas), dtype=int)
	y_indxs, x_indxs = np.nonzero(image)

	# vote
	for x, y in zip(x_indxs, y_indxs):
		for t in range(n_thetas):
			rho = int(x * cos_t[t] + y * sin_t[t]) + diag_len
			votes[rho][t] += 1

	return votes, thetas, rhos, diag_len


def local_max(votes, grid=5, threshold=0.0):

	res_i = []
	res_j = []

	m, n = votes.shape

	res = np.reshape(np.zeros(m*n), (m, n)).astype(int)

	for i in range(m):
		for j in range(n):
			min_i, max_i = max(0, i-grid), i + grid + 1
			min_j, max_j = max(0, j-grid), j + grid + 1

			if votes[i][j] > threshold and np.all(votes[i][j] >= votes[min_i:max_i, min_j:max_j]):
				res_i.append(i)
				res_j.append(j)
				res[i][j] = 255

	return res_i, res_j, res


def vote(image, bin_theta=1):
	binary_edge = cv2.Canny(image, canny_thres_low, canny_thres_high)
	votes, thetas, rhos, diag_len = hough(binary_edge, bin_theta)
	dis = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height, width = dis.shape

	votes[diag_len] = np.zeros(len(thetas))
	
	# len(peaks) represent the number of lines in the image
	peaks_i, peaks_j, peaks = local_max(votes, grid, votes.max() * threshold_peak)

	# peaks = (votes / (int(votes.max()*0.8))) * 255
	# peaks_indx_x, peaks_indx_y = np.nonzero(peaks)

	for x, y in zip(peaks_i, peaks_j):
		rho = x - diag_len
		theta = thetas[y]
		if theta:
			for i in range(width):
				j = int((rho - np.cos(theta) * i) / np.sin(theta))
				if 0 <= j < height:
					dis[j][i] = 255

	return dis

def canny_low(val):
	global dis
	global canny_thres_low
	canny_thres_low = val
	dis = cv2.Canny(image, val, canny_thres_high)
	cv2.imshow(window_name, dis)

def canny_high(val):
	global dis
	global canny_thres_high
	canny_thres_high = val
	dis = cv2.Canny(image, canny_thres_low, val)
	cv2.imshow(window_name, dis)

def hough_handler(val):
	global dis
	image = cv2.imread(filename)
	dis = vote(image, val)
	cv2.imshow(window_name, dis)


if __name__ == '__main__':

	while True:

		key = cv2.waitKey(10)

		if key == ord('i'):
			image = cv2.imread(filename)
			dis = image

		if key == ord('e'):
			cv2.createTrackbar('Low:', window_name, 0, 255, canny_low)
			cv2.createTrackbar('High:', window_name, 0, 255, canny_high)
			binary_edge = cv2.Canny(image, canny_thres_low, canny_thres_high)
			dis = binary_edge

		if key == ord('s'):
			cv2.imwrite('../output/output.jpg', image)

		if key == ord('h'):
			# votes, thetas, rhos = hough(binary_edge(image))

			dis = vote(image)
			cv2.createTrackbar('Bin size', window_name, 1, 10, hough_handler)
			cv2.imwrite('../output/hough_line.jpg', dis)
			# cv2.imshow(window_name, image)

		if key == ord('l'):
			
			# Refine the lines with least square error fitting
			# binary edges that are candidates for assigning to a line or none
			binary_edge = cv2.Canny(image, *Canny_thres)

			point_i, point_j = np.nonzero(binary_edge)

			votes, thetas, rhos, diag_len = hough(cv2.Canny(image, *Canny_thres))
			peaks_i, peaks_j, peaks = local_max(votes, grid, votes.max() * threshold_peak)

			dic = defaultdict(list)

			# the length of peaks_i or peaks_j represent the number lines
			for i, j in zip(peaks_i, peaks_j):
				rho = i - diag_len
				theta = thetas[j]
				for y, x in zip(point_i, point_j):
					# the distance from a point (x,y) to a line for a specifc paramter set (theta,rho) is:
					# rho = x cos(theta) + y sin(theta)
					dist = x * np.cos(theta) + y * np.sin(theta)
					if abs(rho - dist) < threshold_lse:
						dic[(rho, theta)].append((x, y))

			# dis = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			dis = cv2.imread(filename)

			height, width, _ = dis.shape

			for key in dic:

				r = random.randint(0,255)
				g = random.randint(0,255)
				b = random.randint(0,255)

				# Do the least square error fitting
				# formula of the line is Ax + By + C = 0
				# \Sigma{x_i * x_i^T} * [A, B, C]^T = 0
				# Where x_i = (x, y, 1)
				# (x, y) is a sample point

				# dic[key] are all the points that are assigned to the line

				# G is the correlation matrix
				G = np.matrix([[0]*3]*3)

				for y, x in dic[key]:

					x_i = np.matrix([y, x, 1]).T
					G += x_i * x_i.T
					dis[x][y] = np.array([b,r,g]).astype(np.uint8)

				# get the eigenvalues and eigenvectors
				w, v = np.linalg.eig(G)

				# the eigenvector belonging to the smallest eigenvalue contains the coefficients
				ev = v[:, np.argmin(abs(w))].T
				A = ev[0,0]
				B = ev[0,1]
				C = ev[0,2]
				if B:
					# y = -A/B x - C/B
					for x in range(width):
						y = int(-1.0 * A/B * x - 1.0 * C/B)
						if 0 <= y < height:
							dis[y][x] = np.array([255,255,255]).astype(np.uint8)

				cv2.imwrite('../output/lse.jpg', dis)


		# Add the mode to display the paramter plane
		if key == ord('p'):
			votes, thetas, rhos, diag_len = hough(cv2.Canny(image, *Canny_thres))
			dis = normalize(votes, 0, 255)

		if key == ord('0'):
			break

		cv2.imshow(window_name, dis)

	