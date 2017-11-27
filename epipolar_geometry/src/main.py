import cv2
import numpy as np
import os
import sys


def normalize(vals, mu, sigma):
	if sigma == 0:
		return vals
	return [(v-mu)/sigma for v in vals]

class AS5:

	def __init__(self):
		self.marked = []
		self.image = None
		self.width, self.height = 0, 0

	def load_data(self, file_path):	
		left, right = os.listdir(file_path)
		left = cv2.imread(os.path.join(file_path, left), cv2.IMREAD_GRAYSCALE)
		right = cv2.imread(os.path.join(file_path, right), cv2.IMREAD_GRAYSCALE)

		assert(left.shape == right.shape)
		return left, right


	def mark(self, event, x, y,flag, params):
		if event == cv2.EVENT_LBUTTONUP:
	 		self.marked.append((x,y))
			cv2.line(self.image, (x,y), (x+1,y+1), (255, 255, 2), 2)
			cv2.imshow("image", self.image)
			print("Mark: " + str(x) + ',' + str(y))


	def select(self):
		clone = self.image.copy()
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", self.mark)
		 
		while True:
			cv2.imshow("image", self.image)
			key = cv2.waitKey(1) & 0xFF
		 
			if key == ord("i"):
				self.image = clone.copy()
				self.marked = []
				print("Reset. Cleaned up the marked.")
				cv2.setMouseCallback("image", self.mark)
			elif key == ord("e"):
				break

		self.image = clone
		cv2.destroyWindow("image")

	def choose(self, event, x, y, flag, params):

		if event == cv2.EVENT_LBUTTONUP:
	 		if x >= self.width:
	 			# the selected point is in the right image
	 			cv2.line(self.image, (x,y), (x+1,y+1), (255, 0, 255), 2)
	 			x -= self.width
	 			p_r = np.matrix([[x],[y],[1]])
	 			eline = self.F.T * p_r
	 			A, B, C = eline[0,0], eline[1,0], eline[2,0]
	 			x1 = 0
	 			y1 = -int(C/B)
	 			x2 = self.width - 1
	 			y2 = int(y1 - A/B * x2)
	 			cv2.line(self.image, (x1, y1),(x2, y2),(255, 255, 1), 1)
	 		else:
	 			# the selected point is in the left image
	 			cv2.line(self.image, (x,y), (x+1,y+1), (255, 0, 255), 2)
	 			p_l = np.matrix([[x],[y],[1]])
	 			eline = self.F * p_l
	 			A, B, C = eline[0,0], eline[1,0], eline[2,0]
	 			x1 = self.width
	 			y1 = -int(C/B)
	 			x2 = self.width - 1
	 			y2 = int(y1 - A/B * x2)
	 			x2 += self.width
	 			cv2.line(self.image, (x1, y1),(x2, y2),(255, 255, 1), 1)
			
			cv2.imshow("epipolarLine", self.image)
			


	def display_epipolar_line(self):
		clone = self.image.copy()
		cv2.namedWindow("epipolarLine")
		cv2.setMouseCallback("epipolarLine", self.choose)
		 
		while True:
			cv2.imshow("epipolarLine", self.image)
			key = cv2.waitKey(1) & 0xFF
		 
			if key == ord("i"):
				self.image = clone.copy()
				print("Reset. Cleaned up the marked.")
				cv2.setMouseCallback("epipolarLine", self.choose)

			elif key == ord("e"):
				break

		self.image = clone
		cv2.destroyWindow("epipolarLine")


	def main(self, file_path):
		left, right = self.load_data(file_path)

		assert(left.shape == right.shape)
		self.height, self.width = left.shape

		self.image = np.concatenate((left, right), axis=1)
		self.marked = []
		self.select()

		assert(len(self.marked) >= 18)

		p_left, p_right = [], []

		for i,p in enumerate(self.marked):
			if i & 1:
				p_right.append((p[0]-self.width,p[1]))
			else:
				p_left.append(p)

		# print p_left, p_right

		# normalize
		p_left_x = [p[0] for p in p_left]
		p_left_y = [p[1] for p in p_left]

		p_right_x = [p[0] for p in p_right]
		p_right_y = [p[1] for p in p_right]

		mu_left_x = np.mean(p_left_x)
		mu_left_y = np.mean(p_left_y)
		mu_right_x = np.mean(p_right_x)
		mu_right_y = np.mean(p_right_y)

		sigma_left_x = np.std(p_left_x)
		sigma_left_y = np.std(p_left_y)
		sigma_right_x = np.std(p_right_x)
		sigma_right_y = np.std(p_right_y)

		# Construct the matrix M
		M_left = np.matrix([[1/sigma_left_x,0,0],[0,1/sigma_left_y,0],[0,0,1]]) * np.matrix([[1,0,-mu_left_x],[0,1,-mu_left_y],[0,0,1]])
		M_right = np.matrix([[1/sigma_right_x,0,0],[0,1/sigma_right_y,0],[0,0,1]]) * np.matrix([[1,0,-mu_right_x],[0,1,-mu_right_y],[0,0,1]])

		p_left_x = normalize(p_left_x, mu_right_x, sigma_left_x)
		p_left_y = normalize(p_left_y, mu_left_y, sigma_left_y)
		p_right_x = normalize(p_right_x, mu_right_x, sigma_right_x)
		p_right_y = normalize(p_right_y, mu_right_y, sigma_right_y)


		A = []
		for i in range(9):
			xl, xr = p_left_x[i], p_right_x[i]
			yl, yr = p_left_y[i], p_right_y[i]
			A.append([xl*xr, xl*yr, xl, yl*xr, yl*yr, yl, xr, yr, 1])

		A = np.matrix(A)
		# print A

		# Compute F from SVD of A
		_,_,v = np.linalg.svd(A)
		F = v[-1:].reshape((3,3))

		# make sure the F is rank 2
		u,D,v = np.linalg.svd(F)
		D[2] = 0
		D = np.diag(D)
		# F = np.dot(np.dot(u,D),v)
		F = u * D * v
		# recover F from the normalization F_p 
		self.F = M_right.T * F * M_left

		# solve e_r
		u,D,v = np.linalg.svd(self.F)
		e_l = v[-1:]

		u,D,v = np.linalg.svd(self.F.T)
		e_r = v[-1:]

		print("Left epipole: ", e_l[0,0]/e_l[0,2], e_l[0,1]/e_l[0,2])
		print("Right epipole:", e_r[0,0]/e_r[0,2], e_r[0,1]/e_r[0,2])

		self.display_epipolar_line()


if __name__ == '__main__':

	folders = [f for f in os.listdir('../data') if os.isdir(f)]
	as5 = AS5()
	while True:
		for i,f in enumerate(folders):
			print(str(i) + ': ' + f)
		option = int(raw_input("select the image (-1 to exit): "))
		if option == -1:
			break
		f = folders[option]
		file_path = os.path.join('../data', f)
		as5.main(file_path)
