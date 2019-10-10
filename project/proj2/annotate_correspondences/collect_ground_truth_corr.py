
# CS 6476 Computer Vision, Georgia Tech
# Written by James Hays, John Lambert

# This file is completely optional for the assignment, but is a way to provide
# helpful service.

# An interactive method to specify and then save many point correspondences
# between two photographs, which will be used to generate a projective
# transformation.

# Pick a dozen corresponding points throughout the images, although more is
# better.

import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import pdb
FIGURE_HEIGHT = 6
FIGURE_WIDTH = 10
plt.rcParams["figure.figsize"] = (FIGURE_WIDTH,FIGURE_HEIGHT)

sys.path.append('../code')

from utils import load_image, show_correspondence_lines
from pathlib import Path

class CorrespondenceAnnotator(object):
	def __init__(self):

		self.image1 = load_image('./sydney_opera_house1.jpg')
		self.image2 = load_image('./sydney_opera_house2.jpg')
		self.corr_file = Path('./sydney_opera_house_correspondences.pkl')
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
		self.ax1 = ax1
		self.ax2 = ax2
		self.x1 = [] # x locations in image 1
		self.y1 = [] # y locations in image 1
		self.x2 = [] # corresponding x locations in image 2
		self.y2 = [] # corresponding y locations in image 2

	def collect_ground_truth_corr(self):
		"""
		Collect ground truth image-to-image correspondences by manually annotating them.

		This function checks if some corresponding points are already saved, and
		if so, resumes work from there.
		"""
		if self.corr_file.exists():
			self.load_pkl_correspondences()

			# The correspondences that already exist
			corr_image = show_correspondence_lines(	self.image1, self.image2, 
													np.array(self.x1), np.array(self.y1), 
													np.array(self.x2), np.array(self.y2))
		else:
			self.x1 = [] 
			self.y1 = [] 
			self.x2 = [] 
			self.y2 = [] 

		self.ax1.imshow(self.image1)
		self.ax2.imshow(self.image2)

		self.mark_corrs_with_clicks()
		self.dump_pkl_correspondences()

		corr_image = show_correspondence_lines(	self.image1, self.image2, 
												np.array(self.x1), np.array(self.y1), 
												np.array(self.x2), np.array(self.y2))
		plt.gcf().clear()
		plt.imshow(corr_image)
		plt.show()

	def load_pkl_correspondences(self):
		with open(str(self.corr_file), 'rb') as f:
			d = pickle.load(f)

		self.x1 = d['x1']
		self.y1 = d['y1']
		self.x2 = d['x2']
		self.y2 = d['y2']

	def dump_pkl_correspondences(self):
		print('saving matched points')
		data_dict = {}
		data_dict['x1'] = self.x1
		data_dict['y1'] = self.y1
		data_dict['x2'] = self.x2
		data_dict['y2'] = self.y2

		with open(str(self.corr_file), 'wb') as f:
			pickle.dump(data_dict,f)

	def mark_corrs_with_clicks(self):
		"""
		Mark correspondences with clicks
		"""
		print('Exit the matplotlib window to stop annotation.')
		title = 'Click on a point in the left window\n'
		title += 'then on a point in the right window.\n'
		title += 'Exit the matplotlib window to stop annotation.\n'
		title += 'Afterwards, you will see the plotted correspondences.'
		self.ax1.set_title(title)
		while(1):
			pt = plt.ginput(1)
			if len(pt) == 0:
				break
			x = pt[0][0]
			y = pt[0][1]

			self.ax1.scatter(x,y,30,color='r', marker='o')
			self.x1 += [x]
			self.y1 += [y]

			pt = plt.ginput(1)
			if len(pt) == 0:
				break
			x = pt[0][0]
			y = pt[0][1]

			self.ax2.scatter(x,y,30,color='r', marker='o')
			self.x2 += [x]
			self.y2 += [y]
		    
			print('({}, {}) matches to ({},{})'.format(	self.x1[-1], 
														self.y1[-1], 
														self.x2[-1], 
														self.y2[-1]))
			print('{} total points corresponded'.format(len(self.x1)))

if __name__ == '__main__':
	ca = CorrespondenceAnnotator()
	ca.collect_ground_truth_corr()
