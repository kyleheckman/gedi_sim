import sys
sys.path.append('ret/')

import numpy as np
import las_tools

# LiDAR Parameters
FWIDTH = 23	# footprint width in meters
FWHM = 15	# pulse width

ACROSS_SPACING = 600
ALONG_SPACING = 60


class Block():
	def __init__(self, img_fn, pc_fn):
		self.img_file = img_fn
		self.pc_file = pc_fn
		self.pos_matrix, self.bounds = las_tools.get_pos_matrix(img_fn)
		self.kdtree, self.photons, self.gnd_kd, self.ground = las_tools.get_pc_data(self.pc_file, generate_kdtree=True)

class Coll_Photons():
	def __init__(self, center, photons):
		self.center = center
		self.photons = photons

class Pulses():
	def __init__(self, block, coords):
		#self.pulse_centers = self.get_pulse_centers(block)
		self.pulse_centers = coords
		self.ret_photons = self.get_photons_in_pulses(block.kdtree)
		self.ret_ground = self.get_photons_in_pulses(block.gnd_kd)

	def get_pulse_centers(self, block):
		pulse_centers = []

		for indx in np.arange(int(0.5*FWIDTH), np.shape(block.pos_matrix)[0], ACROSS_SPACING):
			for indy in np.arange(int(0.5*FWIDTH), np.shape(block.pos_matrix)[1], ALONG_SPACING):
				pulse_centers.append([block.pos_matrix[indx, indy, 0], block.pos_matrix[indx, indy, 1]])

		return pulse_centers

	def get_photons_in_pulses(self, kdtree):
		returns = []

		for indx in range(np.shape(self.pulse_centers)[0]):
			photons = kdtree.query_ball_point([self.pulse_centers[indx][0], self.pulse_centers[indx][1]], r=0.5*FWIDTH)
			returns.append(Coll_Photons(self.pulse_centers[indx], photons))

		return returns