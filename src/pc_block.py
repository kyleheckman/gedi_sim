import sys
sys.path.append('src/')

import numpy as np
import las_tools

class Block():
	def __init__(self, img_fn, pc_fn, downsample=None):
		self.img_file = img_fn
		self.pc_file = pc_fn
		self.pos_matrix, self.bounds = las_tools.get_pos_matrix(img_fn)
		self.kdtree, self.photons, self.gnd_kd, self.ground = las_tools.get_pc_data(self.pc_file, generate_kdtree=True, downsample=downsample)

class Coll_Photons():
	def __init__(self, center, photons):
		self.center = center
		self.photons = photons

class Pulses():
	def __init__(self, block, coords, config):
		self.pulse_centers = coords
		self.ret_photons = self.get_photons_in_pulses(block.kdtree, config)
		self.ret_ground = self.get_photons_in_pulses(block.gnd_kd, config)

	def get_photons_in_pulses(self, kdtree, config):
		returns = []

		for indx in range(np.shape(self.pulse_centers)[0]):
			photons = kdtree.query_ball_point([self.pulse_centers[indx][0], self.pulse_centers[indx][1]], r=0.5*config['sim_config']['gedi_config']['fwidth'])
			returns.append(Coll_Photons(self.pulse_centers[indx], photons))

		return returns