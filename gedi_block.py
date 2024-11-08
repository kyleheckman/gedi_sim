import sys
sys.path.append('src/')
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import las_tools


# GEDI LiDAR Parameters
FWIDTH = 24	# footprint width in meters
FWHM = 15.6	# pulse width

ACROSS_SPACING = 100
ALONG_SPACING = FWIDTH

class Block():
	def __init__(self, img_fn, pc_fn):
		self.cam_file = img_fn
		self.pc_file = pc_fn
		self.pos_matrix, self.bounds = las_tools.get_pos_matrix(img_fn)
		self.kdtree, self.photons = las_tools.get_pc_data(pc_fn, generate_kdtree=True)
		self.pulse_centers = self.get_pulse_centers(self.bounds, self.pos_matrix)
		self.pulse_returns = get_photons_in_pulses(self)

	def get_pulse_centers(self, bounds, pos_matrix):
		pulse_centers = []

		for indx in np.arange(int(0.5*FWIDTH), np.shape(pos_matrix)[0] , ACROSS_SPACING):
			for indy in np.arange(int(0.5*FWIDTH), np.shape(pos_matrix)[1], ALONG_SPACING):
				pulse_centers.append([pos_matrix[indx,indy, 0],pos_matrix[indx,indy,1]])

		return pulse_centers

def get_photons_in_pulses(block):
	pulse_returns = np.zeros([np.shape(block.pulse_centers)[0], np.shape(block.pulse_centers)[1]], dtype='object')

	for indx in range(np.shape(block.pulse_centers)[0]):
		photons = block.kdtree.query_ball_point([block.pulse_centers[indx][0], block.pulse_centers[indx][1]], r=0.5*FWIDTH)
		pulse_returns[indx,0], pulse_returns[indx,1] = [block.pulse_centers[indx][0], block.pulse_centers[indx][1]], photons

	return pulse_returns

def plot_photons(ind_in_kdtree, photons):
	photons_to_plot = photons[ind_in_kdtree]

	colors = photons_to_plot[:,3:6]
	colors_norm = (colors - np.min(colors))/np.ptp(colors)

	fig = plt.figure(figsize=(6,6))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(photons_to_plot[:,0],photons_to_plot[:,1],photons_to_plot[:,2], color=colors_norm,s=4)
	ax.set_zlim3d(-10,50)
	ax.autoscale()
	
	plt.show()


if __name__ == '__main__':
	data_dir = 'data/raw/SJER/2023-04/'
	img_fn = f'{data_dir}camera/2023_SJER_6_251000_4106000_image.tif'
	pc_fn = f'{data_dir}lidar/NEON_D17_SJER_DP1_251000_4106000_classified_point_cloud_colorized.laz'

	block = Block(img_fn, pc_fn,)
	pulse_returns = get_photons_in_pulses(block)

	im = rasterio.open(img_fn)
	
	fig, ax = plt.subplots()
	ax.imshow(im.read(1), cmap='grey', extent=(block.bounds.left, block.bounds.right, block.bounds.bottom, block.bounds.top), origin='lower')
	for center in block.pulse_centers:
		circ = plt.Circle(center, 0.5*FWIDTH, edgecolor='red', facecolor='none', linewidth=1)
		ax.add_patch(circ)
	ax.set_xlim(block.bounds.left, block.bounds.right)
	ax.set_ylim(block.bounds.bottom, block.bounds.top)
	plt.show()