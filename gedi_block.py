import sys
sys.path.append('src/')
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import simulation
import read_colorized_pc


# GEDI LiDAR Parameters
FWIDTH = 24	# footprint width in meters
FWHM = 15.6	# pulse width



ACROSS_SPACING = 100
ALONG_SPACING = FWIDTH

class Block():
	def __init__(self, img_fn, pc_fn):#, pos_matrix, bounds, kdtree, photons):
		self.cam_file = img_fn
		self.pc_file = pc_fn
		# self.bounds = bounds
		# self.pos_matrix = pos_matrix
		#self.photons = photons
		#self.kdtree = kdtree
		self.pos_matrix, self.bounds = simulation.get_pos_matrix(img_fn)
		self.kdtree, self.photons = read_colorized_pc.get_pc_data(pc_fn, generate_kdtree=True)
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
		#print(f'Center:{block.pulse_centers[indx]} Photons:{len(photons)}')
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

def plot_hist(ind_in_kdtree, photons):
	z = photons[ind_in_kdtree][:,2]
	valid_heights = simulation.get_height_bins(z, voxel_height=0.5)
	fig, ax = plt.subplots()
	ax.hist(z, bins=valid_heights)
	plt.show()

def plot_hist_weights(z_weights):
	valid_heights = simulation.get_height_bins(z_weights[:,0], voxel_height=0.5)
	fig, ax = plt.subplots()
	ax.hist(z_weights[:,0], bins=valid_heights, weights=z_weights[:,1])
	plt.show()

def compute_weight(photon, center, rel_weight='count', std=2*FWIDTH):
	if rel_weight == 'count':
		scale=1
	return scale*np.exp(-1*(np.square(center[0]-photon[0])+np.square(center[1]-photon[1]))/(2*np.square(std)))*(1/(std*np.sqrt(2*np.pi)))

def get_weights(sel_photons, center):
	z_weights = np.zeros((len(sel_photons),2))
	for ind in range(len(sel_photons)):
		z_weights[ind,:] = np.array([sel_photons[ind][2],compute_weight(sel_photons[ind],center,rel_weight='count',std=2)])
	return z_weights

def get_sampled_photons(pulse_returns, photons):
	sampled_ind = simulation.sample_photons_from_fp(pulse_returns[9,0],photons[pulse_returns[9,1]][:,:2],pulse_returns[9,1])
	plot_hist(sampled_ind, photons)

# for testing support functions
if __name__ == '__main__':
	data_dir = 'data/raw/SJER/2023-04/'

	img_fn = f'{data_dir}camera/2023_SJER_6_251000_4106000_image.tif'
	pc_fn = f'{data_dir}lidar/NEON_D17_SJER_DP1_251000_4106000_classified_point_cloud_colorized.laz'

	#pos_matrix, bounds = simulation.get_pos_matrix(img_fn)
	#kdtree, photons = read_colorized_pc.get_pc_data(pc_fn, generate_kdtree=True)

	block = Block(img_fn, pc_fn,)# pos_matrix, bounds, kdtree, photons)
	pulse_returns = get_photons_in_pulses(block)
	#print(pulse_returns)
	#tmp = 9
	#photons_w_weights = get_weights(block.photons[pulse_returns[tmp,1]],pulse_returns[tmp,0])
	

	#print(pulse_returns[tmp,0])

	#plot_hist(pulse_returns[tmp,1], block.photons)
	#plot_hist_weights(photons_w_weights)
	#get_sampled_photons(pulse_returns,block.photons)
	#plot_photons(pulse_returns[tmp,1], block.photons)

	

	im = rasterio.open(img_fn)
	
	fig, ax = plt.subplots()
	ax.imshow(im.read(1), cmap='grey', extent=(block.bounds.left, block.bounds.right, block.bounds.bottom, block.bounds.top), origin='lower')
	for center in block.pulse_centers:
		circ = plt.Circle(center, 0.5*FWIDTH, edgecolor='red', facecolor='none', linewidth=1)
		ax.add_patch(circ)
	ax.set_xlim(block.bounds.left, block.bounds.right)
	ax.set_ylim(block.bounds.bottom, block.bounds.top)
	plt.show()