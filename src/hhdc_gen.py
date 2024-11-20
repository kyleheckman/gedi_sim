import sys
sys.path.append('src/')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf

import simulation
import pc_block

def get_hhdc_centers(block, config):
	square_size = config['hhdc_config']['square_size']
	x_dim = np.arange(block.bounds.left+(square_size/2), block.bounds.right, step=square_size)
	y_dim = np.arange(block.bounds.bottom+(square_size/2), block.bounds.top, step=square_size)

	x_coords, y_coords = np.meshgrid(x_dim, y_dim)

	centers = np.dstack((x_coords, y_coords))
	
	return centers

def prep_waveform(waveform, z_min, block_z_min, config):
	col_wf = waveform.nfw / np.sum(waveform.nfw)
	col_wf = col_wf[np.min(np.argwhere(col_wf != 0)):waveform.rh_idx[0]+1]
	print(len(col_wf))


	shift = int((z_min - block_z_min) / config['sim_config']['gedi_config']['resolution'])
	column = np.zeros((config['hhdc_config']['height_bins']))

	column[shift:shift+len(col_wf)] = col_wf

	return column
	
def disp_dem(hhdc_slice, bounds, config):
	bins = config['hhdc_config']['height_bins']
	resolution = config['sim_config']['gedi_config']['resolution']

	fig, ax = plt.subplots()

	ax.imshow(hhdc_slice[::-1].T, aspect=12, interpolation='none', cmap=matplotlib.colormaps['Greens'], extent=[bounds.left,bounds.right,0,bins*resolution])

	plt.show()
	plt.close()

if __name__ == '__main__':
	target = 'data/raw/SJER/2023-04/'
	bl = '252000_4109000'
	output = 'output_hhdc/'

	img_fn = f'{target}camera/2023_SJER_6_{bl}_image.tif'
	pc_fn = f'{target}lidar/NEON_D17_SJER_DP1_{bl}_classified_point_cloud_colorized.laz'

	diz_path = os.path.dirname(os.path.realpath(__file__))
	config = OmegaConf.load(f'{diz_path}/config/sim_config.yaml')
	print(config)

	block = pc_block.Block(img_fn, pc_fn)
	centers = get_hhdc_centers(block, config)

	hhdc = np.zeros((np.shape(centers)[0], np.shape(centers)[1], config['hhdc_config']['height_bins']))
	centers = np.concatenate(centers, axis=0)

	pulses = pc_block.Pulses(block, centers, config)

	block_z_min = np.min(block.photons[:,2])
	#block_z_max = np.max(block.photons[:,2])

	for indx in range(np.shape(centers)[0]):
		print(f'Processing block {indx} / {np.shape(centers)[0]} ...')
		if len(pulses.ret_photons[indx].photons) == 0:
			continue

		sim_wf = simulation.Waveform(block.photons, pulses.ret_photons[indx], block.ground, pulses.ret_ground[indx], config)
		z_min = np.min(block.photons[pulses.ret_photons[indx].photons][:,2])

		column = prep_waveform(sim_wf, z_min, block_z_min, config)
		
		ix = indx % np.shape(hhdc)[0]
		iy = int(indx / np.shape(hhdc)[1])

		hhdc[iy, ix, :] = column

		if (indx + 1) % np.shape(hhdc)[0] == 0:

			disp_dem(hhdc[iy], block.bounds, config)