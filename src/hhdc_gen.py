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

	shift = int((z_min - block_z_min) / config['sim_config']['gedi_config']['resolution'])
	column = np.zeros((config['hhdc_config']['height_bins']), dtype=np.uint16)

	print(f'Shift {shift}')

	column[shift:shift+len(col_wf)] = col_wf[::-1]

	return column
	
def disp_dem(hhdc_slice, bounds, fn, indx, z_min, config):
	bins = config['hhdc_config']['height_bins']
	resolution = config['sim_config']['gedi_config']['resolution']

	fig, ax = plt.subplots()

	ax.imshow(hhdc_slice.T[::-1], aspect=5, interpolation='none', cmap=matplotlib.colormaps['Greens'], extent=[bounds.left,bounds.right,0,bins*resolution])
	ax.set_title(f'Digital Elevation Model for Across Track Slice {indx}')
	ax.set_ylabel(f'Height from Reference Ground: {np.round(z_min,1)} [m]')
	ax.set_xlabel(f'Along Track Coordinate [m]')

	plt.savefig(fn)
	plt.close()

if __name__ == '__main__':
	target = 'data/raw/SJER/2023-04/'
	bl = '252000_4109000'
	output = 'output_hhdc/'

	img_fn = f'{target}camera/2023_SJER_6_{bl}_image.tif'
	pc_fn = f'{target}lidar/NEON_D17_SJER_DP1_{bl}_classified_point_cloud_colorized.laz'

	diz_path = os.path.dirname(os.path.realpath(__file__))
	config = OmegaConf.load(f'{diz_path}/config/sim_config.yaml')

	block = pc_block.Block(img_fn, pc_fn)
	centers = get_hhdc_centers(block, config)

	hhdc = np.zeros((np.shape(centers)[0], np.shape(centers)[1], config['hhdc_config']['height_bins']), dtype=np.uint16)
	centers = np.concatenate(centers, axis=0)

	pulses = pc_block.Pulses(block, centers, config)

	block_z_min = np.min(block.photons[:,2])

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
			fn = f'{output}dem_{bl.split('_')[0]}-{bl.split('_')[1]}_slice-{(iy*20)+10}'
			disp_dem(hhdc[iy], block.bounds, fn, iy, block_z_min, config)

	fn = f'{output}SJER_{bl}_hhdc_20x20_gridded.npy'
	np.save(fn, hhdc)