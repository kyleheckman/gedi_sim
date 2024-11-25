import sys
sys.path.append('/src')

import os
from omegaconf import OmegaConf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import utm
import rasterio

def rh_vals(waveform, config):
	if np.sum(waveform) == 0:
			return None

	energy_int = np.array(np.cumsum(waveform)/np.sum(waveform))
	energy_int = energy_int

	energy_marks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.98]

	indx = [np.argwhere(energy_int>=i).min() for i in energy_marks]
	height_adjusted = (indx-np.min(np.argwhere(waveform > 0)))*config['sim_config']['gedi_config']['resolution']
	return indx, height_adjusted

def calc_agbd(rh, type):
	if type == 'ent_na':
		agbd = 1.013 * np.square(-114.355 + (8.401 *  np.sqrt(rh[3] + 100)) + (3.346 *  np.sqrt(rh[-1] + 100)))
		return agbd
	elif type == 'dbt_na':
		agbd = 1.052 * np.square(-120.777 + (5.508 *  np.sqrt(rh[1] + 100)) + (3.955 *  np.sqrt(rh[-1] + 100)))
		return agbd

if __name__ == '__main__':
	data_dir = 'data/raw/'

	bl = '252000_4109000'

	tif_tgt = f'SJER/2023-04/camera/2023_SJER_6_{bl}_image.tif'

	output = 'output_hhdc/'
	target = f'{output}SJER_{bl}_hhdc_20x20_gridded.npy'

	diz_path = os.path.dirname(os.path.realpath(__file__))
	config = OmegaConf.load(f'{diz_path}/config/sim_config.yaml')

	hhdc = np.load(target)
	biomass = np.zeros(np.shape(hhdc)[:2])

	for indx in range(np.shape(hhdc)[0]):
		for indy in range(np.shape(hhdc)[1]):
			ind, height_adjusted = rh_vals(hhdc[indx,indy], config)

			agbd = calc_agbd(height_adjusted, 'ent_na')
			biomass[np.shape(biomass)[0]-indx-1, indy] = agbd


	img = rasterio.open(f'{data_dir}{tif_tgt}')

	fig, ax = plt.subplots(2,2)
	ax[0,0].imshow(img.read(1), cmap='Greys')
	ax[0,0].imshow(biomass, interpolation='none', cmap=matplotlib.colormaps['plasma'], extent=[0,10000,10000,0], alpha=0.6)
	ax[0,0].xaxis.tick_top()

	ax[0,1].imshow(biomass, interpolation='none', cmap=matplotlib.colormaps['plasma'])
	ax[0,1].set_xticks([])
	ax[0,1].set_yticks([])

	ax[1,0].imshow(img.read(1), cmap='Greys')
	ax[1,0].set_xticks([])
	ax[1,0].set_yticks([])
	
	ax[1,1].remove()
	
	plt.subplots_adjust(wspace=-0.25, hspace=0.2)

	plt.show()
	plt.close()