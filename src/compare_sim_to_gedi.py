import sys
sys.path.append('src/')
import os
import numpy as np
import matplotlib.pyplot as plt
import utm
import h5py
import scipy.stats as stats

import gedi_block
import bh_sim
import las_tools


l1bdir = 'data/gedi/GEDI01_B_002-20241108_175212/'
l1bdata = 'GEDI01_B_2023036011021_O23498_03_T11222_02_005_02_V002_subsetted.h5'

datadir = 'data/raw/SJER/2023-04/'

FWIDTH=23	# footprint width in meters
FWHM=14	# pulse FWHM in nanoseconds
SIGMA_P = FWHM/2.354

# class Block():
# 	def __init__(self, img_fn, pc_fn):#, pos_matrix, bounds, kdtree, photons):
# 		self.cam_file = img_fn
# 		self.pc_file = pc_fn
# 		self.pos_matrix, self.bounds = las_tools.get_pos_matrix(img_fn)
# 		self.kdtree, self.photons, self.gnd_kd, self.ground = las_tools.get_pc_data(pc_fn, generate_kdtree=True)
# 		# self.pulse_centers = self.get_pulse_centers(self.bounds, self.pos_matrix)
# 		# self.pulse_returns = get_photons_in_pulses(self)

def gauss_weight(d2, sigma):
	return np.exp(-1*d2/(2*np.square(sigma)))/(np.sqrt(2*np.pi)*sigma)

def get_blocks_w_data(pc_dir):
	fl = np.array([entry.split('_') for entry in os.listdir(f'{datadir}camera/')])
	return fl[:,3:5]

def get_result_mask(arr, block_coords):
	tmp = arr[:,2:4]/1000
	tmp = tmp.astype(int)*1000
	return np.isin(tmp, block_coords).all(axis=1)

def gen_rx_parameters(fk, block_coords):
	#samples = np.random.randint(len(fk['rx_sample_start_index']), size=size)



	parameters = np.zeros((len(fk['rx_sample_start_index']),5))
	parameters[:,0] = np.array(fk['rx_sample_start_index'])
	parameters[:,1] = np.array(fk['rx_sample_count'])
	parameters[:,4] = np.array(fk['geolocation']['degrade'])

	begin = utm.from_latlon(np.array(fk['geolocation']['latitude_bin0']),np.array(fk['geolocation']['longitude_bin0']))
	end = utm.from_latlon(np.array(fk['geolocation']['latitude_lastbin']),np.array(fk['geolocation']['longitude_lastbin']))

	lat_mid = (begin[0] + end[0])/2
	lon_mid = (begin[1] + end[1])/2

	parameters[:,2] = lat_mid.astype(int)
	parameters[:,3] = lon_mid.astype(int)

	mask = get_result_mask(parameters, block_coords)

	print(f'Before mask {len(parameters)}')

	parameters = parameters[mask]

	print(f'After mask {len(parameters)}')

	parameters = parameters[parameters[:,4] == 0]

	print(f'After degrade {len(parameters)}')

	parameters = parameters[parameters[:,1] > 701]

	print(f'After filter {len(parameters)}')

	return parameters


if __name__ == '__main__':

	block_coords = get_blocks_w_data(datadir)

	for entry in os.listdir(l1bdir):
		l1b_fn = f'{l1bdir}{entry}'

		with h5py.File(l1b_fn, 'r') as f:
			key = list(f.keys())[0]

			waveform = np.array(f[key]['rxwaveform'])

			flags = list(f[key]['geolocation']['degrade'])
			samples = list(f[key]['rx_sample_count'])
			print(f'{len(flags)} of {len(samples)} FLAGS: {flags}')

			rx_params = gen_rx_parameters(f[key], block_coords)
			print(f'{len(rx_params)} located within blocks')

			for coord in block_coords:
				img_fn = f'{datadir}camera/2023_SJER_6_{coord[0]}_{coord[1]}_image.tif'
				pc_fn = f'{datadir}lidar/NEON_D17_SJER_DP1_{coord[0]}_{coord[1]}_classified_point_cloud_colorized.laz'

				mask = get_result_mask(rx_params, [np.array(coord).astype(int)])

				print(f'Checking block E: {np.array(coord).astype(int)[0]} N: {np.array(coord).astype(int)[1]}')
				
				gedi_wf = rx_params[mask]
				print(f'--> Located {len(gedi_wf)} returns')

				if len(gedi_wf) == 0:
					continue

				block = gedi_block.Block(img_fn, pc_fn)

				for wf in gedi_wf:
					center = wf[2:]
					sel_ph = block.kdtree.query_ball_point([center[0], center[1]], r=0.5*FWIDTH)
					photons = block.photons[sel_ph]

					if len(photons) == 0:
						print('--> No photons detected')
						continue

					sim_wf = bh_sim.simulate_waveform(photons, center, noise=False)

					fig, ax = plt.subplots(4)
					ax[0].plot(sim_wf[0])

					strt = int(wf[0])
					end = int(wf[0]+wf[1])
					print(f'S {strt} E {end}')
					ax[1].plot(waveform[strt:end])

					autocorr = np.convolve(sim_wf[0][::-1],waveform[strt:end],mode='valid')
					ax[2].plot(autocorr)
					shift = np.argmax(autocorr)
					print(f'Shift: {shift}')
					sst = strt+shift
					sed = sst+len(sim_wf[0])
					ax[3].plot(waveform[sst:sed])

					corr = stats.pearsonr(sim_wf[0],waveform[sst:sed])
					print(f'Corr: {corr}')
					plt.show()
					plt.close()
