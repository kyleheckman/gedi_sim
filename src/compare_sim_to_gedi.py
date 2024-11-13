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


l1bdir = 'data/gedi/GEDI01_B_002-20241107_214221/'
l1bdata = 'GEDI01_B_2023036011021_O23498_03_T11222_02_005_02_V002_subsetted.h5'
l2a_dir = 'data/gedi/GEDI02_A_002-20241112_190836/'


datadir = 'data/raw/SJER/2023-04/'

FWIDTH=23	# footprint width in meters
FWHM=14	# pulse FWHM in nanoseconds
SIGMA_P = FWHM/2.354

OUTPUT = 'output_sim/'
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
	tmp = np.column_stack((arr['lat'], arr['long']))/1000
	tmp = tmp.astype(int)*1000
	return np.isin(tmp, block_coords).all(axis=1)

def gen_rx_parameters(fk, block_coords, rh=None):
	#samples = np.random.randint(len(fk['rx_sample_start_index']), size=size)

	parameters = {}

	#parameters = np.zeros((len(fk['rx_sample_start_index']),5+adj))
	parameters['start_indx'] = np.array(fk['rx_sample_start_index'])
	parameters['count'] = np.array(fk['rx_sample_count'])
	parameters['degrade'] = np.array(fk['geolocation']['degrade'])

	if rh.any():
		parameters['rh'] = rh

	begin = utm.from_latlon(np.array(fk['geolocation']['latitude_bin0']),np.array(fk['geolocation']['longitude_bin0']))
	end = utm.from_latlon(np.array(fk['geolocation']['latitude_lastbin']),np.array(fk['geolocation']['longitude_lastbin']))

	lat_mid = (begin[0] + end[0])/2
	lon_mid = (begin[1] + end[1])/2

	parameters['lat'] = lat_mid.astype(int)
	parameters['long'] = lon_mid.astype(int)

	mask = get_result_mask(parameters, block_coords)

	print(f'Before mask {len(parameters['start_indx'])}')

	parameters = {key: value[mask] for key, value in parameters.items()}

	print(f'After mask {len(parameters['start_indx'])}')

	mask = [x == 0 for x in parameters['degrade']]
	parameters = {key: value[mask] for key, value in parameters.items()}

	print(f'After degrade {len(parameters['start_indx'])}')

	mask = [x > 701 for x in parameters['count']]
	parameters = {key: value[mask] for key, value in parameters.items()}

	print(f'After filter {len(parameters['start_indx'])}')

	return parameters

def read_h5(fn):
	l1b_fn = f'{l1bdir}{fn}'
	fl = fn.split('_')
	l2a_fn = f'{l2a_dir}GEDI02_A_{fl[2]}_{fl[3]}_{fl[4]}_{fl[5]}_02_003_02_V002_subsetted.h5'


	rn = None
	with h5py.File(l2a_fn, 'r') as f:
		key = list(f.keys())[0]

		rh = np.array(f[key]['rh'])

	with h5py.File(l1b_fn, 'r') as f:
		key = list(f.keys())[0]

		waveform = np.array(f[key]['rxwaveform'])
		rx_params = gen_rx_parameters(f[key], block_coords, rh=rh)

		#flags = list(f[key]['geolocation']['degrade'])
		#samples = list(f[key]['rx_sample_count'])
		#print(f'{len(flags)} of {len(samples)} FLAGS: {flags}')
	
	return waveform, rx_params, fl[3]

if __name__ == '__main__':

	block_coords = get_blocks_w_data(datadir)

	files = os.listdir(l1bdir)
	fl = files[0].split('_')
	l2afn = f'{l2a_dir}GEDI02_A_{fl[2]}_{fl[3]}_{fl[4]}_{fl[5]}_02_003_02_V002_subsetted.h5'

	# with h5py.File(l2afn, 'r') as f:
	# 	key = list(f.keys())[0]
	# 	print(len(f[key]['rh'][-2]))
	# 	print(f[key]['rh'][-2])

	sim_metrics = {}
	sim_metrics['corr'] = []
	sim_metrics['rh25'] = []
	sim_metrics['rh50'] = []
	sim_metrics['rh75'] = []

	for entry in os.listdir(l1bdir):
		# l1b_fn = f'{l1bdir}{entry}'

		# with h5py.File(l1b_fn, 'r') as f:
		# 	key = list(f.keys())[0]
		# 	waveform, rx_params = read_h5(f[key])

		# 	# waveform = np.array(f[key]['rxwaveform'])

		# 	# rx_params = gen_rx_parameters(f[key], block_coords)
		# 	# print(f'{len(rx_params)} located within blocks')

		waveform, rx_params, orbit = read_h5(entry)

# tabbed in

		for coord in block_coords:
			img_fn = f'{datadir}camera/2023_SJER_6_{coord[0]}_{coord[1]}_image.tif'
			pc_fn = f'{datadir}lidar/NEON_D17_SJER_DP1_{coord[0]}_{coord[1]}_classified_point_cloud_colorized.laz'
			tmp_params = rx_params

			mask = get_result_mask(tmp_params, [np.array(coord).astype(int)])

			print(f'Checking block E: {np.array(coord).astype(int)[0]} N: {np.array(coord).astype(int)[1]}')
			
			gedi_wf = tmp_params = {key: value[mask] for key, value in tmp_params.items()}
			print(f'--> Located {len(gedi_wf['start_indx'])} returns')

			if len(gedi_wf['start_indx']) == 0:
				continue

			block = gedi_block.Block(img_fn, pc_fn)

			for indx in range(len(gedi_wf['start_indx'])):
				center = [gedi_wf['lat'][indx], gedi_wf['long'][indx]]
				sel_ph = block.kdtree.query_ball_point([center[0], center[1]], r=0.5*FWIDTH)
				photons = block.photons[sel_ph]
				ground = block.ground[block.gnd_kd.query_ball_point([center[0], center[1]], r=0.5*FWIDTH)]

				if len(photons) == 0:
					print('--> No photons detected')
					continue

				sim_wf = bh_sim.simulate_waveform(photons, center, noise=False)
				bin_size = [np.min(photons[:,2]), np.max(photons[:,2])]
				sim_wf.append(bh_sim.simulate_waveform(ground, center, noise=False, ground=bin_size)[0])

				

				strt = int(gedi_wf['start_indx'][indx])
				end = int(gedi_wf['start_indx'][indx]+gedi_wf['count'][indx])
				print(f'S {strt} E {end}')

				autocorr = np.convolve(sim_wf[0][::-1],waveform[strt:end],mode='valid')
				shift = np.argmax(autocorr)
				print(f'Shift: {shift}')
				sst = strt+shift
				sed = sst+len(sim_wf[0])
				corr = stats.pearsonr(sim_wf[0],waveform[sst:sed])
				print(f'Corr: {corr.statistic}')
				
				rel_heights = bh_sim.get_rh_metrics(sim_wf[0], np.argmax(sim_wf[-1]))
				print(rel_heights)
				print(gedi_wf['rh'])
				diff = [rel_heights[1] - gedi_wf['rh'][indx][25], rel_heights[2] - gedi_wf['rh'][indx][50], rel_heights[3] - gedi_wf['rh'][indx][75]]
				print(diff)

				sim_metrics['corr'].append(corr.statistic)
				sim_metrics['rh25'].append(diff[0])
				sim_metrics['rh50'].append(diff[1])
				sim_metrics['rh75'].append(diff[2])
				
				fig, ax = plt.subplots(4)
				ax[0].plot(sim_wf[0])
				ax[1].plot(waveform[strt:end])
				ax[2].plot(autocorr)
				ax[3].plot(waveform[sst:sed])

				if not os.path.exists(OUTPUT):
					os.makedirs(OUTPUT)

				filename = f'{OUTPUT}_{center[0]}_{center[1]}_{orbit}.png'
				plt.savefig(filename)
				plt.close()

	print(f'Mean Corr: {np.average(sim_metrics['corr'])} | Min: {np.min(sim_metrics['corr'])} | Max: {np.max(sim_metrics['corr'])}')
	print(f'Mean Abs Bias RH25: {np.average(np.abs(sim_metrics['rh25']))} | StDev: {np.std(sim_metrics['rh25'])}')
	print(f'Mean Abs Bias RH50: {np.average(np.abs(sim_metrics['rh50']))} | StDev: {np.std(sim_metrics['rh50'])}')
	print(f'Mean Abs Bias rh75: {np.average(np.abs(sim_metrics['rh75']))} | StDev: {np.std(sim_metrics['rh75'])}')