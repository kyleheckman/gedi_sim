import sys
sys.path.append('src/')
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import utm
import h5py
import scipy.stats as stats

import gedi_block
import bh_sim
import las_tools
import lvis_reader


lvis_dir = 'data/lvis/LVIS_US_CA_day3_2008_WAVE_20081120.114/LVIS_US_CA_day3_2008_WAVE_20081120/'
wav = 'LVIS_US_CA_day3_2008_WAVE_20081120.lgw.1.03'

neon_dir = 'data/raw/TEAK/2021-07/'


FWIDTH = 23 #LVIS footprint width
FWHM = 7
SIGMA_P = FWHM/2.354

OUTPUT = 'output_lvis/'

def get_blocks_w_data(pc_dir):
	fl = np.array([entry.split('_') for entry in os.listdir(f'{pc_dir}camera/')])
	return fl[:,3:5]

def get_result_mask(arr, coords):
	tmp = arr/1000
	tmp = tmp.astype(int)*1000
	return np.isin(tmp, coords).all(axis=1)

def get_rx_parameters(data, coords):
	data = data[::100]
	lat = np.array([x[7] for x in data])
	lon = np.array([x[6] for x in data])
	lon = (lon+180)%360 - 180
	rx = np.array([x[-1] for x in data])
	noise_mean = np.array([x[-3] for x in data])

	converted = utm.from_latlon(lat, lon)
	parameters = np.column_stack([converted[0], converted[1]]).astype(int)

	mask = get_result_mask(parameters, coords)
	parameters = parameters[mask]
	print(len(parameters))
	rx = rx[mask]
	noise_mean = noise_mean[mask]
	
	rx = np.array([rx[ind]-noise_mean[ind] for ind in range(len(rx))])

	return parameters, rx


if __name__ == '__main__':
	block_coords = np.array(get_blocks_w_data(neon_dir)).astype(int)

	lvis_data = lvis_reader.read_legacy_lvis(f'{lvis_dir}{wav}','1.03')
	rx_coords, rx = get_rx_parameters(lvis_data, block_coords)

	for coord in block_coords:
			img_fn = f'{neon_dir}camera/2021_TEAK_5_{coord[0]}_{coord[1]}_image.tif'
			pc_fn = f'{neon_dir}lidar/NEON_D17_TEAK_DP1_{coord[0]}_{coord[1]}_classified_point_cloud_colorized.laz'

			tmp_coords = rx_coords

			mask = get_result_mask(tmp_coords, [np.array(coord).astype(int)])

			print(f'Checking block E: {np.array(coord).astype(int)[0]} N: {np.array(coord).astype(int)[1]}')
			
			lvis_wf = rx[mask]
			tmp_coords = tmp_coords[mask]
			print(f'--> Located {len(lvis_wf)} returns')

			if len(lvis_wf) == 0:
				continue

			block = gedi_block.Block(img_fn, pc_fn)

			for indx in range(len(tmp_coords)):
				center = tmp_coords[indx]
				sel_ph = block.kdtree.query_ball_point(center, r=0.5*FWIDTH)
				photons = block.photons[sel_ph]
				ground = block.ground[block.gnd_kd.query_ball_point(center, r=0.5*FWIDTH)]

				print(f'Center: E {center[0]} N {center[1]}')
				if len(photons) == 0:
					print('--> No photons detected')
					continue

				sim_wf = bh_sim.simulate_waveform(photons, center, noise=False)[0]
				sim_wf = sim_wf / np.max(sim_wf)
				#bin_size = [np.min(photons[:,2]), np.max(photons[:,2])]
				#sim_wf.append(bh_sim.simulate_waveform(ground, center, noise=False, ground=bin_size)[0])

				strt = 0
				end = len(rx[indx])
				waveform = rx[indx]/np.max(rx[indx])
				#print(len(waveform))
				#print(len(sim_wf))

				autocorr = np.convolve(sim_wf[::-1],waveform[strt:end],mode='valid')
				shift = np.argmax(autocorr)

				sst = strt+shift
				sed = sst+len(sim_wf)

				#print(shift)
				#print(f'{sst} {sed}')

				corr = stats.pearsonr(sim_wf,waveform[sst:sed])
				print(f'Corr: {corr.statistic}')

				fig, ax = plt.subplots(1, constrained_layout=True)
				fig.suptitle(f'LVIS Recorded Waveform vs. Simulated Noise-Free Waveform')

				#ax[0].set_title('Noise-Free Simluated Waveform [Normalized]')
				#ax[0].set_ylabel('Intensity')
				ax.plot(sim_wf[::-1], np.arange(len(sim_wf),0,-1), label="Simulated Waveform")

				#ax[1].set_title('Cropped LVIS Waveform [Normalized]')
				ax.set_ylabel('Time [ns]')
				ax.set_xlabel('DN [Normalized]')
				pl = waveform[sst:sed]
				ax.plot(pl[::-1], np.arange(len(pl),0,-1), label="LVIS Waveform")
				ax.invert_yaxis()
				handles, labels = ax.get_legend_handles_labels()
				ax.legend(handles[::-1], labels[::-1])
				plt.show()