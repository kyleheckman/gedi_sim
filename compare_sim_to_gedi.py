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
import simulation
import read_colorized_pc


l1bdir = 'gedi_data/GEDI01_B_002-20241107_214221/'
l1bdata = 'GEDI01_B_2023036011021_O23498_03_T11222_02_005_02_V002_subsetted.h5'

datadir = 'src/data/raw/SJER/2023-04/'

FWIDTH=23	# footprint width in meters
FWHM=14	# pulse FWHM in nanoseconds
STD_P = FWHM/2.354

class Block():
	def __init__(self, img_fn, pc_fn):#, pos_matrix, bounds, kdtree, photons):
		self.cam_file = img_fn
		self.pc_file = pc_fn
		self.pos_matrix, self.bounds = simulation.get_pos_matrix(img_fn)
		self.kdtree, self.photons = read_colorized_pc.get_pc_data(pc_fn, generate_kdtree=True)
		# self.pulse_centers = self.get_pulse_centers(self.bounds, self.pos_matrix)
		# self.pulse_returns = get_photons_in_pulses(self)

def gauss_weight(d2, sigma):
	return np.exp(-1*d2/(2*np.square(sigma)))/(np.sqrt(2*np.pi)*sigma)

if __name__ == '__main__':
	#l1b_fn = f'{l1bdir}{l1bdata}'

	block_coords = []
	for entry in os.listdir(f'{datadir}camera/'):
		tmp = entry.split('_')
		block_coords.append([tmp[3], tmp[4]])
	#print(block_coords)

	for entry in os.listdir(l1bdir):
		l1b_fn = f'{l1bdir}{entry}'

		with h5py.File(l1b_fn, 'r') as f:
			key = list(f.keys())[0]

			waveform = np.array(f[key]['rxwaveform'])

			#print(len(f[key]['rx_sample_start_index']))

			size = len(f[key]['rx_sample_start_index'])
			#samples = np.random.randint(len(f[key]['rx_sample_start_index']), size=size)

			data_params = np.zeros((size,4))
			data_params[:,0] = np.array(f[key]['rx_sample_start_index'])
			data_params[:,1] = np.array(f[key]['rx_sample_count'])

			begin = utm.from_latlon(np.array(f[key]['geolocation']['latitude_bin0']),np.array(f[key]['geolocation']['longitude_bin0']))
			end = utm.from_latlon(np.array(f[key]['geolocation']['latitude_lastbin']),np.array(f[key]['geolocation']['longitude_lastbin']))

			lat_mid = (begin[0] + end[0])/2
			lon_mid = (begin[1] + end[1])/2

			data_params[:,2] = lat_mid.astype(int)
			data_params[:,3] = lon_mid.astype(int)

			arr = data_params[:,2:]/1000
			arr = arr.astype(int)*1000
			mask = np.isin(arr, block_coords).all(axis=1)

			data_params = data_params[mask]
			data_params = data_params[data_params[:,1] > 701]
			print(f'{len(data_params)} located within blocks')
			#print(data_params.astype(int))
			#print(f'D {data_params}')

			#print(arr)
			#print(arr[arr[:,0]==260000])

			for coord in block_coords:
				img_fn = f'{datadir}camera/2023_SJER_6_{coord[0]}_{coord[1]}_image.tif'
				pc_fn = f'{datadir}lidar/NEON_D17_SJER_DP1_{coord[0]}_{coord[1]}_classified_point_cloud_colorized.laz'

				#print(img_fn)

				# if not os.path.exists(img_fn):
				# 	continue
				#print('FOUND')
				arr = data_params[:,2:]/1000
				arr = arr.astype(int)*1000
				mask = np.isin(arr, [np.array(coord).astype(int)]).all(axis=1)

				print(f'Checking block {[np.array(coord).astype(int)]}')
				
				gedi_wf = data_params[mask]
				print(f'Located {len(gedi_wf)} returns')

				if len(gedi_wf) == 0:
					continue

				block = Block(img_fn, pc_fn)

				
				for wf in gedi_wf:
					center = wf[2:]
					sel_ph = block.kdtree.query_ball_point([center[0], center[1]], r=0.5*FWIDTH)
					photons = block.photons[sel_ph]
					#print(f'P: {len(photons)}')
					if len(photons) == 0:
						print('NO PHOTONS FOUND')
						continue

					zmin = int(np.min(photons[:,2])-5)
					zmax = int(np.max(photons[:,2])+5)
					bins = np.arange(zmin, zmax,step=0.15)
					tmp = np.zeros((2, len(bins)))
					tmp[0] = bins

					wint = [[p[2], p[3]*gauss_weight(np.square(p[0]-center[0])+np.square(p[1]-center[1]),sigma=0.25*FWIDTH)] for p in photons]
					for p in wint:
						bucket = int(np.ceil((zmax-p[0])/0.15))
						tmp[1][bucket] += p[1]

					pulse = [gauss_weight(np.square(n),sigma=STD_P) for n in np.arange(-30,30)] #61ns pulse centered on 0
					nfw = np.convolve(pulse,tmp[1])

					fig, ax = plt.subplots(4)
					ax[0].plot(nfw)

					strt = int(wf[0])
					end = int(wf[0]+wf[1])
					print(f'S {strt} E {end}')
					ax[1].plot(waveform[strt:end])

					autocorr = np.convolve(nfw[::-1],waveform[strt:end],mode='valid')
					ax[2].plot(autocorr)
					shift = np.argmax(autocorr)
					print(f'Shift: {shift}')
					sst = strt+shift
					sed = sst+len(nfw)
					ax[3].plot(waveform[sst:sed])

					corr = stats.pearsonr(nfw,waveform[sst:sed])
					print(f'Corr: {corr}')
					plt.show()
					plt.close()
