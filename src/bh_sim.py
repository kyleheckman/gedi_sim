import sys
sys.path.append('src/')

import numpy as np
from scipy import stats
from scipy.stats import rv_continuous

import matplotlib.pyplot as plt
import os

import gedi_block
import las_tools

TARGET = 'data/raw/SJER/2023-04/'
BLOCK = '251000_4106000'
OUTPUT = 'output_bh/'


# SIMULATION SPECS
FWIDTH=23				# footprint width in meters
FWHM=14					# pulse FWHM in nanoseconds
SIGMA_P = FWHM/2.354	# standard deviation for transmitted pulse
SIGMA_N = 3				# noise standard devation

def gauss_weight(d2, sigma):
	return np.exp(-1*d2/(2*np.square(sigma)))/(np.sqrt(2*np.pi)*sigma)

def add_noise(arr, sigma, nbar):
	noise_arr = np.random.normal(loc=nbar, scale=sigma, size=len(arr))
	return arr + noise_arr

def get_intensity_bins(photons, center, ground=None):
	sample_min = np.min(photons[:,2])
	sample_max = np.max(photons[:,2])
	if ground:
		sample_min = ground[0]
		sample_max = ground[1]

	zmin = int(sample_min-5)
	zmax = int(sample_max+5)
	bins = np.arange(zmin, zmax, step=0.15)
	arr = np.zeros((2, len(bins)))
	arr[0] = bins

	#weighted_int = [[p[2], gauss_weight(np.square(p[0]-center[0])+np.square(p[1]-center[1]),sigma=0.25*FWIDTH)] for p in photons]
	weighted_int = [[p[2], p[3]*gauss_weight(np.square(p[0]-center[0])+np.square(p[1]-center[1]),sigma=0.25*FWIDTH)] for p in photons]
	for p in weighted_int:
		bucket = int(np.ceil((zmax-p[0])/0.15))
		arr[1][bucket] += p[1]

	return arr

def simulate_waveform(photons, center, noise=True, ground=None):
	if ground:
		intensity_bins = get_intensity_bins(photons, center, ground=ground)
	else:
		intensity_bins = get_intensity_bins(photons, center)

	pulse = [gauss_weight(np.square(n),sigma=SIGMA_P) for n in np.arange(-30,30)] #61ns pulse centered on 0
	nfw = np.convolve(pulse, intensity_bins[1])	# noise-free waveform
	res = [nfw]

	if noise:
		noise_wf = add_noise(nfw, SIGMA_N, 0)
		res.append(noise_wf)

	return res

def get_rh_metrics(signal, ground):
	total_energy = np.sum(signal)
	energy_int = np.array(np.cumsum(signal[::-1])/total_energy)
	energy_int = energy_int[::-1]
	#print([[i, energy_int[i]] for i in range(len(energy_int))])

	energy_bounds = [0.02, 0.25, 0.5, 0.75, 0.98]
	ind = [np.argwhere(energy_int>i).max() for i in energy_bounds]
	ind = (ground-ind)*0.15
	#print(ind)
	return ind


# ========== / For testing / ==========

def plot_waveforms(wf_arr, size, rh=None, saveFiles=None):
	if size == 3:
		fig, ax = plt.subplots(3)
		ax[0].plot(wf_arr[0])
		for ind in rh:
			ax[0].axvline(x=ind, color='r')
		ax[1].plot(wf_arr[2])
		ax[2].plot(wf_arr[1])
	else:
		fig, ax = plt.subplots(2)
		ax[0].plot(wf_arr[0])
		for ind in rh:
			ax[0].axvline(x=ind, color='r')
		ax[1].plot(wf_arr[1])

	if saveFiles != None:
		fn = f'{OUTPUT}waveform_{saveFiles[0]}_{saveFiles[1]}.png'
		plt.savefig(fn)
	else:
		plt.show()
	plt.close()

def simulate_block(block, noise=True, saveFiles=False):
	if not os.path.exists(f'{OUTPUT}'):
		os.makedirs(f'{OUTPUT}')

	for ind in range(len(block.pulse_returns)):
		pulse = block.pulse_returns[ind]
		g_pulse = block.ground_returns[ind]

		center = pulse[0]
		photons = block.photons[pulse[1]]
		ground = block.ground[g_pulse[1]]

		if len(photons)==0:
			continue

		res = simulate_waveform(photons, center, noise=noise)
		bin_size = [np.min(photons[:,2]), np.max(photons[:,2])]
		res.append(simulate_waveform(ground, center, noise=False, ground=bin_size)[0])

		ind = get_rh_metrics(res[0], np.argmax(res[-1]))

		svf = None
		if saveFiles:
			svf = center
		
		plot_waveforms(res, len(res), rh=ind, saveFiles=svf)

if __name__ == '__main__':
	img_fn = f'{TARGET}camera/2023_SJER_6_{BLOCK}_image.tif'
	pc_fn = f'{TARGET}lidar/NEON_D17_SJER_DP1_{BLOCK}_classified_point_cloud_colorized.laz'

	block = gedi_block.Block(img_fn,pc_fn)
	#block.photons[:,3] = block.photons[:,3]/100		#scale intensity values

	simulate_block(block)