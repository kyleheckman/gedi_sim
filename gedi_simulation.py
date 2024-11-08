import sys
sys.path.append('src/')

import numpy as np
from scipy import stats
from scipy.stats import rv_continuous

import matplotlib.pyplot as plt
import os

import gedi_block
import simulation

TARGET = 'data/raw/SJER/2023-04/'
BLOCK = '251000_4106000'
OUTPUT = 'output/'

FWIDTH=23	# footprint width in meters
FWHM=15.6	# pulse FWHM in nanoseconds
STD_P = FWHM*0.15/2.35

def height_bins(height, step_size):
	return np.arange(int(np.min(height))-5, int(np.max(height))+5, step_size)

def shift_waveform(waveform, hist_scale):
	z_shift = 0.15*(len(waveform) + (len(waveform)%2))/2
	z_center = (np.min(hist_scale) + np.max(hist_scale) + 0.15)/2
	return np.arange(z_center-z_shift, z_center+z_shift, 0.15)

def pulse_weights(x, std):
	return np.exp(-1*np.square(x)/(2*np.square(std)))/std*np.sqrt(2*np.pi)

def get_pulse_waveform(hist):
	x = np.arange(-10,10,0.15)
	pulse = [pulse_weights(xi, std=STD_P) for xi in x]
	#pulse1 = stats.norm.pdf(x, scale=STD_P)
	#print(pulse1)
	#print(pulse)
	pulse = pulse/np.sum(pulse)
	nf_wave = np.convolve(hist[0], pulse)
	y = shift_waveform(nf_wave, hist[1])
	#wave = np.vstack((y[1:], nf_wave))
	#print(wave)
	if len(y) == len(nf_wave):
		plt.plot(y, nf_wave)
	else:
		plt.plot(y[1:],nf_wave)
	return

def compute_weight(photon, center, rel_weight='count', std=0.25*FWIDTH):
	if rel_weight == 'count':
		scale=1
	return scale*np.exp(-1*(np.square(center[0]-photon[0])+np.square(center[1]-photon[1]))/(2*np.square(std)))*(1/(std*np.sqrt(2*np.pi)))

def weighted_histogram(center, photons, axs):
	fn = f'block_{int(center[0])}_{int(center[1])}_both_hist.png'
	z_weights = np.zeros((len(photons),2))
	for ind in range(len(photons)):
		z_weights[ind,:] = np.array([photons[ind][2],compute_weight(photons[ind],center,rel_weight='count')])
	if len(z_weights) == 0:
		print(f'{fn}: EMPTY')
		return
	#valid_heights = simulation.get_height_bins(z_weights[:,0], voxel_height=0.15)
	#print(f'SIM: {valid_heights}')
	valid_heights = height_bins(z_weights[:,0], step_size=0.15)
	#print(f'TEST: {test}')
	#fig, ax = plt.subplots()
	axs[1].hist(z_weights[:,0], bins=valid_heights, weights=z_weights[:,1])
	#plt.savefig(f'{OUTPUT}{fn}')
	#return f'{fn}: {len(z_weights)}'
	return np.histogram(z_weights[:,0], bins=valid_heights, weights=z_weights[:,1])

def complete_histogram(center, photons, axs):
	#for pulse in block.pulse_returns:
	fn = f'block_{int(center[0])}_{int(center[1])}_hist.png'
	z = photons[:,2]
	if len(z) == 0:
		plt.close()
		return f'{fn}: Empty'
	#valid_heights = simulation.get_height_bins(z, voxel_height=0.15)
	valid_heights = height_bins(z, step_size=0.15)
	#fig, ax = plt.subplots()
	axs[0].hist(z, bins=valid_heights)
	#plt.savefig(f'{OUTPUT}{fn}')
	return f'{fn}: {len(z)}'

def compute_waveforms(block):
	if not os.path.exists(f'{OUTPUT}'):
		os.makedirs(f'{OUTPUT}')

	for pulse in block.pulse_returns:
		fig, axs = plt.subplots(2)
		print(complete_histogram(pulse[0],block.photons[pulse[1]], axs))
		hist = weighted_histogram(pulse[0],block.photons[pulse[1]], axs)
		if hist:
			#print(hist)
			get_pulse_waveform(hist)
			plt.savefig(f'{OUTPUT}block_{pulse[0][0]}_{pulse[0][1]}_combined_hist.png')
			plt.close()

if __name__ == '__main__':
	img_fn = f'{TARGET}camera/2023_SJER_6_{BLOCK}_image.tif'
	pc_fn = f'{TARGET}lidar/NEON_D17_SJER_DP1_{BLOCK}_classified_point_cloud_colorized.laz'

	block = gedi_block.Block(img_fn,pc_fn)

	compute_waveforms(block)
	#get_pulse_waveform()
	