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
OUTPUT = 'output_bh/'

FWIDTH=23	# footprint width in meters
FWHM=15.6	# pulse FWHM in nanoseconds
STD_P = FWHM*0.15/2.354

SIGMA_N = 3

def gauss_weight(d2, sigma):
	return np.exp(-1*d2/(2*np.square(sigma)))/(np.sqrt(2*np.pi)*sigma)

def add_noise(arr, sigma, nbar):
	noise_arr = np.random.normal(loc=nbar, scale=sigma, size=len(arr))
	return arr + noise_arr

def simulate_waveforms(block):
	if not os.path.exists(f'{OUTPUT}'):
		os.makedirs(f'{OUTPUT}')

	for pulse in block.pulse_returns:
		center = pulse[0]
		photons = block.photons[pulse[1]]

		if len(photons)==0:
			continue

		zmin = int(np.min(photons[:,2])-5)
		zmax = int(np.max(photons[:,2])+5)
		bins = np.arange(zmin, zmax,step=0.15)
		arr = np.zeros((2, len(bins)))
		arr[0] = bins

		wint = [[p[2], p[3]*gauss_weight(np.square(p[0]-center[0])+np.square(p[1]-center[1]),sigma=0.25*FWIDTH)] for p in photons]
		for p in wint:
			bucket = int(np.ceil((zmax-p[0])/0.15))
			arr[1][bucket] += p[1]

		pulse = [gauss_weight(np.square(n),sigma=STD_P) for n in np.arange(-30,30)] #61ns pulse centered on 0
		nfw = np.convolve(pulse,arr[1])
		res = add_noise(nfw, SIGMA_N, 0)

		fn = f'{OUTPUT}nf_waveform_{center[0]}_{center[1]}.png'

		fig, ax = plt.subplots(2)
		ax[0].plot(nfw)
		ax[1].plot(res)
		plt.savefig(fn)
		plt.close()


if __name__ == '__main__':
	img_fn = f'{TARGET}camera/2023_SJER_6_{BLOCK}_image.tif'
	pc_fn = f'{TARGET}lidar/NEON_D17_SJER_DP1_{BLOCK}_classified_point_cloud_colorized.laz'

	block = gedi_block.Block(img_fn,pc_fn)
	block.photons[:,3] = block.photons[:,3]/100		#scale intensity values

	simulate_waveforms(block)