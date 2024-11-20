import sys
sys.path.append('src/')

import numpy as np
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf

import pc_block

def gauss_weight(d2, sigma):
	return np.exp(-1*d2/(2*np.square(sigma)))/(np.sqrt(2*np.pi)*sigma)

def get_bins(photons):
		zmin = int(np.min(photons[:,2])-5)
		zmax = int(np.max(photons[:,2])+5)
		return (zmin, zmax)

def get_centers(block, config):
	centers = []

	for indx in np.arange(int(0.5*config['sim_config']['gedi_config']['fwidth']), np.shape(block.pos_matrix)[0], config['sim_config']['gedi_config']['across_spacing']):
		for indy in np.arange(int(0.5*config['sim_config']['gedi_config']['fwidth']), np.shape(block.pos_matrix)[1], config['sim_config']['gedi_config']['along_spacing']):
			centers.append([block.pos_matrix[indx, indy, 0], block.pos_matrix[indx, indy, 1]])

	return centers

class Waveform():
	def __init__(self, photons, collected, gnd_phot, ground, config):
		self.nfw, self.gnd = self.get_waveforms(photons, collected, gnd_phot, ground, config)
		self.rh_idx, self.rh = self.get_rh_metrics(config)

	def get_rh_metrics(self, config):
		if self.nfw is None:
			return None

		energy_int = np.array(np.cumsum(self.nfw[::-1])/np.sum(self.nfw))
		energy_int = energy_int[::-1]


		energy_marks = [0.02, 0.25, 0.5, 0.75, 0.98]

		indx = [np.argwhere(energy_int>=i).max() for i in energy_marks]
		height_adjusted = (np.argmax(self.gnd)-indx)*config['sim_config']['gedi_config']['resolution']

		return indx, height_adjusted

	def get_waveforms(self, photons, collected, gnd_phot, ground, config):
		sel_photons = photons[collected.photons]
		sel_ground = gnd_phot[ground.photons]

		if len(sel_photons) == 0:
			return None, None

		bounds = get_bins(sel_photons)
		nfw = self.simulate(sel_photons, collected.center, bounds, config)
		gnd = self.simulate(sel_ground, ground.center, bounds, config)
		return nfw, gnd

	def simulate(self, photons, center, bounds, config):
		arr = np.zeros((int((bounds[1]-bounds[0])/config['sim_config']['gedi_config']['resolution'])))

		weighted_intensity = [[p[2], p[3]*gauss_weight(np.square(p[0]-center[0])+np.square(p[1]-center[1]),sigma=0.25*config['sim_config']['gedi_config']['fwidth'])] for p in photons]
		for pair in weighted_intensity:
			bucket = int(np.ceil((bounds[1]-pair[0])/config['sim_config']['gedi_config']['resolution']))
			arr[bucket] += pair[1]

		pulse = pulse = [gauss_weight(np.square(n),sigma=config['sim_config']['gedi_config']['fwhm']/2.354) for n in np.arange(-25,25)]
		nfw = np.convolve(pulse, arr)

		return nfw

# ==========/ Testing Only /==========

def plot_waveforms(waveform):
	fig, ax = plt.subplots()
	ax.plot(waveform.nfw)
	#ax.plot(waveform.gnd)

	for indx in waveform.rh_idx:
		ax.axvline(x=indx, color='r')
	plt.show()
	plt.close()

if __name__ == '__main__':
	target = 'data/raw/SJER/2023-04/'
	bl = '251000_4106000'
	output = 'output_ref/'

	if not os.path.exists(output):
		os.makedirs(output)

	img_fn = f'{target}camera/2023_SJER_6_{bl}_image.tif'
	pc_fn = f'{target}lidar/NEON_D17_SJER_DP1_{bl}_classified_point_cloud_colorized.laz'

	diz_path = os.path.dirname(os.path.realpath(__file__))
	config = OmegaConf.load(f'{diz_path}/config/sim_config.yaml')
	print(config)

	block = pc_block.Block(img_fn, pc_fn)
	centers = get_centers(block, config)
	pulses = pc_block.Pulses(block, centers, config)

	for indx in range(len(pulses.pulse_centers)):
		if len(pulses.ret_photons[indx].photons) == 0:
			continue

		sim_wf = Waveform(block.photons, pulses.ret_photons[indx], block.ground, pulses.ret_ground[indx], config)
		if sim_wf.nfw is None:
			continue
		print(sim_wf.rh)
		plot_waveforms(sim_wf)
