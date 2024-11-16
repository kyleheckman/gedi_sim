import h5py
import numpy as np
import utm
import os

def get_result_mask(arr, block_coords):
	tmp = np.column_stack((arr['lat'], arr['long']))/1000
	tmp = tmp.astype(int)*1000
	return np.isin(tmp, block_coords).all(axis=1)

def get_blocks_w_data(pc_dir):
	fl = np.array([entry.split('_') for entry in os.listdir(f'{pc_dir}camera/')])
	return fl[:,3:5]

class LVIS_Data():
	def __init__(self, fn, l1bdir, coords):
		self.fn = f'{l1bdir}{fn}'
		self.waveform, self.params = self.read_data()
	
	def read_data(self):
		lvis_data = lvis_reader.read_legacy_lvis(self.fn, '1.03')
		rx_params = self.get_rx_parameters(lvis_data)
		lvis_data = np.array(lvis_data)

		lvis_data[:,-1] = lvis_data[:,-1] - lvis_data[:,-3]

		waveform = np.concatenate(lvis_data[:,-1])

		return waveform, rx_params

	def get_rx_parameters(self, lvis_data):
		parameters = {}

		lat = np.array([x[7] for x in lvis_data])
		lon = np.array([x[6] for x in lvis_data])
		lon = (lon+180)%360 - 180

		begin = utm.from_latlon(lat, lon)

		lat = np.array([x[10] for x in lvis_data])
		lon = np.array([x[9] for x in lvis_data])
		lon = (lon+180)%360 - 180

		end = utm.from_latlon(lat, lon)

		lat_imd = (begin[0] + end[0])/2
		lon_mid = (being[1] + end[1])/2

		parameters['lat'] = lat_mid.astype(int)
		parameters['long'] = lon_mid.astype(int)

		parameters['start_indx'] = np.array([431*indx for indx in range(len(lvis_data ))])
		parameters['count'] = np.full((len(lvis_data)), 431)

		mask = get_result_mask(parameters, coords)
		parameters = {key: value[mask] for key, value in parameters.items()}

		return parameters

class GEDI_Data():
	def __init__(self, fn, l1bdir, l2adir, coords):
		self.f_details = fn.split('_')[3:6]
		self.l1b_dir = l1bdir
		self.l2a_dir = l2adir
		self.waveform, self.params = self.read_data(fn, coords)
	
	def read_data(self, fn, coords):
		l1b_fn = f'{self.l1b_dir}{fn}'
		tmp = fn.split('_')
		l2a_fn = f'{self.l2a_dir}GEDI02_A_{tmp[2]}_{tmp[3]}_{tmp[4]}_{tmp[5]}_02_003_02_V002_subsetted.h5'

		with h5py.File(l2a_fn, 'r') as f:
			key = list(f.keys())[0]
			rh = np.array(f[key]['rh'])

		with h5py.File(l1b_fn) as f:
			key = list(f.keys())[0]
			waveform = np.array(f[key]['rxwaveform'])
			rx_params = self.get_rx_parameters(f[key], coords, rh=rh)

		return waveform, rx_params

	def get_rx_parameters(self, fk, coords, rh=None):
		parameters = {}
		parameters['start_indx'] = np.array(fk['rx_sample_start_index'])
		parameters['count'] = np.array(fk['rx_sample_count'])
		parameters['degrade'] = np.array(fk['geolocation']['degrade'])

		if rh is not None:
			parameters['rh'] = rh

		begin = utm.from_latlon(np.array(fk['geolocation']['latitude_bin0']),np.array(fk['geolocation']['longitude_bin0']))
		end = utm.from_latlon(np.array(fk['geolocation']['latitude_lastbin']),np.array(fk['geolocation']['longitude_lastbin']))

		lat_mid = (begin[0] + end[0])/2
		lon_mid = (begin[1] + end[1])/2

		parameters['lat'] = lat_mid.astype(int)
		parameters['long'] = lon_mid.astype(int)

		mask = get_result_mask(parameters, coords)
		parameters = {key: value[mask] for key, value in parameters.items()}

		mask = [x == 0 for x in parameters['degrade']]
		parameters = {key: value[mask] for key, value in parameters.items()}

		mask = [x > 701 for x in parameters['count']]
		parameters = {key: value[mask] for key, value in parameters.items()}

		return parameters