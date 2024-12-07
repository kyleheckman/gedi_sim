import sys
sys.path.append('src/')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf
import concurrent.futures
from tqdm import tqdm
import itertools

import simulation
import pc_block

def get_hhdc_centers(block, config):
	square_size = config['hhdc_config']['square_size']
	x_dim = np.arange(block.bounds.left+(square_size/2), block.bounds.right, step=square_size)
	y_dim = np.arange(block.bounds.bottom+(square_size/2), block.bounds.top, step=square_size)

	x_coords, y_coords = np.meshgrid(x_dim, y_dim)

	centers = np.dstack((x_coords, y_coords))
	
	return centers

def prep_waveform(waveform, z_min, block_z_min, config):
	col_wf = waveform.nfw / np.sum(waveform.nfw)
	col_wf = col_wf[np.min(np.argwhere(col_wf != 0)):np.max(np.argwhere(col_wf != 0))+1]

	shift = int((z_min - block_z_min) / config['sim_config']['gedi_config']['resolution'])
	shift = 0
	
	column = np.zeros((config['hhdc_config']['height_bins']), dtype=np.float32)

	try:
		column[shift:shift+len(col_wf)] = col_wf[::-1]
	except Exception as e:
		print(f'Error | {e}')


	return column
	
def disp_dem(hhdc_slice, bounds, fn, indx, z_min, config):
	bins = config['hhdc_config']['height_bins']
	resolution = config['sim_config']['gedi_config']['resolution']

	fig, ax = plt.subplots()

	ax.imshow(hhdc_slice.T[::-1], aspect=5, interpolation='none', cmap=matplotlib.colormaps['Greens'], extent=[bounds.left,bounds.right,0,bins*resolution])
	ax.set_title(f'Digital Elevation Model for Across Track Slice {indx}')
	ax.set_ylabel(f'Height from Reference Ground: {np.round(z_min,1)} [m]')
	ax.set_xlabel(f'Along Track Coordinate [m]')

	plt.savefig(fn)
	plt.close()

def get_fn_pairs(data_dir):
	cam_files_dir = os.path.join(data_dir, 'camera/')
	laz_files_dir = os.path.join(data_dir, 'lidar/')

	cam_files = [f for f in os.listdir(cam_files_dir) if os.path.isfile(os.path.join(cam_files_dir, f))]
	laz_files = [f for f in os.listdir(laz_files_dir) if os.path.isfile(os.path.join(laz_files_dir, f))]
	laz_fn_structure = '_'.join(laz_files[0].split('_')[:4])

	full_kms_fn = []
	for camera_file in cam_files:
		coords = camera_file.split('_')[3:5]
		camera_file = f'{cam_files_dir}{camera_file}'
		pc_file = f'{laz_files_dir}{laz_fn_structure}_{coords[0]}_{coords[1]}_classified_point_cloud_colorized.laz'

		# check if the file exists
		if os.path.exists(pc_file):
			full_kms_fn.append((camera_file, pc_file))

	return full_kms_fn

def run_sim(fn_pair, output, config):
	img_fn = fn_pair[0]
	pc_fn = fn_pair[1]

	bl = '_'.join(img_fn.split('_')[-3:-1])

	block = pc_block.Block(img_fn, pc_fn)
	centers = get_hhdc_centers(block, config)

	hhdc = np.zeros((np.shape(centers)[0], np.shape(centers)[1], config['hhdc_config']['height_bins']), dtype=np.float32)
	centers = np.concatenate(centers, axis=0)

	pulses = pc_block.Pulses(block, centers, config)

	block_z_min = np.min(block.photons[:,2])

	for indx in range(np.shape(centers)[0]):
		if len(pulses.ret_photons[indx].photons) == 0:
			continue

		sim_wf = simulation.Waveform(block.photons, pulses.ret_photons[indx], block.ground, pulses.ret_ground[indx], config)
		z_min = np.min(block.photons[pulses.ret_photons[indx].photons][:,2])

		column = prep_waveform(sim_wf, z_min, block_z_min, config)
		
		ix = indx % np.shape(hhdc)[0]
		iy = int(indx / np.shape(hhdc)[1])

		hhdc[iy, ix, :] = column

	fn = f'{output}SJER_{bl}_cc_hhdc_20x20_gridded.npy'
	np.save(fn, hhdc)
	return fn

if __name__ == '__main__':
	target = 'data/raw/SJER/2023-04/'
	bl = '252000_4109000'
	output = 'output_hhdc_cc/'

	# True if concurrency enabled
	cc = False

	fn_pairs = get_fn_pairs(target)
	print(fn_pairs)

	diz_path = os.path.dirname(os.path.realpath(__file__))
	config = OmegaConf.load(f'{diz_path}/config/sim_config.yaml')

	if cc:
		max_workers = 5
		fn_pairs_iter = iter(fn_pairs)
		with tqdm(total=len(fn_pairs)) as pbar:

			with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
				futures = {executor.submit(run_sim, km_fn, output, config) for km_fn in itertools.islice(fn_pairs_iter, max_workers)}
				print(len(futures))

				while len(futures) > 0:
					done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

					for fut in done:
						print(f'\n{len(futures)}\nTask {fut} completed w/ result: {fut.result()}')
						
						pbar.update(1)


					for km_fn in itertools.islice(fn_pairs_iter, len(done)):
						futures.add(executor.submit(run_sim, km_fn, output, config))
	else:
		for km_fn in fn_pairs:
			res = run_sim(km_fn, output, config)
			print(f'Completed {res}')