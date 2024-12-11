import sys
sys.path.append('src/')

import numpy as np
import os
from omegaconf import OmegaConf
import itertools
import concurrent.futures

import simulation
import pc_block

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

def get_footprint_centers(block, config):
	square_size = config['hhdc_config']['square_size']
	x_dim = np.arange(block.bounds.left+(square_size/2), block.bounds.right, step=square_size)
	y_dim = np.arange(block.bounds.bottom+(square_size/2), block.bounds.top, step=square_size)

	x_coords, y_coords = np.meshgrid(x_dim, y_dim)

	centers = np.dstack((x_coords, y_coords))
	
	return centers

def calc_agbd(wf, type):
	try:
		rh = wf.rh
	except Exception as e:
		return 0

	if type == 'ent_na':
		agbd = 1.013 * np.square(-114.355 + (8.401 *  np.sqrt(rh[0.7] + 100)) + (3.346 *  np.sqrt(rh[0.98] + 100)))
		return agbd
	elif type == 'dbt_na':
		agbd = 1.052 * np.square(-120.777 + (5.508 *  np.sqrt(rh[0.5] + 100)) + (3.955 *  np.sqrt(rh[0.98] + 100)))
		return agbd

def run_sim(fn_pair, config):
	img_fn = fn_pair[0]
	pc_fn = fn_pair[1]

	bl = '_'.join(img_fn.split('_')[-3:-1])

	block = pc_block.Block(img_fn, pc_fn)
	centers = get_footprint_centers(block, config)

	agbd = np.zeros((np.shape(centers)[0], np.shape(centers)[1]), dtype=np.float32)
	centers = np.concatenate(centers, axis=0)

	pulses = pc_block.Pulses(block, centers, config)

	for indx in range(np.shape(centers)[0]):
		print(f'{img_fn} completed {indx} / {np.shape(centers)[0]} ...')
		
		if len(pulses.ret_photons[indx].photons) == 0:
			continue

		sim_wf = simulation.Waveform(block.photons, pulses.ret_photons[indx], block.ground, pulses.ret_ground[indx], config)
		val = calc_agbd(sim_wf, 'ent_na')

		ix = indx % np.shape(agbd)[0]
		iy = int(indx / np.shape(agbd)[1])

		agbd[iy, ix] = val
	
	fn = f"{config['directory']['output_dir']}agbd_km_SJER_{bl}.npy"
	np.save(fn, agbd)
	return fn

if __name__ == '__main__':
	diz_path = os.path.dirname(os.path.realpath(__file__))
	config = OmegaConf.load(os.path.join(diz_path, 'config/sim_config.yaml'))

	if not os.path.exists(config['directory']['output_dir']):
		os.makedirs(config['directory']['output_dir'])

	fn_pairs = get_fn_pairs(config['directory']['raw_data_dir'])

	pair_iter = iter(fn_pairs)

	with concurrent.futures.ProcessPoolExecutor(max_workers=config['concurrent']['workers_max']) as executor:
		futures = {
			executor.submit(run_sim, km_fn, config): km_fn for km_fn in itertools.islice(pair_iter, config['concurrent']['workers_max'])
		}

		print(futures)

		while futures:
			done, _ = concurrent.futures.wait(
				futures, return_when=concurrent.futures.FIRST_COMPLETED	
			)

			for fut in done:
				task = futures.pop(fut)
				try:
					print(f'Completed km {fut.result()}')
				except Exception as e:
					print(f'Completed {task} w/ exception {e}')
			
			for km_fn in itertools.islice(pair_iter, len(done)):
				fut = executor.submit(run_sim, km_fn, config)
				futures[fut] = km_fn