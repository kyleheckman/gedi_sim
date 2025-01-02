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

def reshape(hhdc):
	new_hhdc = np.zeros((np.shape(hhdc)[-1], np.shape(hhdc)[0], np.shape(hhdc)[1]), dtype=np.float32)
	for r in range(np.shape(hhdc)[0]):
		for c in range(np.shape(hhdc)[1]):
			for h in range(np.shape(hhdc)[-1]):
				new_hhdc[h,c,r] = hhdc[c,r,-(h+1)]
	return new_hhdc

def get_hhdc_centers(block, config):
	square_size = config['hhdc_config']['square_size']
	
	# Generate meshgrid for pulse centers, origin is offset by half of square size
	x_dim = np.arange(block.bounds.left+(square_size/2), block.bounds.right, step=square_size)
	y_dim = np.arange(block.bounds.bottom+(square_size/2), block.bounds.top, step=square_size)
	x_coords, y_coords = np.meshgrid(x_dim, y_dim)

	# Convert meshgrid to a list of centers
	centers = np.dstack((x_coords, y_coords))
	
	return centers

def prep_waveform(waveform, z_min, block_z_min, config):
	# Convert waveform to probability density, crop zeros from beginning/end
	# Invert waveform to have lower height bins at lower index
	col_wf = waveform.nfw / np.sum(waveform.nfw)
	col_wf = col_wf[np.min(np.argwhere(col_wf != 0)):np.max(np.argwhere(col_wf != 0))+1][::-1]

	# Calculate height offset for waveform if ELEVATION_SHIFT == True
	shift = 0
	if config['hhdc_config']['elevation_shift']:
		shift = int((z_min - block_z_min) / config['sim_config']['gedi_config']['resolution'])

	column = np.zeros((config['hhdc_config']['height_bins']), dtype=np.float32)

	# Crop waveform to fit HHDC height limit (not ideal - simple method for now)
	while shift + len(col_wf) >= config['hhdc_config']['height_bins']:
		col_wf = col_wf[:-1]
		if len(col_wf) == 0:
			break

	column[shift:shift + len(col_wf)] = col_wf

	# Return inverted column, allows for simply transposing a slice to output DEM
	return column[::-1]
	
def disp_dem(hhdc_slice, bounds, fn, indx, z_min, config):
	bins = config['hhdc_config']['height_bins']
	resolution = config['sim_config']['gedi_config']['resolution']

	fig, ax = plt.subplots()

	ax.imshow(hhdc_slice.T, aspect=5, interpolation='none', cmap=matplotlib.colormaps['Greens'], extent=[bounds.left,bounds.right,0,bins*resolution])
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

def run_sim(fn_pair, config):
	img_fn = fn_pair[0]
	pc_fn = fn_pair[1]

	bl = '_'.join(img_fn.split('_')[-3:-1])

	print(f'Simulating block {bl} ...')

	block = pc_block.Block(img_fn, pc_fn, downsample=100)
	centers = get_hhdc_centers(block, config)

	hhdc = np.zeros((np.shape(centers)[0], np.shape(centers)[1], config['hhdc_config']['height_bins']), dtype=np.float32)
	centers = np.concatenate(centers, axis=0)

	pulses = pc_block.Pulses(block, centers, config)

	# Minimum Z-value for entire block, used to calculate vertical offset for each waveform
	block_z_min = np.min(block.photons[:,2])

	for indx in range(np.shape(centers)[0]):
		#print(f'Computing center {indx}/{np.shape(centers)[0]} ...')
		if len(pulses.ret_photons[indx].photons) == 0:
			continue

		sim_wf = simulation.Waveform(block.photons, pulses.ret_photons[indx], block.ground, pulses.ret_ground[indx], config)

		z_min = np.min(block.photons[pulses.ret_photons[indx].photons][:,2])

		column = prep_waveform(sim_wf, z_min, block_z_min, config)

		ix = indx % np.shape(hhdc)[0]
		iy = int(indx / np.shape(hhdc)[1])

		hhdc[iy, ix, :] = column

	# Only save hhdcs at least 90% populated
	if np.sum(hhdc) < 0.9 * hhdc.shape[0] * hhdc.shape[1]:
		return f'{bl} unsaved'

	# reshape from H, W, C -> C, H, W
	hhdc = reshape(hhdc)

	fn = f"{config['directory']['output_hhdc']}{img_fn.split('_')[-5]}_{bl}_hhdc_{np.shape(hhdc)[1]}x{np.shape(hhdc)[-1]}_gridded.npy"
	np.save(fn, hhdc)
	return fn

if __name__ == '__main__':
	# True if concurrency enabled
	cc = True

	diz_path = os.path.dirname(os.path.realpath(__file__))
	config = OmegaConf.load(f'{diz_path}/config/sim_config.yaml')

	if not os.path.exists(config['directory']['output_hhdc']):
		os.makedirs(config['directory']['output_hhdc'])

	fn_pairs = get_fn_pairs(config['directory']['raw_data_dir'])

	print(fn_pairs)

	count = 0
	if cc:
		fn_pairs_iter = iter(fn_pairs[:])

		with concurrent.futures.ProcessPoolExecutor(max_workers=config['concurrent']['workers_max']) as executor:
			futures = {
				executor.submit(run_sim, km_fn, config): km_fn for km_fn in itertools.islice(fn_pairs_iter, config['concurrent']['workers_max'])
			}

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
				
				for km_fn in itertools.islice(fn_pairs_iter, len(done)):
					fut = executor.submit(run_sim, km_fn, config)
					futures[fut] = km_fn
					count += 1
					print(f'Completed: {count}')

	else:

		for km_fn in fn_pairs:
			res = run_sim(km_fn, config)
			print(f'Completed {res}')