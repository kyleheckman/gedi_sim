import sys
sys.path.append('ref/')

import numpy as np
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

import simulation
import comparison_tools
import pc_block

l1b_dir = 'GEDI01_B_002-20241108_175212/'
l2a_dir = 'GEDI02_A_002-20241110_023846/'
gedi_dir = 'data/gedi/'
pc_dir = 'data/raw/SJER/2023-04/'


if __name__ == '__main__':

	coords = comparison_tools.get_blocks_w_data(pc_dir)

	sim_metrics = {}
	sim_metrics['corr'] = []
	sim_metrics['rh25'] = []
	sim_metrics['rh50'] = []
	sim_metrics['rh99'] = []

	for entry in os.listdir(f'{gedi_dir}{l1b_dir}'):
		print(f'Reading {entry} ...')

		data = comparison_tools.GEDI_Data(entry, f'{gedi_dir}{l1b_dir}', f'{gedi_dir}{l2a_dir}', coords)

		for lat_lon in coords:
			img_fn = f'{pc_dir}camera/2023_SJER_6_{lat_lon[0]}_{lat_lon[1]}_image.tif'
			pc_fn = f'{pc_dir}lidar/NEON_D17_SJER_DP1_{lat_lon[0]}_{lat_lon[1]}_classified_point_cloud_colorized.laz'

			coord_params = data.params
			mask = comparison_tools.get_result_mask(coord_params, [np.array(lat_lon).astype(int)])

			print(f'Checking block E: {np.array(lat_lon).astype(int)[0]} N: {np.array(lat_lon).astype(int)[1]}')

			coord_params = {key: value[mask] for key, value in coord_params.items()}
			print(f"--> Located {len(coord_params['start_indx'])}")

			if len(coord_params['start_indx']) == 0:
				continue

			block = pc_block.Block(img_fn, pc_fn)
			centers = [[coord_params['lat'][i], coord_params['long'][i]] for i in range(len(coord_params['lat']))]
			pulses = pc_block.Pulses(block, centers)

			for indx in range(len(pulses.pulse_centers)):
				print(f'Center: E {pulses.pulse_centers[indx][0]} N {pulses.pulse_centers[indx][1]}')
				
				if len(block.photons[pulses.ret_photons[indx].photons]) == 0:
					print('--> No photons detected')
					continue

				sim_wf = simulation.Waveform(block.photons, pulses.ret_photons[indx], block.ground, pulses.ret_ground[indx])

				start = int(coord_params['start_indx'][indx])-1
				end = int(coord_params['start_indx'][indx] + coord_params['count'][indx])-1

				# crop data waveform to match simulated waveform
				autocorr = np.convolve(sim_wf.nfw[::-1], data.waveform[start:end], mode='valid')
				shift = np.argmax(autocorr)
				sh_start = start + shift
				sh_end = sh_start + len(sim_wf.nfw)

				pcorr = stats.pearsonr(sim_wf.nfw, data.waveform[sh_start:sh_end])
				print(f'Corr: {pcorr[0]}')

				diff = [sim_wf.rh[1] - coord_params['rh'][indx][25], sim_wf.rh[2] - coord_params['rh'][indx][50], sim_wf.rh[4] - coord_params['rh'][indx][99]]
				print(f' RH25 {diff[0]} | RH50 {diff[1]} | RH99 {diff[2]}')

				sim_metrics['corr'].append(pcorr[0])
				sim_metrics['rh25'].append(diff[0])
				sim_metrics['rh50'].append(diff[1])
				sim_metrics['rh99'].append(diff[2])

				# fig, ax = plt.subplots(3, constrained_layout=True)
				# fig.suptitle(f'Waveform Comparison for Center: E: {np.array(lat_lon).astype(int)[0]} N: {np.array(lat_lon).astype(int)[1]}')

				# ax[0].set_title('Noise-Free Simluated Waveform')
				# ax[0].set_ylabel('Intensity')
				# ax[0].plot(sim_wf.nfw)

				# ax[1].set_title('Cropped GEDI Waveform')
				# ax[1].set_ylabel('DN')
				# ax[1].set_xlabel('Time (ns)')
				# ax[1].plot(data.waveform[sh_start:sh_end])

				# ax[2].set_title('Complete GEDI Waveform')
				# ax[2].set_ylabel('DN')
				# ax[2].set_xlabel('Time (ns)')
				# ax[2].plot(data.waveform[start:end])

				# plt.show()
				# plt.close()

	print(f"Mean Corr: {np.average(sim_metrics['corr'])} | StDev: {np.std(sim_metrics['corr'])}")
	print(f"Mean Abs Bias RH25: {np.average(sim_metrics['rh25'])} | StDev: {np.std(sim_metrics['rh25'])} | RMSE: {np.sqrt(np.average(np.square(sim_metrics['rh25'])))}")
	print(f"Mean Abs Bias RH50: {np.average(sim_metrics['rh50'])} | StDev: {np.std(sim_metrics['rh50'])} | RMSE: {np.sqrt(np.average(np.square(sim_metrics['rh50'])))}")
	print(f"Mean Abs Bias RH99: {np.average(sim_metrics['rh99'])} | StDev: {np.std(sim_metrics['rh99'])} | RMSE: {np.sqrt(np.average(np.square(sim_metrics['rh99'])))}")