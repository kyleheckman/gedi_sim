import os
import requests

import argparse

from tqdm.auto import tqdm

from tabulate import tabulate

SERVER = 'http://data.neonscience.org/api/v0/'
DPRODUCT_CAM = 'DP3.30010.001'
DPRODUCT_LIDAR = 'DP1.30003.001'

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch NEON data')
    parser.add_argument('-o', '--output', type=str, default='./', help='Raw data output directory', required=True)
    return parser.parse_args()

def fetch_sites():
    sites_request = requests.get(SERVER+'sites/')
    sites_json = sites_request.json()

    reversed_site_data = {}
    for site_data in sites_json['data']:
        custom_keys = ['siteCode', 'siteName', 'siteDescription', 'siteType', 'siteLatitude', 'siteLongitude', 'stateCode', 'stateName', 'domainCode', 'domainName', 'deimsId']
        reversed_site_data[site_data['siteCode']] = {key: site_data[key] for key in custom_keys}

        for d_product in site_data['dataProducts']:
            if d_product['dataProductCode'] == DPRODUCT_LIDAR:
                reversed_site_data[site_data['siteCode']]['Lidar'] = d_product

    return reversed_site_data

def fetch_product(product_code):
    product_request = requests.get(SERVER+'products/'+product_code)
    product_json = product_request.json()

    return product_json

def select_site(sites_json, product_json):
    sites_w_lidar = {site_code['siteCode']: site_code for site_code in product_json['data']['siteCodes']}

    sites_table = []
    for site_code_key in sites_w_lidar.keys():
        site_code = sites_w_lidar[site_code_key]
        site_info = sites_json[site_code['siteCode']]
        sites_table.append([site_info['siteCode'], site_info['siteDescription'], site_info['stateName']])

    print('Available sites with LiDAR data:')
    print(tabulate(sites_table, headers=['Site Code', 'Side Description', 'State'], tablefmt='orgtbl'))

    selected_site = input('Select a site code: ')

    if selected_site in list(sites_w_lidar.keys()):
        return sites_w_lidar[selected_site]
    else:
        print('\nInvalid site code. Please try again.')
        return select_site(sites_json, product_json)
    
def select_month(selected_site):
    months_table = []
    for id, month in enumerate(selected_site['availableMonths']):
        months_table.append([id, month])
    
    selected_site_code = selected_site['siteCode']
    print(f'\nAvailable months for {selected_site_code}:')
    print(tabulate(months_table, headers=['ID', 'Month'], tablefmt='orgtbl'))

    selected_month = input('Select a month ID: ')

    try:
        selected_month = int(selected_month)
    except ValueError:
        print('\nInvalid month ID. Please try again.')
        return select_month(selected_site)

    if selected_month in list(range(len(selected_site['availableMonths']))):
        return selected_site['availableMonths'][selected_month], selected_site['availableDataUrls'][selected_month]
    else:
        print('\nInvalid month ID. Please try again.')
        return select_month(selected_site)

def get_file_urls(data_url, extension='.laz'):
    data_request = requests.get(data_url)
    data_json = data_request.json()

    return {file['url']: file for file in data_json['data']['files'] if file['url'].endswith(extension)}

def download_file(file_url, output_dir):
    file_request = requests.get(file_url)
    file_name = file_url.split('/')[-1]
    with open(output_dir+file_name, 'wb') as file:
        file.write(file_request.content)

def download_files(file_urls, output_dir):
    for file_url in tqdm(file_urls.keys()):
        download_file(file_url, output_dir)

def create_output_dir(selected_site, selected_month, output_dir):
    output_dir_site_root = os.path.join(output_dir, selected_site, selected_month)
    output_dir_lidar = os.path.join(output_dir_site_root, 'lidar/')
    output_dir_camera = os.path.join(output_dir_site_root, 'camera/')

    if not os.path.exists(output_dir_lidar):
        os.makedirs(output_dir_lidar)
    if not os.path.exists(output_dir_camera):
        os.makedirs(output_dir_camera)

    return output_dir_lidar, output_dir_camera, output_dir_site_root

def fetch_data(selected_site_code, selected_month, output_dir):
    camera_data_url = f'https://data.neonscience.org/api/v0/data/{DPRODUCT_CAM}/{selected_site_code}/{selected_month}'
    lidar_data_url = f'https://data.neonscience.org/api/v0/data/{DPRODUCT_LIDAR}/{selected_site_code}/{selected_month}'

    lidar_file_urls = get_file_urls(lidar_data_url, extension='_classified_point_cloud_colorized.laz')
    camera_file_urls = get_file_urls(camera_data_url, extension='.tif')

    output_dir_lidar, output_dir_camera, output_dir_site_root = create_output_dir(selected_site_code, selected_month, output_dir)

    print('\nDownloading files...')
    download_files(lidar_file_urls, output_dir_lidar)
    download_files(camera_file_urls, output_dir_camera)
    print(f'\nFiles downloaded successfully to:\n{output_dir_site_root}\n')

def main(args):
    sites_json = fetch_sites()
    product_json = fetch_product(DPRODUCT_LIDAR)

    selected_site = select_site(sites_json, product_json)
    selected_month, data_url = select_month(selected_site)
    selected_site_code = selected_site['siteCode']

    fetch_data(selected_site_code, selected_month, args.output)

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    main(args)