from pathlib import Path
import laspy
import numpy as np
from scipy.spatial import KDTree
import rasterio

# Credit to Nestor Porras-Diaz (@ZaidUD) for developing the HHDC simulator that provided the initial basis for this GEDI simulator

# ========== / LAZ file support functions / ==========
class Lidar():
    def __init__(self, ldFilePath, ignoreGround=0, downSample=None) -> None:
        datapath = Path(ldFilePath)
        self.datapath = str(datapath)

        lasData = laspy.read(self.datapath)

        self.classification = lasData.classification
        self.intensity = lasData.intensity

        if ignoreGround:
            points = (lasData.xyz[lasData.classification != 2])
        else:
            points = (lasData.xyz)

        if downSample != None:
            dsIndxs = range(0,len(points), downSample)
            points = points[dsIndxs]

        self.points = points.T
        self.ground = lasData.xyz[lasData.classification == 2].T
        self.groundIntensity = lasData.intensity[lasData.classification == 2]

        self.dataTops = [lasData.header.mins, lasData.header.maxs]
        self.time = lasData.gps_time

        # read color
        self.red = lasData.red
        self.green = lasData.green
        self.blue = lasData.blue

def get_pc_data(pc_path, includeColor=False, generate_kdtree=True):
    lidar = Lidar(pc_path)

    photons = np.concatenate([lidar.points.T, lidar.intensity.reshape(-1, 1)], axis=1)
    ground = np.concatenate([lidar.ground.T, lidar.groundIntensity.reshape(-1, 1)], axis=1)
    #include RGB color data
    if includeColor:
        photons = np.concatenate([photons, lidar.red.reshape(-1, 1), lidar.green.reshape(-1, 1), lidar.blue.reshape(-1, 1)], axis=1)

    # remove class 7
    photons = photons[lidar.classification != 7]
    
    photons = photons[::100]
    ground = ground[::100]

    if generate_kdtree:
        kdtree = KDTree(photons[...,:2])
        gnd_kd = KDTree(ground[...,:2])
    else:
        kdtree = None
        gnd_kd = None

    return kdtree, photons, gnd_kd, ground

# ========== / TIF file support functions / ==========
def get_image_information(img_fn):
    with rasterio.open(img_fn) as img:
        return {'meta': img.meta, 
                'bounds': img.bounds, # important - unit : meters
                'crs': img.crs, 
                'transform': img.transform, 
                'width': img.width, # important - unit : pixels
                'height': img.height, # important - unit : pixels
                'count': img.count, 
                'indexes': img.indexes
                }

# ========== / Other support functions / ==========
def get_pos_matrix(img_fn):
    bounds = get_image_information(img_fn)['bounds']

    x_meters = np.linspace(bounds.left, bounds.right, int(bounds.right-bounds.left)+1)
    y_meters = np.linspace(bounds.top, bounds.bottom, int(bounds.top-bounds.bottom)+1)

    x_coods, y_coods = np.meshgrid(x_meters, y_meters)

    pos_matrix = np.zeros((x_coods.shape[0], x_coods.shape[1], 2))
    pos_matrix[:,:,0] = x_coods
    pos_matrix[:,:,1] = y_coods
    
    return pos_matrix, bounds