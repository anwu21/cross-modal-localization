import os, ast
import glob
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import random, math
import itertools

from pandaset import DataSet

def get_tile(lat, lon, zoom=18):
    """
        Generates X,Y tile coordinate based on the latitude, longitude, and zoom level
    """
    n = math.pow(2, zoom)
    # Calculate the tile x:
    tile_x = int(math.floor((lon + 180)*n / 360))
    # Calculate the tile y:
    tile_y = int(math.floor((n/2)*(1-(math.asinh(math.tan(math.radians(lat))))/math.pi)))
     
    return tile_x, tile_y

class DatasetLidNoiseShuffle(object):

    """Data Handler for loading Lidar data and Satellite images. Noisy Lidar heading wrt Satellite"""

    def __init__(self, data_root, train=True, image_size=64, skip_pixels=1, Mfine=1.5, Mcoarse=3.0, noise=0, num_rand=100, \
                 transform_sat=None, transform_lid=None):

        self.root_dir = data_root
        self.image_size = image_size
        self.skip_pixels = skip_pixels
        self.Mfine = Mfine
        self.Mcoarse = Mcoarse
        self.noise = noise
        self.num_rand = num_rand
        self.transform_sat = transform_sat
        self.transform_lid = transform_lid
        self.train = train
        self.phi = 0
        self.dirs = []
        self.map_fnames = []
        self.lid_fnames = []
        self.pixel_gps = []
        self.seed_is_set = False
        self.d = 0

        self.odom = []

        if self.train:
            self.data_dir = os.path.join(self.root_dir, 'train')
        else:
            self.data_dir = os.path.join(self.root_dir, 'test')

        dataset = DataSet(self.data_dir)

        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                if os.path.isdir(os.path.join(self.data_dir, d1, d2)):
                    self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))       

        for d in self.dirs:
            gps_dir = d+'/sat_images'
            map_dir = d+'/sat_images/processed_maps64'
            lid_dir = d+'/lidar_imgs'
            
            num_files = len(os.listdir(lid_dir)) - 3  #chop off last files: sensors have diff number of readings in each ntc bag

            with open(os.path.join(gps_dir, 'pixel_gps.txt'), 'r') as f:
                self.pixel_gps.append(f.read().splitlines()[:num_files])

            # Predefined random crops of 100 samples for test only    
            with open('random_arrays/crop_x_feb1.txt', 'r') as f:
                self.crop_x_3x3 = f.read().splitlines()
            with open('random_arrays/crop_y_feb1.txt', 'r') as f:
                self.crop_y_3x3 = f.read().splitlines()

            # Get heading and gps latlon
            files = os.listdir(d)
            for f in files:
                if 'odom_synced.txt' in f:
                    odom_file = os.path.join(d, f)
                if 'gps_synced.txt' in f:
                    gps_file = os.path.join(d, f)
            with open(odom_file, "r") as file:
                self.odom += [ast.literal_eval(line.rstrip('\n')) for line in file][:num_files]
            with open(gps_file, "r") as file:
                gps = [ast.literal_eval(line.rstrip('\n')) for line in file][:num_files]
            lat_list = [item[1] for item in gps][:num_files]
            lon_list = [item[2] for item in gps][:num_files]

            for i in range(num_files):
                lid_fname = os.path.join(lid_dir, 'img_'+str(i).zfill(10)+'.png')

                lat = float(lat_list[i])
                lon = float(lon_list[i])
                tile_x, tile_y = get_tile(lat, lon, zoom=18)
                map_fname = os.path.join(map_dir, 'map_' + str(tile_x) + '_' + str(tile_y) + '.png')

                self.lid_fnames.append(lid_fname)
                self.map_fnames.append(map_fname)
        self.pixel_gps = list(itertools.chain.from_iterable(self.pixel_gps))


    def __len__(self):
        return len(self.lid_fnames)

    def __getitem__(self, index):
        self.set_seed(index)

        #pytorch image shape: [channels, height, width]
        #batch shape: [batch_size, channels, height, width]
        
        lid_fname = self.lid_fnames[index]
        lid_im = Image.open(lid_fname)
        lid_im = lid_im.convert("L")
        
        if self.image_size == 64:
            lid_im = transforms.Resize(64)(lid_im)

        map_fname = self.map_fnames[index]
        map_im = Image.open(map_fname)
        map_im = map_im.convert("RGB")

        # pixel_gps for 256x256, so must divide by 4 in each axis to get 64x64
        pixel_gps_x = int(self.pixel_gps[index].strip('][').split(', ')[0]) // 4
        pixel_gps_y = int(self.pixel_gps[index].strip('][').split(', ')[1]) // 4

        tile_size = self.image_size

        # make Mfine x Mfine size maps centered around ego-tile, where Mfine is the number of tiles of size tile_size
        map_size_fine = int(self.Mfine*tile_size)
        
        left = pixel_gps_x - self.Mfine*(tile_size // 2)
        top = pixel_gps_y - self.Mfine*(tile_size // 2)
        right = left + map_size_fine
        bottom = top + map_size_fine
        map_im_fine = map_im.crop((left, top, right, bottom))

        # create tiles to feed network
        skip = self.skip_pixels
        num_tiles = ((map_size_fine - tile_size) // skip) + 1

        sat_tile_tensor_arr = []
##        sat_tensor = self.transform_sat(sat_im)

        heading = np.degrees(2*math.atan2(self.odom[index][3], self.odom[index][4]))
        if self.noise == 0:
            lid_tensor = self.transform_lid(F.affine(lid_im, angle=-heading, translate=[0,0], scale=1, shear=0))
        else:
            angle_noise = random.uniform(-self.noise, self.noise)
            lid_im_noisy = F.affine(lid_im, angle=-heading+angle_noise, translate=[0,0], scale=1, shear=0)
            lid_tensor = self.transform_lid(lid_im_noisy)

##        Lid and Sat Image Alignment Test ###################################################################
##        lid_im.save('lid_im'+str(index)+'.png')
##        map_im.save(os.getcwd()+'/map_im'+str(index)+'.png')
##        map_im_fine.save('map_im_fine'+str(index)+'.png')
##        lid_im_aligned = F.affine(lid_im, angle=-heading, translate=[0,0], scale=1, shear=0).convert('RGBA')
##        lid_im_aligned.save('lid_im_aligned'+str(index)+'.png')
##        left = pixel_gps_x - (tile_size // 2)
##        top = pixel_gps_y - (tile_size // 2)
##        right = left + tile_size
##        bottom = top + tile_size
##        sat_im = map_im.crop((left, top, right, bottom)).convert('RGBA')
##        sat_im.save('sat_im'+str(index)+'.png')
##        combined_im = Image.blend(sat_im, lid_im_aligned, 0.5)
##        combined_im.save('combined'+str(index)+'.png')
##        exit()
##############################################################################################################
            
        # make Mcoarse x Mcoarse maps, where Mcoarse is the number of tiles of size tile_size
        map_size = int(self.Mcoarse*tile_size)
        col_min = max(0, pixel_gps_x-map_size+(tile_size//2))       
        col_max = min(pixel_gps_x-(tile_size//2), 9*tile_size-map_size)  
        row_min = max(0, pixel_gps_y-map_size+(tile_size//2))
        row_max = min(pixel_gps_y-(tile_size//2), 9*tile_size-map_size)

        # get top-left corner coordinates
        if self.train:
            crop_x = random.randint(col_min, col_max)
            crop_y = random.randint(row_min, row_max)
        else:
            crop_x = int(self.crop_x_3x3[index//16])
            crop_y = int(self.crop_y_3x3[index//16])
        
        left = crop_x
        top = crop_y
        right = left + map_size
        bottom = top + map_size
        map_im_coarse = map_im.crop((left, top, right, bottom))       

        A = random.randint(0, self.num_rand-1)
        B = self.num_rand - A
        K = map_size-tile_size-1
        # slide window in map of size Mfine x Mfine with intermittent groups of samples from map of size Mcoarse x Mcoarse
        for i in range(num_tiles):
            for m in range(A):
                left = random.randint(0, K) 
                top = random.randint(0, K)
                tile = map_im_coarse.crop((left, top, left+tile_size, top+tile_size))
                sat_tile_tensor = self.transform_sat(tile)
                sat_tile_tensor_arr.append(sat_tile_tensor)
            for j in range(num_tiles):
                left = i*skip
                top = j*skip
                tile = map_im_fine.crop((left, top, left+tile_size, top+tile_size))
                if i == num_tiles//2 and j == num_tiles//2:
                    sat_tensor = self.transform_sat(tile)
                sat_tile_tensor = self.transform_sat(tile)
                sat_tile_tensor_arr.append(sat_tile_tensor)
            for n in range(B):
                left = random.randint(0, K) 
                top = random.randint(0, K)
                tile = map_im_coarse.crop((left, top, left+tile_size, top+tile_size))
                sat_tile_tensor = self.transform_sat(tile)
                sat_tile_tensor_arr.append(sat_tile_tensor)

        sat_tile_tensor_arr = torch.stack(sat_tile_tensor_arr)

        return sat_tile_tensor_arr, lid_tensor, sat_tensor
    
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

