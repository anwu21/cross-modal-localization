import os, argparse, ast
import random, math
import itertools
import progressbar, time
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import utils_xf
import self_src as src_models

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--channels_in', default=4, type=int)
parser.add_argument('--channels_out', default=1, type=int)
parser.add_argument('--dataset', default='cml_shuffle', help='dataset to train with')
parser.add_argument('--data_root', default='data/set', help='root directory for data')
parser.add_argument('--data_threads', default=5, type=int, help='number of data loading threads')
parser.add_argument('--epoch_size', default=210, type=int, help='epoch size')
parser.add_argument('--g_dim', default=256, type=int, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--model', default='cct_2', help='model type (cct_2 | dcgan | vgg | resnet)')
parser.add_argument('--model_dir', default='', help='base directory to save trained models')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--niter', default=1000, type=int, help='number of epochs to train for')
parser.add_argument('--noise', default=0, type=int, help='compass noise in degrees for training')
parser.add_argument('--num_conv_layers', default=2, type=int, help='number of conv layers in tokenizer')
parser.add_argument('--num_rand', default=10, type=int, help='number of random samples in addition to area encasing ego tile')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--pos_emb', default='learnable', help='learnable, sine, or none')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--skip_pixels', default=1, type=int, help='number of pixels to skip for next sat tile for training')
parser.add_argument('--Mfine', default=1.5, type=float, help='size of map area encasing ego tile')
parser.add_argument('--Mcoarse', default=3., type=float, help='size of coarse map')
parser.add_argument('--alpha', default=1.0, type=float, help='sat loss multiplier for sat+lid_arr only')

parser.add_argument('--wp_test', dest='wp_test', default=False, action='store_true', help='waypoint test')
parser.add_argument('--rot_fine', dest='rot_fine', default=False, action='store_true', help='whether to perform +/-0.1 deg fine rotation for 2nd stage')
parser.add_argument('--Mtest', default=3., type=float, help='size of test map')
parser.add_argument('--group_size', default=10000, type=int, help='sat group size')
parser.add_argument('--noise_wp', default=10, type=int, help='compass noise in degrees for wp test')
parser.add_argument('--num_samples', type=int, default=100, help='number of samples')
parser.add_argument('--start', type=int, default=10, help='min epoch to start testing')
parser.add_argument('--skip_pixels_test', default=4, type=int, help='number of pixels to skip for next sat tile for wp test for 1st stage')
      
opt = parser.parse_args()

name = 'cml_self-model=%s-conv_lys=%d-pos_emb=%s-lr=%.6f-epoch_size=%d-niter=%d-bs=%d-Mfine=%.1f-Mcoarse=%.1f-Nrand=%d-Noise=%d-dataset=%s' \
       % (opt.model, opt.num_conv_layers, opt.pos_emb, opt.lr, opt.epoch_size, opt.niter, opt.batch_size, opt.Mfine, opt.Mcoarse, opt.num_rand, opt.noise, opt.data_root[5:])

if opt.wp_test == True:
    opt.model_dir = 'logs/xf_cml'
else:
    opt.model_dir = 'logs/xf_cml_no_wp'
    
opt.log_dir = '%s/%s' % (opt.model_dir, name)
os.makedirs(opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

cml_net = src_models.__dict__[opt.model](img_size=opt.image_width,
                                      num_classes=1,
                                      positional_embedding=opt.pos_emb,
                                      n_conv_layers=opt.num_conv_layers,
                                      kernel_size=3,
                                      patch_size=4,
                                      pretrained=False,
                                      arch='',
                                      progress='')
#print(cml_net)
##model_parameters = filter(lambda p: p.requires_grad, cml_net.parameters())
##params = sum([np.prod(p.size()) for p in model_parameters])
##print ('Total trainable parameters: ', params)
##exit()
cml_net = torch.nn.DataParallel(cml_net)
cml_net_optimizer = opt.optimizer(cml_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

root0 =  os.path.abspath(os.path.join(__file__ ,".."))
        
# --------- loss functions ------------------------------------
criterion = nn.SmoothL1Loss()

# --------- transfer to gpu ------------------------------------
cml_net.cuda()

criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils_xf.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         pin_memory=True)

normalize_sat = transforms.Normalize(mean=(0.53, 0.58, 0.59), std=(0.1, 0.1, 0.1))
normalize_lid = transforms.Normalize([0.02], [0.13])
transform_sat=transforms.Compose([transforms.Resize(64),
                                  transforms.ToTensor(),
                                  normalize_sat])
transform_lid=transforms.Compose([transforms.Resize(64),
                                  transforms.ToTensor(),
                                  normalize_lid])

def train(stacked_arr, sat_arr, sat_label):
    cml_net.zero_grad()

    vector = cml_net(stacked_arr)
    vector = vector.view(opt.batch_size, -1)
    vector_softmax = F.softmax(vector, dim=1)
    
    sat_sum = torch.einsum('bi,bicjk->bcjk', vector_softmax, sat_arr)
    loss = criterion(sat_sum, sat_label)
            
    loss.backward()  
    cml_net_optimizer.step()
    
    return loss.data.cpu().numpy()


def test(stacked_arr, sat_arr, sat_label):

    with torch.no_grad():
        vector = cml_net(stacked_arr)
        vector = vector.view(opt.batch_size, -1)
        vector_softmax = F.softmax(vector, dim=1)
          
        sat_sum = torch.einsum('bi,bicjk->bcjk', vector_softmax, sat_arr)
        loss = criterion(sat_sum, sat_label)

    return loss.data.cpu().numpy()

def get_lidar_image(sample):
    data_dir, k = sample
    lid_im = Image.open(os.path.join(data_dir, 'lidar_imgs', 'img_'+str(k).zfill(10)+'.png'))
    lid_im = lid_im.convert("L")
    
    return lid_im

def get_heading(sample):
    data_dir, k = sample
    files = os.listdir(data_dir)
    for f in files:
        if 'odom_synced.txt' in f:
            odom_file = os.path.join(data_dir, f)
    with open(odom_file, "r") as file:
        odom = [ast.literal_eval(line.rstrip('\n')) for line in file]
    heading = np.degrees(2*math.atan2(odom[k][3], odom[k][4]))
    
    return heading #degrees

def get_map(sample):
    data_dir, k = sample
    files = os.listdir(data_dir)
    for f in files:
        if 'gps_synced.txt' in f:
            gps_file = os.path.join(data_dir, f)
    with open(gps_file, "r") as file:
        gps = [ast.literal_eval(line.rstrip('\n')) for line in file]
    lat = gps[k][1]
    lon = gps[k][2] 
    tile_x, tile_y = get_tile(lat, lon, zoom=18)
    map_dir = os.path.join(data_dir, 'sat_images/processed_maps64')
    map_fname = os.path.join(map_dir, 'map_' + str(tile_x) + '_' + str(tile_y) + '.png')
    map_im = Image.open(map_fname)
    map_im = map_im.convert("RGB")
    
    return map_im

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

def make_dataset(index, sample, skip, tile_size, map_size, num_tiles, noise):

    data_dir, k = sample # "k" is the index of the sample within a run, whereas "index" is the index of the waypoints
    
    # load lidar image, convert to tensor
    lid_im = get_lidar_image(sample)
    heading = get_heading(sample)
    lid_tensor_rot_arr_10deg = []

    noise_mag = noise
    with open(root0+'/random_arrays/noise_%ddeg_mar12.txt' % noise_mag, 'r') as f:
        phi = float(f.read().splitlines()[index])  #degrees
        
    for theta in range(-noise_mag, noise_mag+1, 1):
        lid_im_rot = TF.affine(lid_im, angle=-heading+phi+theta, translate=[0,0], scale=1, shear=0)
        lid_tensor_rot = transform_lid(lid_im_rot)
        lid_tensor_rot_arr_10deg.append(lid_tensor_rot)

    # load map image
    map_im = get_map(sample)

    # get pixel location of online gps coordinates in the 9x9 map
    gps_fname = os.path.join(data_dir, 'sat_images/pixel_gps.txt')
    with open(gps_fname, 'r') as f:
        pixel_gps = f.read().splitlines()
    pixel_gps_x = int(pixel_gps[int(k)].strip('][').split(', ')[0]) // 4
    pixel_gps_y = int(pixel_gps[int(k)].strip('][').split(', ')[1]) // 4
    
    # make MxM maps, where M is the number of tiles of size tile_size
    # get top-left corner coordinates
    with open(root0+'/random_arrays/crop_x_feb1.txt', 'r') as f:
        crop_x = int(f.read().splitlines()[index])
    with open(root0+'/random_arrays/crop_y_feb1.txt', 'r') as f:
        crop_y = int(f.read().splitlines()[index])

    # keep track of online (ego) pixel location in new 3x3 map
    pixel_ego_x = pixel_gps_x - crop_x
    pixel_ego_y = pixel_gps_y - crop_y
    
    left = crop_x
    top = crop_y
    right = left + map_size
    bottom = top + map_size
    map_im = map_im.crop((left, top, right, bottom))

    # create tiles to feed network
    stacked_tensor_arr = []
    sat_tensor_arr = []
    sat_im_list = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            tile = map_im.crop((i*skip, j*skip, i*skip+tile_size, j*skip+tile_size))
            sat_tile_tensor = transform_sat(tile)
            sat_tensor_arr.append(sat_tile_tensor)
            
    return sat_tensor_arr, lid_tensor_rot_arr_10deg, pixel_ego_x, pixel_ego_y, phi, map_im

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def assemble_test_data(samples, noise):
    skip = opt.skip_pixels_test
    tile_size = opt.image_width
    map_size = int(opt.Mtest*tile_size)
    num_tiles = ((map_size - tile_size) // skip) + 1
    data = []
    for index, sample in enumerate(samples):
        print(sample)
        sat_arr, lid_tensor_rot_arr, col_true, row_true, ang_true, map_im = make_dataset(index, sample, skip, tile_size, map_size, num_tiles, noise)
        data.append((sat_arr, lid_tensor_rot_arr, col_true, row_true, ang_true, num_tiles, map_im))
    return data

def make_subset(lid_tensor_rot_arr, map_im, tile_size, rots, cols, rows):
    """
        For stage-2 search, this creates sub-dataset around map regions of top scores from stage-1 search.
    """
    
    skip = opt.skip_pixels_test
    D = skip // 2
    Rot_mag = 1.0
    Rot_res = Rot_mag / 5
    R = list(np.arange(-Rot_mag, Rot_mag+Rot_res, Rot_res))
    stacked_tensor_arr = []
    sat_tensor_arr = []
    sat_im_list = []
    pose_list = []

    for i, rot in enumerate(rots):
        for a in range(-D, D+1):
            left = cols[i][0] + a
            for b in range(-D, D+1):               
                top = rows[i][0] + b
                right = left + tile_size
                bottom = top + tile_size

                tile = map_im.crop((left, top, right, bottom))
                #tile.save(os.getcwd()+'/out_test/'+str(i)+'_'+str(j)+'.png')

                sat_tile_tensor = transform_sat(tile)
                sat_tensor_arr.append(sat_tile_tensor)

                if opt.rot_fine:
                    for rho in R:
                        lid_tensor = TF.affine(lid_tensor_rot_arr[rot[0]], angle=float(rho), translate=[0,0], scale=1, shear=0)
                        stacked_tensor = torch.cat([lid_tensor, sat_tile_tensor], 0)
                        stacked_tensor_arr.append(stacked_tensor)
                        pose_list.append([rot[0]+rho,left,top])
                else:
                    stacked_tensor = torch.cat([lid_tensor_rot_arr[rot[0]], sat_tile_tensor], 0)
                    stacked_tensor_arr.append(stacked_tensor)
                    pose_list.append([rot[0],left,top])
                    
    stacked_tensor_arr = torch.stack(stacked_tensor_arr)
    return stacked_tensor_arr, pose_list

def run_test(test_data, noise):
    with torch.no_grad():
        col_err_arr = []
        row_err_arr = []
        abs_err_arr = []
        ang_err_arr = []
        
        col_err_sum = 0.
        row_err_sum = 0.
        abs_err_sum = 0.
        ang_err_sum = 0.
        hits10 = 0
        hits5 = 0

        skip = opt.skip_pixels_test
        tile_size = opt.image_width

        for sample in test_data:
            sat_array, lid_tensor_rot_arr, col_true, row_true, ang_true, num_tiles, map_im = sample

            vector = torch.empty(0)
            
            row_true = row_true-opt.image_width//2
            col_true = col_true-opt.image_width//2
            #print(len(sat_arr))
            
##            time_start = time.process_time()

            for lid_tensor_rot in lid_tensor_rot_arr:
                for sat_group in chunker(sat_array, opt.group_size):
                    sat_arr = torch.stack(sat_group).cuda()
                    N = sat_arr.shape[0]
                    lid_ego_arr = lid_tensor_rot.repeat(N, 1, 1, 1).cuda()
                    stacked_arr = torch.cat([lid_ego_arr, sat_arr], 1).cuda()                                              
                        
                    vector_sub = cml_net(stacked_arr).data.cpu()
                    
                    vector = torch.cat([vector, vector_sub], 0)

##################################################################################################                    
## For test purposes only for cases where gpu memory large enough so no chunking needed (~2.5 sec)
##                sat_arr = torch.stack(sat_array).cuda()
##                N = sat_arr.shape[0]
##                lid_ego_arr = lid_tensor_rot.repeat(N, 1, 1, 1).cuda()
##                stacked_arr = torch.cat([lid_ego_arr, sat_arr], 1).cuda()                                                 
##                vector = cml_net(stacked_arr)
##################################################################################################
                    
##            time_end = time.process_time()
##            time_diff = time_end - time_start
##            print(time_diff)
##            exit()
            
            tile_topk = torch.topk(vector, dim=0, k=10)
            top_indices = tile_topk.indices.numpy()
            top_values = tile_topk.values.numpy()

            rots = top_indices // (num_tiles * num_tiles)

            top_locs = top_indices % (num_tiles * num_tiles)
            cols = (top_locs // num_tiles) * skip
            rows = (top_locs % num_tiles) * skip

            stacked_arr, poses = make_subset(lid_tensor_rot_arr, map_im, tile_size, rots, cols, rows)
            vector = cml_net(stacked_arr.cuda())

            tile_topk_subset = torch.topk(vector, dim=0, k=3)
            top_indices_subset = tile_topk_subset.indices.data.cpu().numpy()
            
            rot, col, row = poses[top_indices_subset[0][0]]

##            time_end = time.process_time()
##            time_diff = time_end - time_start
##            print(time_diff)

            col_err = col - col_true
            row_err = row - row_true
            col_err_abs = abs(col_err)
            row_err_abs = abs(row_err)
            abs_err = np.sqrt(col_err**2 + row_err**2)
            
            col_err_arr.append(col_err)
            row_err_arr.append(row_err)
            abs_err_arr.append(abs_err)
            
            col_err_sum += col_err_abs
            row_err_sum += row_err_abs
            abs_err_sum += abs_err

            if abs_err < 10:
                hits10 += 1
            if abs_err < 5:
                hits5 += 1

            ang_err = -(rot - noise) - ang_true
 
            ang_err_abs = abs(ang_err)
            ang_err_arr.append(ang_err)
            ang_err_sum += ang_err_abs

        col_err_avg = col_err_sum/opt.num_samples
        row_err_avg = row_err_sum/opt.num_samples
        abs_err_avg = abs_err_sum/opt.num_samples
        ang_err_avg = ang_err_sum/opt.num_samples
       
    return col_err_avg, row_err_avg, abs_err_avg, ang_err_avg, col_err_arr, row_err_arr, abs_err_arr, ang_err_arr, hits5, hits10

# --------- training loop ------------------------------------
best_loss = 10000.0
best_loss_test = 10000.0
best_abs_err_1deg = 10000.0
best_abs_err_5deg = 10000.0
best_abs_err_10deg = 10000.0
best_abs_err_15deg = 10000.0
best_ang_err_1deg = 10000.0
best_ang_err_5deg = 10000.0
best_ang_err_10deg = 10000.0
best_ang_err_15deg = 10000.0
best_hits5 = 0
best_hits10 = 0

err_thresh = 10000.0
noise = opt.noise_wp
samples_all = []
samples = []
data_dir_wp = root0 + '/%s/test/' % opt.data_root
if opt.wp_test == True:
    print('assembling test data...')
    sn_list = os.listdir(data_dir_wp)
    for sn in sn_list:
        for d in os.listdir(os.path.join(data_dir_wp, sn)):
            d_path = os.path.join(data_dir_wp, sn, d)
            num_lids_wp = len(os.listdir(os.path.join(d_path, 'lidar_imgs'))) - 5
            for i in range(num_lids_wp):
                samples_all.append([d_path, i])
    N_wp = len(samples_all)
    sample_indices = list(range(1, N_wp, N_wp//opt.num_samples))[:opt.num_samples]
    for i, sample in enumerate(samples_all):
        if i in sample_indices:
            samples.append(sample)
    wp_test_data = assemble_test_data(samples, noise)
    print('finished assembling test data')
    data_log_dir = opt.log_dir+'/noise_wp_'+str(noise)
    os.makedirs(data_log_dir, exist_ok=True)
                    
for epoch in range(opt.niter):
    
    ###### TRAIN ######
    epoch_err = 0.
    cml_net.train()

    for i in range(opt.epoch_size):
        print('train niter/epoch: %d/%d ' % (epoch,i))
        sat_arr, lid_tensor, sat_tensor = next(iter(train_loader))
        N = sat_arr.shape[1]
        sat_arr = sat_arr.reshape(opt.batch_size*N, 3, opt.image_width, opt.image_width).cuda()
        sat_label = sat_tensor.cuda()
        lid_ego_arr = lid_tensor.repeat(N, 1, 1, 1).cuda()

        stacked_arr = torch.cat([lid_ego_arr, sat_arr], 1)
   
        sat_arr = sat_arr.reshape(opt.batch_size, N, 3, opt.image_width, opt.image_width)
        
        err = train(Variable(stacked_arr), Variable(sat_arr), Variable(sat_label))
        epoch_err += err

    print('TRAIN: [%02d] mse loss: %.10f (%d)' % (epoch, epoch_err/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    
    ###### TEST ######
    test_epoch_size = max(int(opt.epoch_size / 10.0), 1)
    epoch_err_test = 0.
    
    cml_net.eval()
    with torch.no_grad():
        for i in range(test_epoch_size):
            print('test niter/epoch: %d/%d ' % (epoch,i))
            sat_arr, lid_tensor, sat_tensor = next(iter(train_loader))
            N = sat_arr.shape[1]
            sat_arr = sat_arr.reshape(opt.batch_size*N, 3, opt.image_width, opt.image_width).cuda()
            sat_label = sat_tensor.cuda()
            lid_ego_arr = lid_tensor.repeat(N, 1, 1, 1).cuda()

            stacked_arr = torch.cat([lid_ego_arr, sat_arr], 1)
    
            sat_arr = sat_arr.reshape(opt.batch_size, N, 3, opt.image_width, opt.image_width)
            
            err = test(Variable(stacked_arr), Variable(sat_arr), Variable(sat_label))
            epoch_err_test += err

        print('TEST: [%02d] mse loss: %.10f (%d)' % (epoch, epoch_err_test/test_epoch_size, epoch*test_epoch_size*opt.batch_size))

        with open('%s/log_train.txt' % opt.log_dir, 'a') as f:
            f.write('TRAIN: [%02d] mse loss: %.10f (%d)\n' % (epoch, epoch_err/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
        with open('%s/log_test.txt' % opt.log_dir, 'a') as f:
            f.write('TEST: [%02d] mse loss: %.10f (%d)\n' % (epoch, epoch_err_test/test_epoch_size, epoch*test_epoch_size*opt.batch_size))

        opt.lr = 0.99 * opt.lr

        # set error threshold to perform waypoint test runs at lowest err amongst first opt.start epochs
        if epoch <= opt.start:
            if epoch_err_test < err_thresh:
                err_thresh = epoch_err_test
            
        if epoch > opt.start and opt.wp_test:
            if epoch_err_test < best_loss_test:
                best_loss_test = epoch_err_test
                torch.save(cml_net.state_dict(), '%s/model_cml64_dict_dp_loss.pth' % opt.log_dir)
                torch.save(cml_net.module.state_dict(), '%s/model_cml64_dict_module_loss.pth' % opt.log_dir)

            col_err, row_err, abs_err, ang_err, col_err_arr, row_err_arr, abs_err_arr, ang_err_arr, hits5, hits10 = run_test(wp_test_data, noise)

            summary = [epoch, epoch_err_test/test_epoch_size, noise, col_err, row_err, abs_err, ang_err, hits5, hits10]
            with open(data_log_dir+'/summary.txt', 'a') as f:
                f.write('%s\n' % str(summary))
                
            with open(data_log_dir+'/err_col_'+str(epoch)+'.txt', 'w') as f:
                for listitem in col_err_arr:
                    f.write('%s\n' % listitem)
            with open(data_log_dir+'/err_row_'+str(epoch)+'.txt', 'w') as f:
                for listitem in row_err_arr:
                    f.write('%s\n' % listitem)
            with open(data_log_dir+'/err_abs_'+str(epoch)+'.txt', 'w') as f:
                for listitem in abs_err_arr:
                    f.write('%s\n' % listitem)
            with open(data_log_dir+'/err_ang_'+str(epoch)+'.txt', 'w') as f:
                for listitem in ang_err_arr:
                    f.write('%s\n' % listitem)
                    
            if abs_err < best_abs_err_10deg:
                best_abs_err_10deg = abs_err
                torch.save(cml_net.state_dict(), '%s/model_cml64_dict_dp_abs' % opt.log_dir+'.pth')
                torch.save(cml_net.module.state_dict(), '%s/model_cml64_dict_module_abs' % opt.log_dir+'.pth')
                with open(data_log_dir+'/err_abs_best_'+str(noise)+'deg.txt', 'w') as f:
                    for listitem in abs_err_arr:
                        f.write('%s\n' % listitem)
            if ang_err < best_ang_err_10deg:
                best_ang_err_10deg = ang_err
                torch.save(cml_net.state_dict(), '%s/model_cml64_dict_dp_ang' % opt.log_dir+'.pth')
                torch.save(cml_net.module.state_dict(), '%s/model_cml64_dict_module_ang'% opt.log_dir+'.pth')
                with open(data_log_dir+'/err_ang_best_'+str(noise)+'deg.txt', 'w') as f:
                    for listitem in ang_err_arr:
                        f.write('%s\n' % listitem)


