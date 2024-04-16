import math
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import functools
from scipy import signal, ndimage
from PIL import Image, ImageDraw

from torchvision import datasets, transforms
from torch.autograd import Variable

normalize_sat = transforms.Normalize(mean=(0.53, 0.58, 0.59), std=(0.1, 0.1, 0.1))
normalize_lid = transforms.Normalize([0.02], [0.13])

def load_dataset(opt):
    if opt.dataset == 'cml_shuffle':
        from dataset_xf import DatasetLidNoiseShuffle
        train_data = DatasetLidNoiseShuffle(
                    data_root=opt.data_root,
                    train=True,
                    image_size=opt.image_width,
                    skip_pixels=opt.skip_pixels,
                    Mfine=opt.Mfine,
                    Mcoarse=opt.Mcoarse,
                    noise=opt.noise,
                    num_rand=opt.num_rand,
                    transform_sat=transforms.Compose([transforms.ToTensor(),
                                                      normalize_sat]),
                    transform_lid=transforms.Compose([transforms.ToTensor(),
                                                      normalize_lid]))
        test_data = DatasetLidNoiseShuffle(
                    data_root=opt.data_root,
                    train=False,
                    image_size=opt.image_width,
                    skip_pixels=opt.skip_pixels,
                    Mfine=opt.Mfine,
                    Mcoarse=opt.Mcoarse,
                    noise=opt.noise,
                    num_rand=opt.num_rand,
                    transform_sat=transforms.Compose([transforms.ToTensor(),
                                                      normalize_sat]),
                    transform_lid=transforms.Compose([transforms.ToTensor(),
                                                      normalize_lid]))
 
    return train_data, test_data

def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d_fft(f,g):
  size_y = f.size(-2)
  size_x = f.size(-1)

  # anchor of g is 0,0 (flip g and wrap circular)
  g = g.flip(-2)
  g = g.flip(-1)
  g = g.roll(size_y//2, -2)
  g = g.roll(size_x//2, -1)

  # take fft of both f and g
  F_f = torch.rfft(f, signal_ndim=2, onesided=False)
  F_g = torch.rfft(g, signal_ndim=2, onesided=False)

  # complex multiply
  FxG_real = F_f[:, :, :, 0] * F_g[:, :, :, 0] - F_f[:, :, :, 1] * F_g[:, :, :, 1]
  FxG_imag = F_f[:, :, :, 0] * F_g[:, :, :, 1] + F_f[:, :, :, 1] * F_g[:, :, :, 0]
  FxG = torch.stack([FxG_real, FxG_imag], dim=3)
  
  # inverse fft
  fcg = torch.irfft(FxG, signal_ndim=2, onesided=False)
  
  return fcg
