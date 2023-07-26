import os
import time
import numpy as np
import torch

class Config(object):
    def __init__(self, mode='train', date=None):

        if date:
            self.date = date
        else:   
            self.date = '0420'

        self.data_dir = '../'
        # self.data_dir = 'D:/deepmag_data'
        self.save_dir = 'output/weights_date_{}'.format(self.date)
        self.n_channels = 3 # number of channels of input images

        if mode == 'train':
            # General
            self.epochs = 12
            # self.GPUs = '0'
            self.batch_size = 8     # * torch.cuda.device_count()     # len(self.GPUs.split(','))

            # Data
            self.dir_train = os.path.join(self.data_dir, 'train')
            self.frames_train = 'coco100000'        # you can adapt 100000 to a smaller number to train
            self.cursor_end = int(self.frames_train.split('coco')[-1])        # number of training pairs
            self.coco_amp_lst = np.loadtxt(os.path.join(self.dir_train, 'train_mf.txt'))[:self.cursor_end]         # amplication factor for each training pair (train_mf.txt)
            self.videos_train = []
            self.load_all = False        # Don't turn it on, unless you have such a big mem.
                                         # On coco dataset, 100, 000 sets -> 850G

            # Training
            self.lr = 1e-4
            self.betas = (0.9, 0.999)
            self.preproc = ['poisson']   
            # ['resize', 'BF', 'downsample', 'poisson']
            self.pretrained_weights = ''
            self.validate = True
            self.validate_ratio = 0.2

            # Callbacks
    #         self.num_val_per_epoch = 10
            self.log_dir = 'output/logs_date_{}'.format(self.date)
            self.time_st = time.time()
            self.losses = []

        else: # test
            self.dir_test = os.path.join(self.data_dir, 'test')
            self.dir_water = os.path.join(self.data_dir, 'vids/water')
            self.dir_baby = os.path.join(self.data_dir, 'vids/baby')
            self.dir_gun = os.path.join(self.data_dir, 'vids/gun')
            self.dir_drone = os.path.join(self.data_dir, 'vids/drone')
            self.dir_guitar = os.path.join(self.data_dir, 'vids/guitar')
            self.dir_cattoy = os.path.join(self.data_dir, 'vids/cattoy')
            self.dir_drum = os.path.join(self.data_dir, 'vids/drum')

            self.batch_size_test = 1
