import os
import numpy as np
import torch
import cv2
import random
import torch.nn.functional as F
from skimage.io import imread
from skimage.util import random_noise
from sklearn.utils import shuffle
from torch.autograd import Variable
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
Tensor = torch.cuda.FloatTensor


def gen_poisson_noise(unit):
    n = np.random.randn(*unit.shape)

    # Strange here, unit has been in range of (-1, 1),
    # but made example to checked to be same as the official codes.
    n_str = np.sqrt(unit + 1.0) / np.sqrt(127.5)
    poisson_noise = np.multiply(n, n_str)
    return poisson_noise


def load_unit(path):
    # Load a single frame
    file_suffix = path.split('.')[-1].lower()
    # cv2.imread reads in BGR and uint8 format.
    # skimage.io.imread reads in RGB and uint8 format, channel last
    if file_suffix in ['jpg', 'png']:
        try:
            unit = cv2.cvtColor(imread(path).astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print('{} load exception:\n'.format(path), e)
            unit = cv2.cvtColor(np.array(Image.open(path).convert('RGB')), cv2.COLOR_RGB2BGR)
        return unit
    else:
        print('Unsupported file type.')
        return None


def unit_preprocessing(unit, preproc=[], is_test=False):
    """
    Preprocessing: 
    -transform into [-1.0, 1.0]
    -bilateralFilter/resize/downsample/poissonNoise (depending on preproc option)
    -turn BGR into RGB (RBG order is kept throughout training)
    -change the shape into [channel, width, height]

    For training, batch A, B, C and M are preprocessed.
    For test, the inputs are batch A and C (both unperturbed) and both are preprocessed.
    """

    # rescale between [-1.0, 1.0]
    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    unit = unit / 127.5 - 1.0

    if 'BF' in preproc and is_test:
        unit = cv2.bilateralFilter(unit, 9, 75, 75)
    if 'resize' in preproc:
        unit = cv2.resize(unit, (384, 384), interpolation=cv2.INTER_LANCZOS4)
    elif 'downsample' in preproc:
        unit = cv2.resize(unit, unit.shape[1]//2, unit.shape[0]//2, interpolation=cv2.INTER_LANCZOS4)

    try:
        if 'poisson' in preproc:
            # Use poisson noise from official repo or skimage?
            
            # as official repo
            unit = unit + gen_poisson_noise(unit) * np.random.uniform(0, 0.3)
            
            # skimage
#             unit = random_noise(unit, mode='poisson')      # unit: 0 ~ 1.0
#             unit = unit * 255

    except Exception as e:
        print('EX:', e, unit.shape, unit.dtype)

    unit = np.transpose(unit, (2, 0, 1))
    return unit

def unit_postprocessing(unit, vid_size=None):
    """
    postprocessing:
    -clip the output frame between 0 and 255 as uint8
    -change the shape into [width, height, channel]
    """
    unit = unit.squeeze()
    unit = unit.cpu().detach().numpy()
    unit = np.clip(unit, -1, 1)
    unit = np.round((np.transpose(unit, (1, 2, 0)) + 1.0) * 127.5).astype(np.uint8)
    if unit.shape[:2][::-1] != vid_size and vid_size is not None:
        unit = cv2.resize(unit, vid_size, interpolation=cv2.INTER_CUBIC)
    return unit

def get_paths_ABC(config, mode):
    if mode in ('train', 'test_on_trainset'):
        dir_root = config.dir_train

        if config.cursor_end <= 0:
            print('No input file for training')
            return []

    elif mode == 'test_on_testset':
        dir_root = config.dir_test
    else:
        val_vid = mode.split('-')[-1]
        # dir_root = eval('config.dir_{}'.format(val_vid))
        dir_root = os.path.join(config.data_dir, 'vids/{}'.format(val_vid))

    dir_A = os.path.join(dir_root, 'frameA')
    files_A = sorted(os.listdir(dir_A), key=lambda x: int(x.split('.')[0]))
    paths_A = [os.path.join(dir_A, file_A) for file_A in files_A]
    if mode == 'train' and isinstance(config.cursor_end, int):
        paths_A = paths_A[:config.cursor_end]
    
    return paths_A

def get_gen_ABC(config, mode='train', dynamic=True):
#     paths_A = get_paths_ABC(config, mode)[0]
    paths_A = get_paths_ABC(config, mode)
    gen_train_A = DataGen(paths_A, config, mode, dynamic)
    return gen_train_A

class DataGen():
    def __init__(self, paths, config, mode, dynamic):
        self.is_train = 'test' not in mode
        self.anchor = 0
        self.paths = paths # path of frameA
        self.data_len = len(paths)

        if self.is_train: # training mode
            self.load_all = config.load_all
            self.preproc = config.preproc
            self.coco_amp_lst = list(config.coco_amp_lst)
            self.batch_size = config.batch_size

            temp = list(zip(self.paths, self.coco_amp_lst))
            random.shuffle(temp)
            random.shuffle(temp)
            random.shuffle(temp)
            self.paths, self.coco_amp_lst = zip(*temp)

            if config.validate:
                self.data_len = int(len(self.paths) * (1 - config.validate_ratio))
                self.validate_data_len = len(self.paths) - self.data_len
                self.anchor_validate = self.data_len

            if self.load_all:
                self.units_A, self.units_C, self.units_M, self.units_B = [], [], [], []
                for idx_data in range(self.data_len):
                    if idx_data % 500 == 0:
                        print('Processing {} / {}.'.format(idx_data, self.data_len))
                    unit_A = load_unit(self.paths[idx_data])
                    unit_C = load_unit(self.paths[idx_data].replace('frameA', 'frameC'))
                    unit_M = load_unit(self.paths[idx_data].replace('frameA', 'amplified'))
                    unit_B = load_unit(self.paths[idx_data].replace('frameA', 'frameB'))
                    unit_A = unit_preprocessing(unit_A, preproc=self.preproc)
                    unit_C = unit_preprocessing(unit_C, preproc=self.preproc)
                    unit_M = unit_preprocessing(unit_M, preproc=[])
                    unit_B = unit_preprocessing(unit_B, preproc=self.preproc)
                    self.units_A.append(unit_A)
                    self.units_C.append(unit_C)
                    self.units_M.append(unit_M)
                    self.units_B.append(unit_B)

        elif 'temporal' not in mode: # test mode without a temporal filter
            self.dynamic = dynamic
            self.batch_size = config.batch_size_test
            if not self.dynamic:
                self.anchor_0 = self.anchor

    def gen(self):
        batch_A = []
        batch_C = []
        batch_M = []
        batch_B = []
        batch_amp = []

        for _ in range(self.batch_size):
            if not self.load_all:
                unit_A = load_unit(self.paths[self.anchor])
                unit_C = load_unit(self.paths[self.anchor].replace('frameA', 'frameC'))
                unit_M = load_unit(self.paths[self.anchor].replace('frameA', 'amplified'))
                unit_B = load_unit(self.paths[self.anchor].replace('frameA', 'frameB'))
                unit_A = unit_preprocessing(unit_A, preproc=self.preproc)
                unit_C = unit_preprocessing(unit_C, preproc=self.preproc)
                unit_M = unit_preprocessing(unit_M, preproc=[])
                unit_B = unit_preprocessing(unit_B, preproc=self.preproc)
            else:
                unit_A = self.units_A[self.anchor]
                unit_C = self.units_C[self.anchor]
                unit_M = self.units_M[self.anchor]
                unit_B = self.units_B[self.anchor]
            unit_amp = self.coco_amp_lst[self.anchor]

            batch_A.append(unit_A)
            batch_C.append(unit_C)
            batch_M.append(unit_M)
            batch_B.append(unit_B)
            batch_amp.append(unit_amp)

            self.anchor = (self.anchor + 1) % self.data_len

        batch_A = numpy2cuda(batch_A)
        batch_C = numpy2cuda(batch_C)
        batch_M = numpy2cuda(batch_M)
        batch_B = numpy2cuda(batch_B)
        batch_amp = numpy2cuda(batch_amp).reshape(self.batch_size, 1, 1, 1)
        return batch_A, batch_B, batch_C, batch_M, batch_amp

    def gen_validate(self):
        batch_A = []
        batch_C = []
        batch_M = []
        batch_B = []
        batch_amp = []

        for _ in range(self.batch_size):
            unit_A = load_unit(self.paths[self.anchor_validate])
            unit_C = load_unit(self.paths[self.anchor_validate].replace('frameA', 'frameC'))
            unit_M = load_unit(self.paths[self.anchor_validate].replace('frameA', 'amplified'))
            unit_B = load_unit(self.paths[self.anchor_validate].replace('frameA', 'frameB'))
            unit_A = unit_preprocessing(unit_A, preproc=[], is_test=True)
            unit_C = unit_preprocessing(unit_C, preproc=[], is_test=True)
            unit_M = unit_preprocessing(unit_M, preproc=[], is_test=True)
            unit_B = unit_preprocessing(unit_B, preproc=[], is_test=True)
            unit_amp = self.coco_amp_lst[self.anchor_validate]
            batch_A.append(unit_A)
            batch_C.append(unit_C)
            batch_M.append(unit_M)
            batch_B.append(unit_B)
            batch_amp.append(unit_amp)

            self.anchor = (self.anchor_validate + 1)

        batch_A = numpy2cuda(batch_A)
        batch_C = numpy2cuda(batch_C)
        batch_M = numpy2cuda(batch_M)
        batch_B = numpy2cuda(batch_B)
        batch_amp = numpy2cuda(batch_amp).reshape(self.batch_size, 1, 1, 1)

        return batch_A, batch_B, batch_C, batch_M, batch_amp

    def gen_test(self):
        batch_A = []
        batch_C = []

        for _ in range(self.batch_size):

            if not self.dynamic:
                unit_A = load_unit(self.paths[self.anchor_0])
            else:
                unit_A = load_unit(self.paths[self.anchor])
            unit_C = load_unit(self.paths[self.anchor].replace('frameA', 'frameC'))
            unit_A = unit_preprocessing(unit_A, preproc=[], is_test=True)
            unit_C = unit_preprocessing(unit_C, preproc=[], is_test=True)
            batch_A.append(unit_A)
            batch_C.append(unit_C)

            self.anchor = (self.anchor + 1) % self.data_len

        batch_A = numpy2cuda(batch_A)
        batch_C = numpy2cuda(batch_C)

        return batch_A, batch_C

    def gen_test_temporal(self):
        batch_C = []

        unit_C = load_unit(self.paths[self.anchor].replace('frameA', 'frameC'))
        unit_C = unit_preprocessing(unit_C, preproc=[], is_test=True)

        self.anchor = (self.anchor + 1) % self.data_len

        batch_C.append(unit_C)
        batch_C = numpy2cuda(batch_C)

        return batch_C


def cuda2numpy(tensor):
    array = tensor.detach().cpu().squeeze().numpy()
    return array


def numpy2cuda(array):
    tensor = torch.from_numpy(np.asarray(array)).float().cuda()
    return tensor


def resize2d(img, size):
    with torch.no_grad():
        img_resized = (F.adaptive_avg_pool2d(Variable(img, volatile=True), size)).data
    return img_resized
