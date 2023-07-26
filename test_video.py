"""MagNet TEST.

-put the test video frames A and C in a folder named X (for example) or folders X, Y, ...These frames can be generated with make_frameACB.py from test video frames.
-add self.dir_X, self.dir_Y, ... to config.py
-10, 20, ... as amplification factor

Usage:
    python test_video.py date X-Y-Z 10-20-30 30-24-300 yes
"""
import os
import sys
import cv2
import torch
import numpy as np
import imageio

from tqdm import trange
from config import Config
from magnet import MagNet
from data import get_gen_ABC, unit_postprocessing, numpy2cuda
from callbacks import gen_state_dict

from subprocess import call 

# Change here if you use avconv.
DEFAULT_VIDEO_CONVERTER = 'ffmpeg'

date = sys.argv[1]
testsets = sys.argv[2]
ampfacs = sys.argv[3]
frates = sys.argv[4]
dynamic = (sys.argv[5] == 'yes')

testsets = testsets.split('-')
ampfacs = ampfacs.split('-')
frates = frates.split('-')
# config
config = Config(mode='test', date=date)
dir_results = config.save_dir.replace('weights', 'results')
# Load weights
ep = ''
weights_file = sorted(
    [p for p in os.listdir(config.save_dir) if '_loss' in p and '_epoch{}'.format(ep) in p and 'D' not in p],
    key=lambda x: float(x.rstrip('.pth').split('_loss')[-1])
)[0]
weights_path = os.path.join(config.save_dir, weights_file)
ep = int(weights_path.split('epoch')[-1].split('_')[0])
state_dict = gen_state_dict(weights_path)

model_test = MagNet().cuda()
model_test.load_state_dict(state_dict)
print("Loading weights:", weights_file)

model_test.eval()

for testset, frate in zip(testsets, frates):
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    data_loader = get_gen_ABC(config, mode='test-on-'+testset, dynamic=dynamic)
    print('Number of test image pairs:', data_loader.data_len)
    vid_size = cv2.imread(data_loader.paths[0]).shape[:2][::-1]

    # Test
    for amp in ampfacs:
        frames = []
        data_loader = get_gen_ABC(config, mode='test-on-'+testset, dynamic=dynamic)
        for idx_load in trange(0, data_loader.data_len, data_loader.batch_size):
            batch_A, batch_B = data_loader.gen_test()

            if idx_load == 0:
                frames.append( unit_postprocessing(batch_A[0], vid_size=vid_size) )

            amp_factor = numpy2cuda(np.float32(amp))
            for _ in range(len(batch_A.shape) - len(amp_factor.shape)):
                amp_factor = amp_factor.unsqueeze(-1)
            with torch.no_grad():
                y_hats = model_test(batch_A, batch_B, 0, 0, amp_factor, mode='evaluate')
            for y_hat in y_hats:
                y_hat = unit_postprocessing(y_hat, vid_size=vid_size)
                frames.append(y_hat)

        # Make videos of framesMag
        video_dir = os.path.join(dir_results, testset+'_'+config.date+'_'+amp+'_'+sys.argv[5])
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        out = cv2.VideoWriter(
            os.path.join(video_dir, '{}_{}_{}_{}.avi'.format(testset, config.date, amp, sys.argv[5])),
            cv2.VideoWriter_fourcc(*'DIVX'),
            np.float32(frate), frames[0].shape[-2::-1] # ??? height and width reversed?
        )

        for i, frame in enumerate(frames):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(video_dir, '%06d.png'%(i+1)), frame)
            # cv2.putText(frame, 'amp_factor={}'.format(amp), (7, 37),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            out.write(frame)
        out.release()

        # call([DEFAULT_VIDEO_CONVERTER, '-y', '-f', 'image2', '-r', '30', '-i',
        #       os.path.join(video_dir, '%06d.png'), '-c:v', 'libx264',
        #       os.path.join(video_dir, '{}_{}_{}_{}.mp4'.format(testset, config.date, amp, sys.argv[4]))]
        #     )

        # call([DEFAULT_VIDEO_CONVERTER, '-y', '-f', 'image2', '-r', frate, '-i', os.path.join(video_dir, '%06d.png'), os.path.join(video_dir, '{}_{}_{}_{}.avi'.format(testset, config.date, amp, sys.argv[5]))]
        #     )

        # imageio.mimsave(os.path.join(video_dir, '{}_{}_{}_{}.gif'.format(testset, config.date, amp, sys.argv[4])), frames, 'GIF', fps=FPS)

        # print('{} has been processed.'.format(os.path.join(video_dir, '{}_amp{}.avi'.format(testset, amp))))

