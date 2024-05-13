import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
sys.path.append(project_root)

import pickle
import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy
from skimage import color as color_module

import src.data.data_interface as data_interface
import src.utils.args_test as utils_args
import src.models.top as top_module

import materialistic.src.models.model_utils_pl as model_utils
import materialistic.src.utils.utils as materialistic_utils

from collections import deque


def bfs(im, pos, tar_val, mask, eps=1e-4):
    queue = deque([pos])
    while queue:
        pp = queue.popleft()
        if pp[1] < 0 or pp[1] >= im.shape[1] or pp[2] < 0 or pp[2] >= im.shape[2]:
            continue
        if mask[pp] or not (tar_val - eps <= im[pp] <= tar_val + eps):
            continue
        mask[pp] = True
        queue.append((pp[0], pp[1] + 1, pp[2]))
        queue.append((pp[0], pp[1] - 1, pp[2]))
        queue.append((pp[0], pp[1], pp[2] + 1))
        queue.append((pp[0], pp[1], pp[2] - 1))


def getMask(image, pos, channel):
    mask = np.zeros_like(image)
    for i in range(channel):
        bfs(image, (i, pos[0], pos[1]), image[i, pos[0], pos[1]], mask)
    return mask


#        args & config        #
args, cfg = utils_args.parse_args()
num_gpus = len(cfg.experiment.device_ids)


#       data      #
# Function to change the random seed for all workers
def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)


#       data set        #
test_set = data_interface.MPDataset(cfg.dataset, mphase='ALL')
test_data_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=16,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)

for i, data in enumerate(test_data_loader):
    # for k in data['img']:
    #     print(k)

    albedo = data['img']['albedo'][0]
    rough = data['img']['roughness'][0]
    metal = data['img']['metallic'][0]
    # mat = data['img']['material']

    ref_pos = data['ref_pos'][0]
    print(ref_pos)
    # ref_pos = (rough.shape[1]//2, rough.shape[2]//2)

    msk1 = getMask(rough, ref_pos, 1)
    msk2 = getMask(metal, ref_pos, 1)
    mrgMsk = np.logical_and(msk1, msk2)

    if not os.path.exists(os.path.join(args.results_dir, 'mask')):
        os.mkdir(os.path.join(args.results_dir, 'mask'))
    if not os.path.exists(os.path.join(args.results_dir, 'marked')):
        os.mkdir(os.path.join(args.results_dir, 'marked'))

    mrgMsk = mrgMsk.reshape(rough.shape[1], rough.shape[2], 1)
    msk1 = msk1.transpose(1, 2, 0)
    msk2 = msk2.transpose(1, 2, 0)

    combined_image = np.zeros((rough.shape[1], rough.shape[1], 3), dtype=float)
    combined_image[:, :, 0] = msk1[:, :, 0]
    combined_image[:, :, 1] = msk2[:, :, 0]
    combined_image[:, :, 2] = mrgMsk[:, :, 0]

    combined_image[ref_pos[0], ref_pos[1], :] = 1,0,0
    mrgMsk[ref_pos[0], ref_pos[1], 0] = 0

    combined_image = (combined_image * 255).astype(np.uint8)
    # mrgMsk = (mrgMsk * 255).astype(np.uint8)
    mrgMsk = mrgMsk.astype(np.float32)

    cv2.imwrite(os.path.join(args.results_dir, 'mask', f"{i}_color.png"), combined_image)
    cv2.imwrite(os.path.join(args.results_dir, 'mask', f"{i}_mask.exr"), mrgMsk)

    rough = rough.numpy().transpose(1, 2, 0)
    rough[ref_pos[0], ref_pos[1], 0] = 1
    rough = (rough * 255).astype(np.uint8)

    metal = metal.numpy().transpose(1, 2, 0)
    metal[ref_pos[0], ref_pos[1], 0] = 1
    metal = (metal * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(args.results_dir, 'marked', f"{i}_rough.png"), rough)
    cv2.imwrite(os.path.join(args.results_dir, 'marked', f"{i}_metal.png"), metal)

    if i > 0:
        break


## test ##
# # with cv2.imread('../test.png', -1) as im:
# im = cv2.imread('test.exr', -1)
# if im is None:
#     raise ValueError(f"ERROR: Open failed!")
# # im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation=interpolation)
# im = np.transpose(im, [2, 0, 1])
# # im = im[::-1, :, :].copy()
#
# ref_pos = (im.shape[1]//2, im.shape[2]//2)
# msk = getMask(im, ref_pos, 3)
# mrgMsk = np.logical_and(np.logical_and(msk[0], msk[1]), msk[2])
#
# mrgMsk = mrgMsk.reshape(im.shape[1], im.shape[2], 1)
# mrgMsk[ref_pos[0], ref_pos[1], 0] = 0
# mrgMsk = (mrgMsk * 255).astype(np.uint8)
# cv2.imwrite('test_mask.png', mrgMsk)
#
# msk = msk.transpose(1, 2, 0)
# msk[ref_pos[0], ref_pos[1], :] = 0, 0, 0
# msk = (msk * 255).astype(np.uint8)
# cv2.imwrite('test_colorMask.png', msk)
#
# im = im.transpose(1, 2, 0)
# im[ref_pos[0], ref_pos[1], :] = 0, 0, 0
# im = (im * 255).astype(np.uint8)
# cv2.imwrite('test_marked.png', im)
