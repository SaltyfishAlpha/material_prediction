import os
import sys
from typing import Any

from tqdm import tqdm

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
import src.utils.args as utils_args
import src.models.top as top_module

import materialistic.src.models.model_utils_pl as model_utils
import materialistic.src.utils.utils as materialistic_utils

from collections import deque
import torchvision.utils as vutils
import material_prediction.src.utils as m_utils
from collections import Counter
import random
from random import randint as randi

random.seed(0)

index = 3
torch.cuda.set_device(index)


#       model for material ID regression    #
class Regression_Materialistic(pl.LightningModule):
    def __init__(self, mconf, margs):
        self.mtModule = model_utils.create_model(mconf, margs)

    def forward(self, x, ref_pos):
        return self.mtModule.net(x, ref_pos)


#        args & config        #
args, cfg = utils_args.parse_args()
num_gpus = len(cfg.experiment.device_ids)

# #       data      #
# # Function to change the random seed for all workers
# def worker_init_fn(worker_id):
#     np.random.seed(worker_id)
#     torch.manual_seed(worker_id)
#
# #       data set        #
# test_set = data_interface.MPDataset(cfg.dataset, mphase='ALL')
# test_data_loader = torch.utils.data.DataLoader(test_set,
#                                                batch_size=1,
#                                                shuffle=False,
#                                                num_workers=16,
#                                                pin_memory=True,
#                                                worker_init_fn=worker_init_fn)
#
# materialistic
# with open('materialistic_checkpoint/args.pkl', 'rb') as f:
#     configuration = pickle.load(f)
#     margs, mconf = configuration["args"], configuration["conf"]
#     # config pre-train ckp files
#     margs.batch_size = cfg.train.batch_size
#     margs.checkpoint_dir = './materialistic_checkpoint/'
#     margs.config = './materialistic/configs/transformer.cfg'
# mtModule = Regression_Materialistic(mconf, margs)
# mtModule.cuda()
# mtModule.load_checkpoint(margs.checkpoint_dir, map_location=torch.device(f'cuda:{index}'))
# mtModule.eval()


# todo: 1. load materialistic dataset; 2. check id; 3. finish regression module

# step. 1
DATA_ROOT = "/home/disk1/Dataset/materialistic_synthetic_dataset/"
TRAIN_PATH = os.path.join(DATA_ROOT, 'train')
VAL_PATH = os.path.join(DATA_ROOT, 'val')
TEST_PATH = os.path.join(DATA_ROOT, 'test')


def _get_path_with_name(arr, name):
    return [f for f in arr if name in f][0]


if not os.path.exists(os.path.join(args.results_dir, 'matID')):
    os.mkdir(os.path.join(args.results_dir, 'matID'))


# sample: train/*(dir)
def matID_statistic(path):
    i = 0
    eps = 1e-4
    counter = Counter()
    samples = sorted([os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))])
    pbar = tqdm(samples)
    for sample in pbar:
        pbar.set_description('counter_size: %d\t' % len(counter))

        files = [os.path.join(sample, x) for x in os.listdir(sample)]
        material_label_path = _get_path_with_name(files, "segmentation")

        im = np.flip(cv2.imread(material_label_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), -1)
        mat_label = torch.tensor(im[:, :, 0].copy()).int()

        unique_values, counts = torch.unique(mat_label, return_counts=True)
        # counts = counts / (mat_label.shape[0] * mat_label.shape[1])

        counter.update(dict(zip(unique_values.tolist(), torch.ones((len(counts))).int().tolist())))

        # assert 1 - eps <= counts.sum() <= 1 + eps, "sum unequal to 1."

        i += 1
        if i > 20:
            break

    # sorted_counter = sorted(counter.items(), key=lambda x: x[0], reverse=True)
    #
    # all_unique_values = torch.tensor(list(sorted_counter.keys()))
    # all_counts = torch.tensor(list(sorted_counter.values()))
    #
    # print(all_counts.size())

    return counter


val_counter = matID_statistic(VAL_PATH)
# val_sorted_counter = sorted(val_counter.items(), key=lambda x: x[0], reverse=True)

val_color = [k for k, v in val_counter.items() if v > 1]

color_map = dict([ (k, (randi(0, 255), randi(0, 255), randi(0, 255))) for k, v in val_counter.items()])

counter = Counter()
samples = sorted([os.path.join(VAL_PATH, x) for x in os.listdir(VAL_PATH) if os.path.isdir(os.path.join(VAL_PATH, x))])
i = 0
mrgImg = []
for sample in samples:
    i += 1
    if i % 10 == 0:
        mrgAll = mrgImg[0]
        for img in mrgImg[1:]:
            mrgAll = np.vstack((mrgAll, img))
        cv2.imwrite(os.path.join(args.results_dir, 'matID', f"{i}.png"), mrgAll)
        mrgImg = []

    if i > 20:
        break

    files = [os.path.join(sample, x) for x in os.listdir(sample)]
    image_path = _get_path_with_name(files, "Image")
    material_label_path = _get_path_with_name(files, "segmentation")

    im1 = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    image = torch.tensor(im1.copy())
    image = m_utils.correct_exposure(image)

    # tmp = (image.numpy()*255).astype(np.uint8)
    # cv2.imwrite(os.path.join(args.results_dir, 'matID', f"{i}_image.png"), tmp)

    im2 = cv2.imread(material_label_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    mat_label = torch.tensor(im2[:, :, 0].copy()).int()

    unique_values, counts = torch.unique(mat_label, return_counts=True)

    same_id = [x for x in unique_values.tolist() if val_counter[x]>1]
    if len(same_id) == 0:
        print(sample)
        continue

    # write to file
    mapped_image = np.zeros((mat_label.shape[0], mat_label.shape[1], 3), dtype=np.uint8)
    for _id in unique_values.tolist():
        mapped_image[mat_label == _id] = color_map[_id]

    same_image = np.zeros((mat_label.shape[0], mat_label.shape[1], 3), dtype=np.uint8)
    for _id in same_id:
        same_image[mat_label == _id] = (255, 0, 0)

    alpha = 0.5
    image = (image.numpy() * 255).astype(np.uint8)
    mapped_image1 = cv2.addWeighted(image, 1, mapped_image, alpha, 0)
    same_image = cv2.addWeighted(image, 1, same_image, alpha, 0)
    mrgImg.append(np.hstack((image, mapped_image1, mapped_image, same_image)))

def findDft(id): # id: id appeared more than once
    i=0
    mrgImg = []
    for sample in samples:
        i += 1
        if i > 20:
            break

        files = [os.path.join(sample, x) for x in os.listdir(sample)]
        image_path = _get_path_with_name(files, "Image")
        material_label_path = _get_path_with_name(files, "segmentation")

        im1 = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image = torch.tensor(im1.copy())
        image = m_utils.correct_exposure(image)

        # tmp = (image.numpy()*255).astype(np.uint8)
        # cv2.imwrite(os.path.join(args.results_dir, 'matID', f"{i}_image.png"), tmp)

        im2 = cv2.imread(material_label_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        mat_label = torch.tensor(im2[:, :, 0].copy()).int()

        unique_values, counts = torch.unique(mat_label, return_counts=True)


        if not id in unique_values:
            print(id, unique_values)
            continue

        # write to file
        mapped_image = np.zeros((mat_label.shape[0], mat_label.shape[1], 3), dtype=np.uint8)
        mapped_image[mat_label == id] = color_map[id]

        same_image = np.zeros((mat_label.shape[0], mat_label.shape[1], 3), dtype=np.uint8)
        same_image[mat_label == id] = (255, 0, 0)

        alpha = 0.5
        image = (image.numpy() * 255).astype(np.uint8)
        mapped_image1 = cv2.addWeighted(image, 1, mapped_image, alpha, 0)
        same_image = cv2.addWeighted(image, 1, same_image, alpha, 0)
        mrgImg.append(np.hstack((image, mapped_image1, mapped_image, same_image)))

    if len(mrgImg) != 0:
        mrgAll = mrgImg[0]
        for img in mrgImg[1:]:
            mrgAll = np.vstack((mrgAll, img))
        cv2.imwrite(os.path.join(args.results_dir, 'matID', f"{id}.png"), mrgAll)
    mrgImg = []

for _id in val_color[1:]:
    findDft(_id)

# with open('val_counter_output.txt', 'w', encoding='utf-8') as f:
#     # 遍历Counter对象，写入键和值
#     for key, value in val_sorted_counter:
#         f.write(f'{key}: {value}\n')

# test_counter = matID_statistic(TEST_PATH)
# test_sorted_counter = sorted(test_counter.items(), key=lambda x: x[0], reverse=True)
# with open('test_counter_output.txt', 'w', encoding='utf-8') as f:
#     # 遍历Counter对象，写入键和值
#     for key, value in test_sorted_counter:
#         f.write(f'{key}: {value}\n')
#
# train_counter = matID_statistic(TRAIN_PATH)
# train_sorted_counter = sorted(train_counter.items(), key=lambda x: x[0], reverse=True)
# with open('train_counter_output.txt', 'w', encoding='utf-8') as f:
#     # 遍历Counter对象，写入键和值
#     for key, value in train_sorted_counter:
#         f.write(f'{key}: {value}\n')
#
# counters = train_counter + val_counter + test_counter
# sorted_counter1 = sorted(counters.items(), key=lambda x: x[1], reverse=True)
# print(len(sorted_counter1))
# with open('counter_output_1.txt', 'w', encoding='utf-8') as f:
#     # 遍历Counter对象，写入键和值
#     for key, value in sorted_counter1:
#         f.write(f'{key}: {value}\n')
#
# sorted_counter0 = sorted(counters.items(), key=lambda x: x[0], reverse=True)
# # print(len(sorted_counter))
# with open('counter_output_0.txt', 'w', encoding='utf-8') as f:
#     # 遍历Counter对象，写入键和值
#     for key, value in sorted_counter0:
#         f.write(f'{key}: {value}\n')


# sample examples
