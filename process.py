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
import torchvision.utils as vutils
import material_prediction.src.utils as m_utils

index = 3
torch.cuda.set_device(index)

def bfs(im, pos, tar_val, mask, eps=1e-4):
    queue = deque([pos])
    while queue:
        pp = queue.popleft()
        print(pp)
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

# materialistic
with open('materialistic_checkpoint/args.pkl', 'rb') as f:
    configuration = pickle.load(f)
    margs, mconf = configuration["args"], configuration["conf"]
    # config pre-train ckp files
    margs.batch_size = cfg.train.batch_size
    margs.checkpoint_dir = './materialistic_checkpoint/'
    margs.config = './materialistic/configs/transformer.cfg'
mtModule = model_utils.create_model(mconf, margs)
mtModule.cuda()
mtModule.load_checkpoint(margs.checkpoint_dir, map_location=torch.device(f'cuda:{index}'))
mtModule.eval()

eps = 1e-2
for i, data in enumerate(test_data_loader):
    # for k in data['img']
    #     print(k)

    img = data['img']['im'][0]
    rough = data['img']['roughness'][0]
    metal = data['img']['metallic'][0]
    # mat = data['img']['material']

    ref_pos = data['ref_pos'][0]
    print(ref_pos)
    # ref_pos = (rough.shape[1]//2, rough.shape[2]//2)

    image = data['img']['im'].cuda()
    image = m_utils.correct_exposure(image)
    reference_locations = data['ref_pos'].cuda()
    scores, _, _, _, _, _, _, _, _, _, _, _, _ = mtModule.net(image, reference_locations)

    image = image.cpu()
    image[0, 0, ref_pos[0]-1, ref_pos[1]] = 1.0
    image[0, 0, ref_pos[0]+1, ref_pos[1]] = 1.0
    image[0, 0, ref_pos[0], ref_pos[1]-1] = 1.0
    image[0, 0, ref_pos[0], ref_pos[1]+1] = 1.0
    scores = scores.cpu().detach().numpy()
    scores = cv2.cvtColor(scores[0], cv2.COLOR_GRAY2BGR)
    scores = torch.from_numpy(scores).permute(2, 0, 1)
    imgscores = torch.cat((image[0], scores), dim=1)
    vutils.save_image(imgscores, os.path.join(args.results_dir, f"{data['img']['scene'][0]}_imgscore.png"), nrow=imgscores.shape[0])

    scores = scores.numpy()
    print(scores.shape)


    rough = rough.numpy().transpose(1, 2, 0)
    # rough = (rough * 255).astype(np.uint8)

    k = np.ones((5, 5), np.float32)
    open_rough = cv2.morphologyEx(rough,cv2.MORPH_OPEN,k)
    close_rough = cv2.morphologyEx(rough,cv2.MORPH_CLOSE,k)

    print(close_rough[ref_pos[0], ref_pos[1]])
    if close_rough[ref_pos[0], ref_pos[1]] <= eps:
        _, mask1 = cv2.threshold(close_rough, close_rough[ref_pos[0], ref_pos[1]] + eps, 1.0, cv2.THRESH_BINARY_INV)
    else:
        _, mask1 = cv2.threshold(close_rough, close_rough[ref_pos[0], ref_pos[1]] + eps, 1.0, cv2.THRESH_TOZERO_INV)
        _, mask1 = cv2.threshold(mask1, mask1[ref_pos[0], ref_pos[1]] - eps, 1.0, cv2.THRESH_BINARY)
    if open_rough[ref_pos[0], ref_pos[1]] <= eps:
        _, mask11 = cv2.threshold(open_rough, open_rough[ref_pos[0], ref_pos[1]] + eps, 1.0, cv2.THRESH_BINARY_INV)
    else:
        _, mask11 = cv2.threshold(open_rough, open_rough[ref_pos[0], ref_pos[1]] + eps, 1.0, cv2.THRESH_TOZERO_INV)
        _, mask11 = cv2.threshold(mask11, mask11[ref_pos[0], ref_pos[1]] - eps, 1.0, cv2.THRESH_BINARY)
    if rough[ref_pos[0], ref_pos[1], 0] <= eps:
        _, mask12 = cv2.threshold(rough, rough[ref_pos[0], ref_pos[1], 0] + eps, 1.0, cv2.THRESH_BINARY_INV)
    else:
        _, mask12 = cv2.threshold(rough, rough[ref_pos[0], ref_pos[1], 0] + eps, 1.0, cv2.THRESH_TOZERO_INV)
        _, mask12 = cv2.threshold(mask12, mask12[ref_pos[0], ref_pos[1]] - eps, 1.0, cv2.THRESH_BINARY)

    open_rough[ref_pos[0]-1, ref_pos[1]] = 1.0
    open_rough[ref_pos[0]+1, ref_pos[1]] = 1.0
    open_rough[ref_pos[0], ref_pos[1]-1] = 1.0
    open_rough[ref_pos[0], ref_pos[1]+1] = 1.0
    close_rough[ref_pos[0]-1, ref_pos[1]] = 1.0
    close_rough[ref_pos[0]+1, ref_pos[1]] = 1.0
    close_rough[ref_pos[0], ref_pos[1]-1] = 1.0
    close_rough[ref_pos[0], ref_pos[1]+1] = 1.0
    row1 = np.hstack((open_rough, close_rough, rough[:, :, 0]))
    row2 = np.hstack((mask11, mask1, mask12))
    write_img = np.vstack((row1,row2))
    write_img = (write_img*255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.results_dir, f"{data['img']['scene'][0]}_eroded_dilated_rough.png"), write_img)

    metal = metal.numpy().transpose(1, 2, 0)
    # metal = (metal * 255).astype(np.uint8)

    open_metal = cv2.morphologyEx(metal,cv2.MORPH_OPEN,k)
    close_metal = cv2.morphologyEx(metal,cv2.MORPH_CLOSE,k)

    print(open_metal[ref_pos[0], ref_pos[1]])
    if open_metal[ref_pos[0], ref_pos[1]] <= eps:
        _, mask2 = cv2.threshold(open_metal, open_metal[ref_pos[0], ref_pos[1]] + eps, 1.0, cv2.THRESH_BINARY_INV)
    else:
        _, mask2 = cv2.threshold(open_metal, open_metal[ref_pos[0], ref_pos[1]] + eps, 1.0, cv2.THRESH_TOZERO_INV)
        _, mask2 = cv2.threshold(mask2, mask2[ref_pos[0], ref_pos[1]] - eps, 1.0, cv2.THRESH_BINARY)
    if close_metal[ref_pos[0], ref_pos[1]] <= eps:
        _, mask22 = cv2.threshold(close_metal, close_metal[ref_pos[0], ref_pos[1]] + eps, 1.0, cv2.THRESH_BINARY_INV)
    else:
        _, mask22 = cv2.threshold(close_metal, close_metal[ref_pos[0], ref_pos[1]] + eps, 1.0, cv2.THRESH_TOZERO_INV)
        _, mask22 = cv2.threshold(mask22, mask22[ref_pos[0], ref_pos[1]] - eps, 1.0, cv2.THRESH_BINARY)
    if metal[ref_pos[0], ref_pos[1], 0] <= eps:
        _, mask23 = cv2.threshold(metal, metal[ref_pos[0], ref_pos[1], 0] + eps, 1.0, cv2.THRESH_BINARY_INV)
    else:
        _, mask23 = cv2.threshold(metal, metal[ref_pos[0], ref_pos[1], 0] + eps, 1.0, cv2.THRESH_TOZERO_INV)
        _, mask23 = cv2.threshold(mask23, mask23[ref_pos[0], ref_pos[1]] - eps, 1.0, cv2.THRESH_BINARY)

    open_metal[ref_pos[0]-1, ref_pos[1]] = 1.0
    open_metal[ref_pos[0]+1, ref_pos[1]] = 1.0
    open_metal[ref_pos[0], ref_pos[1]-1] = 1.0
    open_metal[ref_pos[0], ref_pos[1]+1] = 1.0
    close_metal[ref_pos[0]-1, ref_pos[1]] = 1.0
    close_metal[ref_pos[0]+1, ref_pos[1]] = 1.0
    close_metal[ref_pos[0], ref_pos[1]-1] = 1.0
    close_metal[ref_pos[0], ref_pos[1]+1] = 1.0
    row1 = np.hstack((open_metal, close_metal, metal[:, :, 0]))
    row2 = np.hstack((mask2, mask22, mask23))
    write_img = np.vstack((row1,row2))
    write_img = (write_img*255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.results_dir, f"{data['img']['scene'][0]}_eroded_dilated_metal.png"), write_img)

    mask = np.logical_and(mask1, mask2).astype(np.float32)

    mrgImg1 = np.zeros((rough.shape[1], rough.shape[1], 3))
    mrgImg1[:, :, 0] = mask1
    mrgImg1[:, :, 1] = mask2
    # mrgImg1[:, :, 2] = 0

    mrgImg2 = np.zeros((rough.shape[1], rough.shape[1], 3))
    mrgImg2[:, :, 0] = close_rough
    mrgImg2[:, :, 1] = open_metal
    mrgImg2[ref_pos[0]-1, ref_pos[1], 2] = 1.0
    mrgImg2[ref_pos[0]+1, ref_pos[1], 2] = 1.0
    mrgImg2[ref_pos[0], ref_pos[1]+1, 2] = 1.0
    mrgImg2[ref_pos[0], ref_pos[1]-1, 2] = 1.0

    img = img.clamp(0, 1) ** (1 / 2.2)
    # img = img.numpy().transpose(1, 2, 0)
    # img[ref_pos[0]-1, ref_pos[1], :] = 1.0, 0, 0
    # img[ref_pos[0]+1, ref_pos[1], :] = 1.0, 0, 0
    # img[ref_pos[0], ref_pos[1]+1, :] = 1.0, 0, 0
    # img[ref_pos[0], ref_pos[1]-1, :] = 1.0, 0, 0
    img[:, ref_pos[0]-1, ref_pos[1]] = torch.tensor([1.0, 0, 0])
    img[:, ref_pos[0]+1, ref_pos[1]] = torch.tensor([1.0, 0, 0])
    img[:, ref_pos[0], ref_pos[1]+1] = torch.tensor([1.0, 0, 0])
    img[:, ref_pos[0], ref_pos[1]-1] = torch.tensor([1.0, 0, 0])

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # newImg = np.hstack((mask_color, mrgImg1, mrgImg2, img))
    # newImg = (newImg*255).astype(np.uint8)
    # cv2.imwrite(os.path.join(args.results_dir, f"{data['img']['scene'][0]}.png"), newImg)
    # cv2.imwrite(os.path.join(args.results_dir, f"{data['img']['scene'][0]}_mask_color.png"), mask_color)
    # cv2.imwrite(os.path.join(args.results_dir, f"{data['img']['scene'][0]}_mrgImg1.png"), mrgImg1)
    # cv2.imwrite(os.path.join(args.results_dir, f"{data['img']['scene'][0]}_mrgImg2.png"), mrgImg2)
    # cv2.imwrite(os.path.join(args.results_dir, f"{data['img']['scene'][0]}_img.png"), img)
    mask_color = torch.tensor(mask_color.transpose(2, 0, 1))
    mrgImg1 = torch.tensor(mrgImg1.transpose(2, 0, 1))
    mrgImg2 = torch.tensor(mrgImg2.transpose(2, 0, 1))
    newImg = torch.cat((mask_color, mrgImg1), dim=2)
    newImg = torch.cat((newImg, mrgImg2), dim=2)
    newImg = torch.cat((newImg, img), dim=2)

    vutils.save_image(newImg, os.path.join(args.results_dir, f"{data['img']['scene'][0]}_img.png"), nrow=img.shape[0])
    vutils.save_image(img, os.path.join(args.results_dir, f"{data['img']['scene'][0]}_img1.png"), nrow=img.shape[0])

    # print(1)

    # msk1 = getMask(rough, ref_pos, 1)
    # msk2 = getMask(metal, ref_pos, 1)
    # mrgMsk = np.logical_and(msk1, msk2)
    #
    # if not os.path.exists(os.path.join(args.results_dir, 'mask')):
    #     os.mkdir(os.path.join(args.results_dir, 'mask'))
    #
    # mrgMsk = mrgMsk.reshape(rough.shape[1], rough.shape[2], 1)
    # msk1 = msk1.transpose(1, 2, 0)
    # msk2 = msk2.transpose(1, 2, 0)
    #
    # combined_image = np.zeros((rough.shape[1], rough.shape[1], 3), dtype=float)
    # combined_image[:, :, 0] = msk1[:, :, 0]
    # combined_image[:, :, 1] = msk2[:, :, 0]
    # combined_image[:, :, 2] = mrgMsk[:, :, 0]
    #
    # combined_image[ref_pos[0], ref_pos[1], :] = 1,0,0
    # mrgMsk[ref_pos[0], ref_pos[1], 0] = 0
    #
    # combined_image = (combined_image * 255).astype(np.uint8)
    # # mrgMsk = (mrgMsk * 255).astype(np.uint8)
    # mrgMsk = mrgMsk.astype(np.float32)
    #
    # cv2.imwrite(os.path.join(args.results_dir, 'mask', f"{data['img']['scene'][0]}_{i}_color.png"), combined_image)
    # cv2.imwrite(os.path.join(cfg.dataset.dataRoot, data['img']['dir'][0], f"{data['img']['scene'][0]}_cluster.exr"), mrgMsk)
    #
    # rough = rough.numpy().transpose(1, 2, 0)
    # metal = metal.numpy().transpose(1, 2, 0)
    #
    # combined_image = np.zeros((rough.shape[1], rough.shape[1], 3), dtype=float)
    # combined_image[:, :, 0] = rough[:, :, 0]
    # combined_image[:, :, 1] = metal[:, :, 0]
    #
    # combined_image[ref_pos[0], ref_pos[1], :] = 0, 0, 1
    # combined_image = (combined_image * 255).astype(np.uint8)
    #
    # cv2.imwrite(os.path.join(args.results_dir, 'mask', f"{data['img']['scene'][0]}_{i}_marked.png"), combined_image)

    if i > 5:
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
