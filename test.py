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
import src.utils.args as utils_args
import src.models.top as top

import materialistic.src.models.model_utils_pl as model_utils
import materialistic.src.utils.utils as materialistic_utils

from torchinfo import summary


#        args & config        #
args, cfg = utils_args.parse_args()
num_gpus = len(cfg.experiment.device_ids)


#       data      #
# Function to change the random seed for all workers
def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)


test_set = data_interface.MPDataset(cfg.dataset, mphase='TEST')
test_data_loader = torch.utils.data.DataLoader(test_set,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1,
                                                pin_memory=True,
                                                worker_init_fn=worker_init_fn)


##############################
#       model Prepare
##############################
# Download Pretrain Model
# open
with open('materialistic_checkpoint/args.pkl', 'rb') as f:
    configuration = pickle.load(f)
    margs, mconf = configuration["args"], configuration["conf"]

    margs.batch_size = cfg.train.batch_size
    margs.checkpoint_dir = './materialistic_checkpoint/'
    margs.config = './materialistic/configs/transformer.cfg'
    # margs.data_dir =
    # print(margs)
# load checkpoint
checkpoint_files = os.listdir(args.checkpoint_dir)
checkpoint_files = [f for f in checkpoint_files if 'ckpt' in f]
checkpoint_files.sort()
# create model
net = top.TopModule(mconf, margs, cfg)
net = net.cuda()
net.load_checkpoint(os.path.join(args.checkpoint_dir, checkpoint_files[-1]))
net.eval()
# print(net)

#######################################################
for i, data in enumerate(test_data_loader):
    image = data['img']['im'].cuda()
    reference_locations = data['ref_pos'].cuda()
    # print(net(image, reference_locations))
    with torch.no_grad():
        predictions, scores, albedoPred, roughPred, metalPred = net.net(image, reference_locations)
    print(predictions.shape, scores.shape)
    # print(scores)
    predictions = predictions.cpu()
    scores = scores.cpu()

    if not os.path.exists(os.path.join(args.results_dir, 'albedoPred')):
        os.mkdir(os.path.join(args.results_dir, 'albedoPred'))
    if not os.path.exists(os.path.join(args.results_dir, 'roughPred')):
        os.mkdir(os.path.join(args.results_dir, 'roughPred'))
    if not os.path.exists(os.path.join(args.results_dir, 'metalPred')):
        os.mkdir(os.path.join(args.results_dir, 'metalPred'))

    cv2.imwrite(os.path.join(args.results_dir, 'albedoPred', f"{i}.png"),
                materialistic_utils.convert_to_opencv_image(albedoPred[0].numpy()))
    cv2.imwrite(os.path.join(args.results_dir, 'roughPred', f"{i}.png"),
                materialistic_utils.convert_to_opencv_image(roughPred[0].numpy()))
    cv2.imwrite(os.path.join(args.results_dir, 'metalPred', f"{i}.png"),
                materialistic_utils.convert_to_opencv_image(metalPred[0].numpy()))
    if i > 1:
        break

