import os
import sys
from typing import Any

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

# materialistic
with open('materialistic_checkpoint/args.pkl', 'rb') as f:
    configuration = pickle.load(f)
    margs, mconf = configuration["args"], configuration["conf"]
    # config pre-train ckp files
    margs.batch_size = cfg.train.batch_size
    margs.checkpoint_dir = './materialistic_checkpoint/'
    margs.config = './materialistic/configs/transformer.cfg'
# mtModule = Regression_Materialistic(mconf, margs)
# mtModule.cuda()
# mtModule.load_checkpoint(margs.checkpoint_dir, map_location=torch.device(f'cuda:{index}'))
# mtModule.eval()


# todo: 1. load materialistic dataset; 2. check id; 3. finish regression module