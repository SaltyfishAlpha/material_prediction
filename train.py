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


# os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'

# torch.cuda.set_device(2)

###############################
#       fix random seed
###############################
# def set_random_seed(seed: int) -> None:
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
#
# seed = 114514
# set_random_seed(seed)

#        args & config        #
args, cfg = utils_args.parse_args()
num_gpus = len(cfg.experiment.device_ids)

# checkpoint dir
if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

# Create a tensorboard writer
print("Logging to: ", cfg.experiment.path_logs)
logger = pl_loggers.TensorBoardLogger(save_dir=cfg.experiment.path_logs, name=cfg.experiment.id, version=args.version)


#       data      #
# Function to change the random seed for all workers
def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)


# if train :
train_set = data_interface.MPDataset(cfg.dataset, small_size=args.small_size)
val_set = data_interface.MPDataset(cfg.dataset, mphase='VAL')
# test_set = data_interface.MPDataset(cfg.dataset, mphase='TEST')
######
# for k in train_set[-1]['img']:
#     # print(k)
#     if k != 'scene':
#         cv2.imwrite(os.path.join(args.results_dir, f"{k}.png"),
#                     materialistic_utils.convert_to_opencv_image(train_set[-1]['img'][k].numpy()))
######
train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=cfg.train.batch_size,
                                                num_workers=64,
                                                pin_memory=True,
                                                shuffle=False,
                                                worker_init_fn=worker_init_fn,
                                                drop_last=True)
val_data_loader = torch.utils.data.DataLoader(dataset=val_set,
                                              batch_size=cfg.val.batch_size,
                                              num_workers=16,
                                              pin_memory=True,
                                              shuffle=False,
                                              worker_init_fn=worker_init_fn,
                                              drop_last=True)
# f


# test_data_loader = torch.utils.data.DataLoader(test_set,
#                                                 batch_size=1,
#                                                 shuffle=False,
#                                                 num_workers=1,
#                                                 pin_memory=True,
#                                                 worker_init_fn=worker_init_fn)

######################################################
#   CheckPoint
######################################################
# checkpoint callback to save model to checkpoint_dir
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.checkpoint_dir,
                                                   filename="model-{epoch:02d}",
                                                   save_top_k=-1,
                                                   every_n_train_steps=1000)

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
# create model
net = top.TopModule(mconf, margs, cfg, use_swin=args.swin)
# net = net.cuda()
# net.eval()
# print(net)

# summary(net.net, input_size=[(1, 3, 512, 512), (1, 2)])


######################################################
#   Train
######################################################
# Write pytorch lightning trainer
if args.resume:
    checkpoint_files = os.listdir(args.checkpoint_dir)
    checkpoint_files = [f for f in checkpoint_files if 'ckpt' in f]
    checkpoint_files.sort()
    trainer = pl.Trainer(accelerator="gpu",
                         # devices=len(cfg.experiment.device_ids),
                         devices=[2],
                         val_check_interval=args.print_every,  # cfg.val.interval,
                         limit_val_batches=1.0 if not args.small_size else 0,
                         # strategy=DDPStrategy(find_unused_parameters=True),
                         precision=args.precision,
                         max_epochs=cfg.train.epochs,
                         logger=logger,
                         # log_every_n_steps=args.print_every,  # cfg.val.interval,
                         callbacks=[checkpoint_callback],
                         check_val_every_n_epoch=None,  # cfg.val.interval,
                         gradient_clip_val=0.5,
                         gradient_clip_algorithm="value",
                         resume_from_checkpoint=os.path.join(args.checkpoint_dir, checkpoint_files[-1]))
else:
    trainer = pl.Trainer(accelerator="gpu",
                         # devices=len(cfg.experiment.device_ids),
                         devices=[2],
                         val_check_interval=args.print_every,  # cfg.val.interval,
                         limit_val_batches=1.0 if not args.small_size else 0,
                         # strategy=DDPStrategy(find_unused_parameters=True),
                         precision=args.precision,
                         max_epochs=cfg.train.epochs,
                         logger=logger,
                         # log_every_n_steps=args.print_every,  # cfg.val.interval,
                         check_val_every_n_epoch=None,  # cfg.val.interval,
                         gradient_clip_val=0.5,
                         gradient_clip_algorithm="value",
                         callbacks=[checkpoint_callback])

trainer.fit(net, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

###############################
#   For Test
###############################
# tnet = top.TestMod(mconf, margs, cfg)
# tnet = tnet.cuda()
# tnet.eval()
# print(tnet)
# summary(tnet.net, input_size=[(1, 3, 512, 512), (1, 2)])

# trainer.fit(tnet, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

#######################################################
# for i, data in enumerate(train_data_loader):
#     image = data['img']['im'].cuda()
#     reference_locations = data['ref_pos'].cuda()
#     # print(net(image, reference_locations))
#     with torch.no_grad():
#         predictions, scores = net(image, reference_locations)
#     print(predictions.shape, scores.shape)
#     # print(scores)
#     predictions = predictions.cpu()
#     scores = scores.cpu()
#     cv2.imwrite(os.path.join(args.results_dir, f"_result_{i}.png"),
#                 materialistic_utils.convert_to_opencv_image(predictions[0].numpy()))
#     cv2.imwrite(os.path.join(args.results_dir, f"_score_{i}.png"),
#                 materialistic_utils.convert_to_opencv_image(scores.numpy()))
#     if i > 1:
#         break
