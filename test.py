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

from torchinfo import summary

# torch.cuda.set_device(1)

#        args & config        #
args, cfg = utils_args.parse_args()
num_gpus = len(cfg.experiment.device_ids)


###############################
# Create a tensorboard writer
print("Logging to: ", cfg.experiment.path_logs)
logger = pl_loggers.TensorBoardLogger(save_dir=cfg.experiment.path_logs, name=cfg.experiment.id, version=args.version)

# checkpoint callback to save model to checkpoint_dir
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.checkpoint_dir,
                                                   filename="model-{epoch:02d}",
                                                   save_top_k=-1,
                                                   every_n_train_steps=args.print_every,
                                                   save_on_train_epoch_end=True,
                                                   )
###############################


#       data      #
# Function to change the random seed for all workers
def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)


test_set = data_interface.MPDataset(cfg.dataset, mphase='TEST')
test_data_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=cfg.val.batch_size,
                                               shuffle=False,
                                               num_workers=16,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)

##############################
#       Model Prepare
##############################
# Download Pretrain Model
with open('materialistic_checkpoint/args.pkl', 'rb') as f:
    configuration = pickle.load(f)
    margs, mconf = configuration["args"], configuration["conf"]
    # config pre-train ckp files
    margs.batch_size = cfg.train.batch_size
    margs.checkpoint_dir = './materialistic_checkpoint/'
    margs.config = './materialistic/configs/transformer.cfg'
    # margs.data_dir =
    # print(margs)
# create model
net = top_module.TopModule(mconf, margs, cfg, use_swin=args.swin, load_location=torch.device('cuda:1'), use_prec=args.prec)

# load checkpoint
checkpoint_files = os.listdir(args.checkpoint_dir)
checkpoint_files = [f for f in checkpoint_files if 'ckpt' in f]
checkpoint_files.sort()
print()

#######################################
#           Evaluation
#######################################
if args.metrics:
    trainer = pl.Trainer(accelerator="gpu",
                         devices=[1],
                         precision=args.precision,
                         max_epochs=cfg.train.epochs,
                         logger=logger,
                         callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar(leave=True)],
                         )
    # mkdir results
    os.makedirs(os.path.join(cfg.experiment.path_logs, "test/albedo"), exist_ok=True)
    os.makedirs(os.path.join(cfg.experiment.path_logs, "test/roughness"), exist_ok=True)
    os.makedirs(os.path.join(cfg.experiment.path_logs, "test/metallic"), exist_ok=True)
    trainer.test(net, ckpt_path=os.path.join(args.checkpoint_dir, checkpoint_files[-1]), dataloaders=test_data_loader)

else: # output result
    net = net
    net.load_from_checkpoint(os.path.join(args.checkpoint_dir, checkpoint_files[-1]), mconf=mconf, margs=margs, cfg=cfg,
                             use_swin=args.swin, load_location=torch.device('cuda:1'), use_prec=args.prec)
    net.eval()
    print(net)
    # output pic to results dir
    for i, data in enumerate(test_data_loader):
        # for k in data['img']:
        #     print(k)
        #     if k != 'scene':
        #         print(data['img'][k].shape)
        #     else:
        #         print(data['img'][k])

        # segAlb = data['img']['segAlb']
        # segMat = data['img']['segMat']
        #
        # pixAlbNum = torch.sum(segAlb).item()
        # pixMatNum = torch.sum(segMat).item()

        # print("pixAlbNum ", pixAlbNum, "pixMatNum ", pixMatNum)

        # if pixAlbNum == 0 or pixMatNum == 0:
        #     print(i, data, pixAlbNum, pixMatNum)
        #     print("--------------------------------------")

        # break

        image = data['img']['im'].cuda()
        reference_locations = data['ref_pos'].cuda()
        # print(net(image, reference_locations))
        with torch.no_grad():
            predictions, scores, albedoPred, roughPred, metalPred = net.net(image, reference_locations)
        print(predictions.shape, scores.shape)
        # print(scores)
        # predictions = predictions.cpu()
        albedoPred = albedoPred.cpu()
        roughPred = roughPred.cpu()
        metalPred = metalPred.cpu()
        # scores = scores.cpu()
        #
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

torch.cuda.empty_cache()
