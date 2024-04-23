import os,sys
sys.path.append("../../src")
import argparse
import yaml
from omegaconf import OmegaConf

from src.utils.cfgnode import CfgNode


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for training the model.')

    # Experiment arguments
    parser.add_argument(
        "--config", type=str, help="Path to (.yml) config file."
    )
    parser.add_argument("--version", '-v', type=int, default=None)
    parser.add_argument('--results_dir', type=str, default="./results/", help="results directory")
    parser.add_argument('--method_name', type=str, default="materialistic", help="results directory")
    parser.add_argument('--resume', action='store_true', default=False,
                        help="Whether to resume training from a checkpoint.")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                        help='Path to the checkpoint directory.')
    parser.add_argument('--precision', type=int, default=32, help='Precision level to use.')
    parser.add_argument('--print_every', type=int, default=3000, help='Print loss every n iterations.')

    # parser.add_argument("--test", action='store_true')
    parser.add_argument('--swin', action='store_true', default=False, help="Test")
    parser.add_argument('--small_size', action='store_true', default=False, help="Test")
    parser.add_argument('--prec', action='store_true', default=False, help="Test")

    args = parser.parse_args()
    with open(args.config) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    return args, cfg
