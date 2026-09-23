from argparse import ArgumentParser
import torch
from models.trainer import *
import os
import data_config
import utils
import warnings

warnings.filterwarnings('ignore')


"""
the main function for training the CD networks
"""


def train(args, mode=''):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models(mode=mode)


def test(args, mode=''):
    from models.evaluator import CDEvaluator

    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test', dataset=args.dataset)
    test_root = data_config.DataConfig().get_data_config(args.data_name).root_dir
    test_project = args.project_name
    model = CDEvaluator(args=args, dataloader=dataloader, test_name=os.path.join(test_root, test_project))

    model.eval_models(checkpoint_name='best_ckpt.pt', mode=mode)


def test2(args):
    from models.evaluator import CDEvaluator

    model = CDEvaluator(args=args, dataloader='')
    model.pred_gdal_blocks_write(r'E:\bianhuajiance\cq\fengjiexian\nanbu\2020\500115_clip4.tif',
                                 r'E:\bianhuajiance\cq\fengjiexian\nanbu\2021\500115_clip4.tif')


if __name__ == '__main__':
    # ------------
    # args
    # ------------

    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='BIE_EdgeNet-CDCD', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='CDCD', type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=32, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--multi_scale_train', default=False, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0])

    parser.add_argument('--net_G', default='BIE_EdgeNet', type=str,
                        help='BIE_EdgeNet | BIE_EdgeNet_DINO | BIE_ChangeDINO')
    parser.add_argument('--loss', default='eas', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    utils.get_device(args)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args, mode='ALL')
    test(args, mode='ALL')
