import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector

import general_processing
import pandas as pd
import numpy as np

from mmdet.apis import init_detector, inference_detector

import sys

num2class_file = '/cos_person/275/1745/object_detection/class-descriptions-boxable.csv'
num2class_csv = pd.read_csv(num2class_file)
classes = []
for cat in num2class_csv['Id']:
    classes.append(cat) 

sample_submit_path_name = num2class_file = '/cos_person/275/1745/object_detection/sample_submission.csv'
output_file_path_name = '/cos_person/275/1745/object_detection/output/sample_submission.csv'

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    # build the model and load checkpoint
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    num2class_file = '/cos_person/275/1745/object_detection/class-descriptions-boxable.csv'
    num2class_csv = pd.read_csv(num2class_file)
    classes = []
    for cat in num2class_csv['Id']:
        classes.append(cat) 

    model.eval()
    output_csv = pd.read_csv(sample_submit_path_name)
    output_csv.loc[:, 'PredictionString'] = np.zeros([len(output_csv)])  # ensure all is predicted

    cfg = mmcv.Config.fromfile(args.config)
    dataset = get_dataset(cfg.data.test)

    for i, result in enumerate(inference_detector(model, dataset.image_list)):
        string_output = general_processing.object_result_to_string(None, result, classes)
        ID = os.path.splitext(os.path.basename(dataset.image_list[i]))[0]
        output_csv.loc[output_csv['ImageId']==ID, 'PredictionString'] = string_output

    output_csv.to_csv(output_file_path_name, index=False)

if __name__ == '__main__':
    main()
