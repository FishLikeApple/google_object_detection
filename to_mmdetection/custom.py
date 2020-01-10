import os.path as osp

import mmcv
import pandas as pd
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
import glob
import os
import datetime

from random import shuffle

pass_flag_labels = False

class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 num2class_file,
                 ann_file,
                 img_prefix_list,
                 img_scale,
                 img_norm_cfg,
                 raw_dataset_len=2000,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False):

        self.num2class_csv = pd.read_csv(num2class_file)
        CLASSES = self.num2class_csv['Name']
        self.CLASSES_IDs = self.num2class_csv['Id']
        self.id2label = {cat: i+1 for i, cat in enumerate(self.CLASSES_IDs)}

        # images paths, note that it's a list due to multiple datasets
        self.img_prefix_list = img_prefix_list

        self.ann_file = ann_file

        # set sampling configs
        self.label_counting = np.zeros(len(self.id2label)+1)
        self.highest_class_proportion = 1.0 / 75.0  # this proportion is (bbox number / raw dataset image number)
        self.lowest_class_proportion = 0  # this proportion is (bbox number / raw dataset image number)
        self.enough_class = []
        self.lacking_class = list(range(1, len(self.label_counting)-1))
        self.label_index_mapping = []
        self.class_tolerance = 1
        self.raw_dataset_len = raw_dataset_len

        # load annotations proposals)
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        '''
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        '''
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode

        # set the image data
        self.dataset_epoch = 1  # start from 1
        self.update_dataset()

        # set group flag for the sampler
        self.flag = np.zeros(len(self), dtype=np.uint8)

        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.image_list)

    def load_annotations(self, ann_file):
        return pd.read_csv(ann_file)

    def load_proposals(self, proposal_file):
        return pd.read_csv(proposal_file)

    def get_ann_info(self, img_info):
        """changed for open images dataset"""

        ImageID = os.path.splitext(img_info['filename'])[0] #test
        ann_pieces = self.img_infos[self.img_infos['ImageID']==ImageID]
        assert len(ann_pieces) != 0
        
        bboxes = []
        labels = []
        flags = []
        for i in range(len(ann_pieces)):
            #bboxes.append(ann_pieces[i:i+1][['XMin', 'YMin', 'XMax', 'YMax']].values[0])
            bboxes_f = ann_pieces[i:i+1][['XMin', 'YMin', 'XMax', 'YMax']].values[0]
            bboxes.append([bboxes_f[0]*img_info['width'],bboxes_f[1]*img_info['height'],bboxes_f[2]*img_info['width'],bboxes_f[3]*img_info['height']])
            labels.append(self.id2label[ann_pieces[i:i+1]['LabelName'].values[0]])
            flags.append(ann_pieces[i:i+1][['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']].values[0])

        bboxes = np.array(bboxes, ndmin=2)
        labels = np.array(labels)
        flags = np.array(flags, ndmin=2)
        """
        with open("/cos_person/275/1745/object_detection/output/logs.txt", "a") as f:
            print('before', file=f)
            print(bboxes.shape, file=f)
            print(bboxes, file=f)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file=f)
            print('', file=f)
        """
        # will change if using ignore 
        bboxes_ignore = np.array([[]])
        labels_ignore = np.array([])
        flags_ignore = np.array([[]])
        ann = dict(bboxes=bboxes.astype(np.float32),
                   labels=labels.astype(np.int64),
                   flags=flags.astype(np.int64),
                   bboxes_ignore=bboxes_ignore.astype(np.float32),
                   labels_ignore=labels_ignore.astype(np.int64),
                   flags_ignore=flags.astype(np.int64))
        return ann

    def get_labels_and_flags(self, img_info):
        # a short version of the function above

        ImageID = os.path.splitext(img_info['filename'])[0] #test
        ann_pieces = self.img_infos[self.img_infos['ImageID']==ImageID]
        
        bboxes = []
        labels = []
        flags = []
        for i in range(len(ann_pieces)):
            labels.append(self.id2label[ann_pieces[i:i+1]['LabelName'].values[0]])
            flags.append(ann_pieces[i:i+1][['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']].values[0])

        labels = np.array(labels)
        flags = np.array(flags, ndmin=2)
        """
        with open("/cos_person/275/1745/object_detection/output/logs.txt", "a") as f:
            print('before', file=f)
            print(bboxes.shape, file=f)
            print(bboxes, file=f)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file=f)
            print('', file=f)
        """
        ann = dict(labels=labels.astype(np.int64),
                   flags=flags.astype(np.int64))
        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            #if min(img_info['width'], img_info['height']) >= min_size:  # no need for filtering
            valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self, idx, img_info):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        if img_info['width'] > img_info['height']:
            self.flag[idx] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    # it's very strange that the function above seems unable to operate CPU memory and harddisk.
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    # not used
    def sequentially_sample_images(self):
        """sample images to achieve balance"""

        new_image_list = []

        # first iteration
        for i in range(len(self.image_list)):
            img_info = {}
            img_info['filename'] = os.path.basename(self.image_list[i])
            ann = self.get_labels_and_flags(img_info)
            # set mapping
            for label in ann['labels']:
                self.label_index_mapping[label].append(i)
            # set label_counting
            if len(set(ann['labels']).intersection(set(self.enough_class))) <= self.class_tolerance:
                new_image_list.append(self.image_list[i])                    
                for label in ann['labels']: 
                    self.label_counting[label] += 1
                    if self.label_counting[label] >= self.highest_class_amount:
                        if label not in self.enough_class:
                            self.enough_class.append(label)
                    if self.label_counting[label] >= self.lowest_class_amount:
                        if label in self.lacking_class:
                            self.lacking_class.remove(label)

        # second iteration
        if self.lowest_class_amount > 0:
            while len(self.lacking_class) != 0:
                min_class = np.where(np.min(self.label_counting))
                index = sample(self.label_index_mapping[min_class], 1)
                img_info = {}
                img_info['filename'] = os.path.basename(self.image_list[index[0]])
                ann = self.get_labels_and_flags(img_info)
                for label in ann['labels']: 
                    self.label_counting[label] += 1
                    if self.label_counting[label] >= self.lowest_class_amount:
                        if label in self.lacking_class:
                            self.lacking_class.remove(label)
            
            self.image_list = new_image_list

    def update_dataset(self):
        """change the dataset for next epoch"""

        self.image_list = []
        extensions = ["jpg", "jpeg"]
        dataset = (self.dataset_epoch-1) % len(self.img_prefix_list)
        self.img_prefix = self.img_prefix_list[dataset]
        for extension in extensions:
            file_glob = glob.glob(self.img_prefix_list[dataset]+"/*."+extension)  #不分大小写
            self.image_list.extend(file_glob)   #添加文件路径到file_list

        if self.test_mode != True:
            if self.ann_file == None:
                self.ann_file = self.img_prefix + '/image_ann.csv'
            self.img_infos = self.load_annotations(self.ann_file)

        # actively sample and shuffle the dataset if in training
        if self.test_mode:
            print('testing')
        else:
            print('training '+self.img_prefix)

            # set new mapping of label to index
            self.label_index_mapping = []
            for i in range(len(self.label_counting)):
                self.label_index_mapping.append([])

            # achieve data balance
            shuffle(self.image_list)
            #self.highest_class_amount = len(self.image_list) * self.highest_class_proportion
            #self.lowest_class_amount = len(self.image_list) * self.lowest_class_proportion
            #self.sequentially_sample_images()
            #shuffle(self.image_list)

        print('number of images')
        print(len(self.image_list))

    def prepare_train_img(self, idx):
        img_info = {}
        img_info['filename'] = os.path.basename(self.image_list[idx]) # test
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # note that height and width above are before transformation
        img_info['height'] = img.shape[0]
        img_info['width'] = img.shape[1]   

        if not self.test_mode:
            self._set_group_flag(idx, img_info)

        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(img_info)
        gt_bboxes = ann['bboxes']
        if pass_flag_labels:
            gt_labels = np.transpose(np.array([ann['labels'], ann['flags'][:,2]]))
        else:
            gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)

        """
        with open("/cos_person/275/1745/object_detection/output/logs.txt", "a") as f:
            print('gt_labels:', file=f)
            print(gt_labels, file=f)
            print('gt_bboxes:', file=f)
            print(gt_bboxes, file=f)
            print('height and width:', file=f)
            print([img_info['height'], img_info['width']], file=f)
            print('image_id:', file=f)
            print(os.path.splitext(img_info['filename'])[0], file=f)
        """
        """
        # save the numbers of labels
        num_workers = 10
        for label in ann['labels']: 
            self.label_counting[label] += 1
        with open("/cos_person/275/1745/object_detection/output/logs.txt", "a") as f:
            print('self.label_counting:', file=f)
            for i_lc in range(60):
                print(self.label_counting[i_lc*10:(i_lc+1)*10], file=f)
            print(self.label_counting[i_lc*10:], file=f)
            print('', file=f)
        """

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = {}
        img_info['filename'] = os.path.basename(self.image_list[idx])
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        img_info['height'] = img.shape[0]
        img_info['width'] = img.shape[1]

        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)

        ImageID = os.path.splitext(img_info['filename'])[0]
        data = dict(img=imgs, img_meta=img_metas, image_id=ImageID)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data