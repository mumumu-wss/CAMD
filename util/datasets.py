# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# ------------------------------------------------------------------------

import os
import PIL
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN            # 用于标准化图像的均值，通常是 [0.485, 0.456, 0.406]
    std = IMAGENET_DEFAULT_STD              # 用于标准化图像的标准差，[0.229, 0.224, 0.225]

    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,     # 输入图像的尺寸   224
            is_training=True,       # 是否处于训练模式，影响数据增强策略
            color_jitter=args.color_jitter,     # 颜色抖动参数，用于随机调整图像的亮度、对比度和饱和度
            auto_augment=args.aa,       # 自动增强策略，可以使用预定义的增强策略集
            interpolation='bicubic',        # 图像插值方法，这里使用双三次插值 ('bicubic')
            re_prob=args.reprob,        # 随机擦除（Random Erasing）的概率
            re_mode=args.remode,        # 随机擦除的模式
            re_count=args.recount,      #  随机擦除的次数
            mean=mean,      # 用于标准化图像的均值
            std=std,        # 用于标准化图像的标准差
        )
        return transform

    # eval transform
    t = []
    # 计算裁剪比例
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    # 调整图像大小
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    # 中心裁剪
    t.append(transforms.CenterCrop(args.input_size))
    # 转换为张量
    t.append(transforms.ToTensor())
    # 标准化
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class ImagenetWithMask(datasets.ImageFolder):
    def __init__(self, root,
                transform = None,
                with_blockwise_mask=False, ### !!! set to True, enable blockwise masking
                 blockwise_num_masking_patches=75, ### !!! 75 / 196 = 0.38 -> Modify this to increase mask ratio
                 input_size=224, patch_size=16, # no need to change now
                 max_mask_patches_per_block=None, # BEiT default setting, no need to change
                 min_mask_patches_per_block=16, # BEiT default setting, no need to change
                 fixed_num_masking_patches=True, ### set to true, fixed number of masking patch to blockwise_num_masking_patches for sim training 
                 ):
        super().__init__(root, transform)
        self.with_blockwise_mask = with_blockwise_mask
        if with_blockwise_mask:
            from .masking_generator import MaskingGenerator
            window_size = input_size // patch_size
            self.masked_position_generator = MaskingGenerator(
                (window_size, window_size), 
                num_masking_patches=blockwise_num_masking_patches,
                max_num_patches=max_mask_patches_per_block,
                min_num_patches=min_mask_patches_per_block,
                fixed_num_masking_patches=fixed_num_masking_patches
            )
    
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        if self.with_blockwise_mask:
            return sample, target, self.masked_position_generator()
        return sample, target
