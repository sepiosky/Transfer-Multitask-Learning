from albumentations import OneOf, Compose, MotionBlur, MedianBlur, Blur, RandomBrightnessContrast, GaussNoise, \
    GridDistortion, Rotate, HorizontalFlip, CoarseDropout
from yacs.config import CfgNode
from typing import Union, Tuple
from numpy import ndarray
from cv2 import resize
import numpy as np
import torch


class Preprocessor(object):

    def __init__(self, node_cfg_dataset: CfgNode):
        aug_cfg = node_cfg_dataset.AUGMENTATION


        # !!!Training ONLY!!!
        self.color_aug = self.generate_color_augmentation(aug_cfg)
        self.shape_aug = self.generate_shape_augmentation(aug_cfg)
        self.cutout_aug = self.generate_cutout_augmentation(aug_cfg)
        self.reverse_background = node_cfg_dataset.REVERSE_BACKGROUND

        # !!!~~~BOTH~~~!!!
        self.resize_shape = node_cfg_dataset.RESIZE_SHAPE
        self.to_rgb = node_cfg_dataset.TO_RGB
        self.normalize = node_cfg_dataset.NORMALIZE
        self.normalize_mean = node_cfg_dataset.NORMALIZE_MEAN
        self.normalize_std = node_cfg_dataset.NORMALIZE_STD

        if not self.to_rgb: # 1 channel input
            self.normalize_mean = np.mean(self.normalize_mean)
            self.normalize_std = np.mean(self.normalize_std)

    @staticmethod
    def generate_color_augmentation(aug_cfg: CfgNode) -> Union[Compose, None]:
        color_aug_list = []
        if aug_cfg.BRIGHTNESS_CONTRAST_PROB > 0:
            color_aug_list.append(RandomBrightnessContrast(p=aug_cfg.BRIGHTNESS_CONTRAST_PROB))

        if aug_cfg.BLURRING_PROB > 0:
            blurring = OneOf([
                MotionBlur(aug_cfg.BLUR_LIMIT, p=1),
                MedianBlur(aug_cfg.BLUR_LIMIT, p=1),
                Blur(aug_cfg.BLUR_LIMIT, p=1),
            ], p=aug_cfg.BLURRING_PROB)
            color_aug_list.append(blurring)

        if aug_cfg.GAUSS_NOISE_PROB > 0:
            color_aug_list.append(GaussNoise(p=aug_cfg.GAUSS_NOISE_PROB))
        if len(color_aug_list) > 0:
            color_aug = Compose(color_aug_list, p=1)
            return color_aug
        else:
            return None

    @staticmethod
    def generate_shape_augmentation(aug_cfg: CfgNode) -> Union[Compose, None]:
        shape_aug_list = []
        if aug_cfg.ROTATION_PROB > 0:
            shape_aug_list.append(
                Rotate(limit=aug_cfg.ROTATION_DEGREE, border_mode=1, p=aug_cfg.ROTATION_PROB)
            )
        if aug_cfg.GRID_DISTORTION_PROB > 0:
            shape_aug_list.append(GridDistortion(p=aug_cfg.GRID_DISTORTION_PROB))
        if aug_cfg.HORIZONTAL_FLIP_PROB > 0:
            shape_aug_list.append(HorizontalFlip(p=aug_cfg.HORIZONTAL_FLIP_PROB ))
        if len(shape_aug_list) > 0:
            shape_aug = Compose(shape_aug_list, p=1)
            return shape_aug
        else:
            return None

    @staticmethod
    def generate_cutout_augmentation(aug_cfg: CfgNode):

        cutout_aug_list = []
        if aug_cfg.CUTOUT_PROB > 0:
            # Deprecated
            # cutout_aug_list.append(Cutout(num_holes=1, max_h_size=aug_cfg.HEIGHT//2, max_w_size=aug_cfg.WIDTH//2,
            #                             fill_value=255, p=aug_cfg.CUTOUT_PROB))
            cutout_aug_list.append(CoarseDropout(max_holes=1, max_height=aug_cfg.HEIGHT//2, max_width=aug_cfg.WIDTH//2,
                                        fill_value=255, p=aug_cfg.CUTOUT_PROB))

        if len(cutout_aug_list) > 0:
            cutout_aug = Compose(cutout_aug_list, p=1)
            return cutout_aug
        else:
            return None

    def __call__(self, img: ndarray, is_training: bool, normalize: bool = True) -> Union[ndarray, Tuple]:
        x = img

        if self.reverse_background:
            x = 255 - x

        x = resize(x, self.resize_shape)

        if len(x.shape) < 3:
            if self.to_rgb:
                x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)
            else:
                x = np.expand_dims(x, axis=-1)

        if is_training:
            if self.shape_aug is not None:
                x = self.shape_aug(image=x)['image']

            if self.color_aug is not None:
                x = self.color_aug(image=x)['image']

            if self.cutout_aug is not None:
                x = self.cutout_aug(image=x)['image']

        if self.normalize:
            x = self.normalize_img(x)

        return x

        # Resume the permutation
        img = torch.tensor(x)
        img = img.permute([2, 0, 1])

        return img

    def normalize_img(self, x: ndarray) -> ndarray:
        x = x / 255.
        if self.normalize_mean is not None:
            x = (x - self.normalize_mean) / self.normalize_std
        return x