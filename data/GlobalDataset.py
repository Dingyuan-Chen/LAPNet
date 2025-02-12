from torch.utils.data import Dataset
import os
import numpy as np
from albumentations import *
from albumentations.pytorch import ToTensorV2

from PIL import Image
import random
random.seed(199)

class GlobalDataset(Dataset):
    def __init__(self, paths, train=False, transforms=None, fn_mapping=lambda name: name, image_suffix=None):
        self.im_names = os.listdir(paths['images'])
        self.transforms = transforms
        self.train = train
        if image_suffix is not None:
            self.im_names = [n for n in self.im_names if image_suffix in n]

        self.paths = paths
        self.fn_mapping = fn_mapping
        self.targets = self.im_names

    def __getitem__(self, idx):
        image = np.array(Image.open(os.path.join(self.paths['images'], self.im_names[idx])))
        mask = np.array(Image.open(
            os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.im_names[idx]))))

        blob = self.transforms(**dict(image=image, mask=mask))
        img = blob['image']
        mask = blob['mask']

        return img, mask

    def __len__(self):
        return len(self.im_names)


class ToTensor(ToTensorV2):
    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, 'masks': self.apply_to_masks}

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(m, **params) for m in masks]

class ConstantPad(DualTransform):
    def __init__(self,
                 min_height=1024,
                 min_width=1024,
                 value=None,
                 mask_value=None,
                 always_apply=False,
                 p=1.0, ):
        super(ConstantPad, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(ConstantPad, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if rows < self.min_height:
            h_pad_top = 0
            h_pad_bottom = self.min_height - rows
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < self.min_width:
            w_pad_left = 0
            w_pad_right = self.min_width - cols
        else:
            w_pad_left = 0
            w_pad_right = 0

        params.update(
            {"pad_top": h_pad_top, "pad_bottom": h_pad_bottom, "pad_left": w_pad_left, "pad_right": w_pad_right}
        )
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant',
                      constant_values=self.value)

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                      constant_values=self.mask_value)

    def get_transform_init_args_names(self):
        return ("min_height", "min_width", "value", "mask_value")

train_transforms = Compose([
    OneOf([
        HorizontalFlip(True),
        VerticalFlip(True),
        RandomRotate90(True)
    ], p=0.75),
    RandomCrop(512, 512),
    Normalize(mean=(123.675, 116.28, 103.53),
              std=(58.395, 57.12, 57.375),
              max_pixel_value=1),
    ToTensor(),
])

test_transforms = Compose([
    Normalize(mean=(123.675, 116.28, 103.53),
              std=(58.395, 57.12, 57.375),
              max_pixel_value=1),
    ToTensor(),
])