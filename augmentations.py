"""
Augmentations module for Fire and Smoke detection
"""
import albumentations as A
import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F


class FireSmokeAugmentor:
    """Augmentation pipeline for fire and smoke detection"""
    
    def __init__(self, img_size=640, train=True):
        self.img_size = img_size
        self.train = train
        
        if train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.OneOf([
                    A.MotionBlur(blur_limit=7),
                    A.GaussianBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                ], p=0.4),
                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120)),
                    A.CLAHE(clip_limit=4.0)
                ], p=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
                A.OneOf([
                    A.CoarseDropout(max_holes=8, min_holes=1, min_height=8, min_width=8, p=0.2),
                    A.RandomFog(p=0.2),
                ], p=0.3),
                A.HorizontalFlip(p=0.5),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def __call__(self, image, bboxes, labels):
        """Apply augmentations to image and bounding boxes"""
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
        
        # Apply augmentations
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            labels=labels
        )
        
        # Convert back to tensor
        image = transformed['image']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'bboxes': transformed['bboxes'],
            'labels': transformed['labels']
        }

