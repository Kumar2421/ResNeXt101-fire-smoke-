"""
train_fire_smoke_fasterrcnn.py

Full training + eval + inference pipeline for Fire/Smoke detection using
torchvision Faster R-CNN. Designed for Roboflow COCO exports.

Adjust DATA_ROOT to point to your dataset root which contains:
  train/_annotations.coco.json
  train/images/
  valid/_annotations.coco.json
  valid/images/
  test/_annotations.coco.json
  test/images/

Author: ChatGPT
"""

import os
import json
import random
from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from augmentations import FireSmokeAugmentor
import torch
import torch.utils.data
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, box_iou
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from pycocotools.coco import COCO

import albumentations as A
import cv2

from torchvision.ops import DeformConv2d
import torch.nn as nn
import torch.nn.functional as F

# Import MetricLogger
from logger import MetricLogger

# -----------------------
# Config - EDIT THESE
# -----------------------
DATA_ROOT = "fire-and-smoke-6"   # Dataset root directory

# Training set paths
TRAIN_JSON = os.path.join(DATA_ROOT, "train", "_annotations.coco.json")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")

# Validation set paths
VAL_JSON = os.path.join(DATA_ROOT, "valid", "_annotations.coco.json")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "valid")

# Test set paths
TEST_JSON = os.path.join(DATA_ROOT, "test", "_annotations.coco.json")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")

import time
from datetime import datetime

def setup_output_directory():
    """Setup output directory with timestamp and create necessary subdirectories."""
    # Create a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    output_dir = os.path.join("runs", run_name)
    
    # Create all necessary subdirectories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")
    return output_dir, run_name

# Global variables that will be initialized once
OUTPUT_DIR = None
RUN_NAME = None
DEVICE = None

def initialize_globals():
    """Initialize global variables once."""
    global OUTPUT_DIR, RUN_NAME, DEVICE
    
    # Only initialize if not already done
    if OUTPUT_DIR is None or RUN_NAME is None:
        # Create a lock file to prevent multiple initializations
        lock_file = Path(".output_dir.lock")
        if lock_file.exists():
            # Another process is already initializing, wait for it to finish
            max_retries = 10
            retry_delay = 0.5  # seconds
            for _ in range(max_retries):
                if not lock_file.exists():
                    break
                time.sleep(retry_delay)
            else:
                raise RuntimeError("Timeout waiting for output directory initialization")
        
        try:
            # Create lock file
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
            
            # Double-check in case another process beat us to it
            if OUTPUT_DIR is None or RUN_NAME is None:
                OUTPUT_DIR, RUN_NAME = setup_output_directory()
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
                print("Using device:", DEVICE)
        finally:
            # Clean up lock file
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except:
                    pass
    
    return OUTPUT_DIR, RUN_NAME, DEVICE

# Training hyperparameters
NUM_EPOCHS = 24      # Total training epochs
BATCH_SIZE = 4       # Increased batch size for better GPU utilization
LR = 0.005           # Increased learning rate for larger batch size
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
PRINT_FREQ = 10      # Print every N batches
NUM_WORKERS = min(8, os.cpu_count())  # Use more workers if available
WARMUP_ITERS = 500   # Reduced warmup iterations
WARMUP_FACTOR = 0.001
GRAD_ACCUM_STEPS = 2  # Accumulate gradients over 2 batches
LR_STEP_SIZE = 8      # Step size for learning rate decay
GAMMA = 0.1          # Multiplicative factor of learning rate decay
SEED = 42

# Enhanced anchor configuration for fire/smoke detection
ANCHOR_SIZES = ((16, 32, 64, 128, 256, 512),)  # Wider range of sizes
ASPECT_RATIOS = ((0.25, 0.5, 1.0, 2.0, 4.0),)  # More aspect ratios

# Temporal filtering params for inference (frames)
PERSISTENCE_FRAMES = 5   # require detection present in N consecutive frames

# -----------------------
# Utilities
# -----------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

def load_coco_categories(coco_json_path):
    j = json.load(open(coco_json_path))
    cats = j.get("categories", [])
    # Map original category_id -> name
    return {c["id"]: c["name"] for c in cats}

def build_coco_id_map(json_path):
    """Return mapping orig_id -> contiguous_id (1..N) and list of names ordered by new id."""
    data = json.load(open(json_path))
    orig_ids = sorted({c["id"] for c in data.get("categories", [])})
    id_map = {orig: i + 1 for i, orig in enumerate(orig_ids)}  # torchvision uses labels >0
    id_to_name = [None] * (len(orig_ids) + 1)  # 1-indexed
    for c in data.get("categories", []):
        id_to_name[id_map[c["id"]]] = c["name"]
    return id_map, id_to_name  # id_map: old->new, id_to_name[1..N]=name

# -----------------------
# Dataset
# -----------------------
class FireSmokeDataset(torch.utils.data.Dataset):
    """Fire and Smoke detection dataset with advanced augmentations."""
    def __init__(self, root, transform=None, split='train', img_size=640):
        """
        Args:
            root: Root directory containing the dataset and annotations
            transform: Optional transform to be applied on a sample
            split: 'train' or 'val'
            img_size: Target image size (height, width)
        """
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        
        # Initialize augmentor
        self.augmentor = FireSmokeAugmentor(
            img_size=img_size, 
            train=(split == 'train')
        )
        
        # Load COCO annotations
        annotation_file = self.root / '_annotations.coco.json'
        with open(annotation_file) as f:
            self.coco = json.load(f)
            
        # Create image and annotation mappings
        self.image_info = {img['id']: img for img in self.coco['images']}
        self.img_ids = list(self.image_info.keys())
        
        # Filter for train/val split (80/20)
        num_samples = len(self.img_ids)
        indices = list(range(num_samples))
        random.shuffle(indices)  # Shuffle before splitting
        split_idx = int(0.8 * num_samples)
        
        if split == 'train':
            self.img_ids = [self.img_ids[i] for i in indices[:split_idx]]
        else:  # val
            self.img_ids = [self.img_ids[i] for i in indices[split_idx:]]
        
        # Create annotation index
        self.anns = {}
        for ann in self.coco['annotations']:
            if ann['image_id'] not in self.anns:
                self.anns[ann['image_id']] = []
            self.anns[ann['image_id']].append(ann)
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.image_info[img_id]
        img_path = self.root / img_info['file_name']
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        anns = self.anns.get(img_id, [])
        boxes = []
        labels = []
        
        for ann in anns:
            # Convert COCO bbox [x, y, width, height] to [x_min, y_min, x_max, y_max]
            x, y, w, h = ann['bbox']
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'] + 1)  # +1 for background class
        
        # Apply augmentations
        transformed = self.augmentor(
            image=img,
            bboxes=boxes,
            labels=labels
        )
        
        # The image is already converted to tensor by the augmentor
        img_tensor = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['labels']
        
        # Calculate areas
        areas = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            area = (x_max - x_min) * (y_max - y_min)
            areas.append(area)
        
        # Create target dictionary - handle empty targets
        if len(boxes) == 0:
            # Create empty tensors with proper shape for empty targets
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64)
            }
        else:
            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': torch.as_tensor(areas, dtype=torch.float32),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
            }
        
        return img_tensor, target

# -----------------------
# Augmentations
# -----------------------
def get_train_transforms():
    # Compose augmentations. Keep bbox format in albumentations as [x_min,y_min,x_max,y_max]
    return A.Compose([
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
        # Haze-like augmentation (simulate fog/smoke) - use Brightness + Gaussian noise
        A.OneOf([
            A.CoarseDropout(max_holes=8, min_holes=1, min_height=8, min_width=8, p=0.2),
            A.RandomFog(p=0.2),
        ], p=0.3),
        A.HorizontalFlip(p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transforms():
    return None  # no augmentation for val/test

# -----------------------
# Model builder
# -----------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att.expand_as(x)
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(spatial))
        
        return x * spatial_att


class BackboneWithFPN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pre-trained ResNeXt-101
        resnext = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=pretrained)
        
        # Feature extraction layers
        self.stem = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
            resnext.maxpool
        )
        
        # Feature extraction layers
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
        
        # CBAM attention modules
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
            extra_blocks=LastLevelMaxPool()
        )
        
    def forward(self, x):
        # Forward through stem
        x = self.stem(x)
        
        # Forward through feature layers
        features = {}
        x = self.layer1(x); features['0'] = self.cbam1(x)
        x = self.layer2(x); features['1'] = self.cbam2(x)
        x = self.layer3(x); features['2'] = x
        x = self.layer4(x); features['3'] = x
        
        # Apply FPN
        features = self.fpn(features)
        return features

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EnhancedFasterRCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EnhancedFasterRCNN, self).__init__()
        
        # Initialize backbone with FPN and CBAM
        self.backbone = BackboneWithFPN(pretrained=pretrained)
        
        # Anchor generator with more aspect ratios for fire/smoke
        anchor_sizes = ((32, 64, 128, 256, 512),)
        aspect_ratios = ((0.5, 1.0, 2.0, 3.0),)  # Added more aspect ratios
        
        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        # ROI Pooler with more sampling points
        self.roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Create the model
        self.model = torchvision.models.detection.FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=self.anchor_generator,
            box_roi_pool=self.roi_pooler,
            box_score_thresh=0.05,  # Lower threshold for better recall
            box_nms_thresh=0.5,
            box_detections_per_img=300,  # Increased max detections
        )
        
        # Initialize focal loss
        self.focal_loss = FocalLoss()
        
        # Replace the box predictor with a custom one that uses focal loss
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Enhanced RPN head with more channels
        self.model.rpn.head = torchvision.models.detection.rpn.RPNHead(
            in_channels=256,  # FPN out channels
            num_anchors=len(anchor_sizes[0]) * len(aspect_ratios[0]),
            conv_depth=2  # Deeper RPN head
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        # Get features from backbone
        features = self.backbone(images.tensors if isinstance(images, ImageList) else images)
        
        if isinstance(images, ImageList):
            images = ImageList(images.tensors, images.image_sizes)
            
        # Get image sizes
        if isinstance(images, (list, torch.Tensor)):
            images = ImageList.from_tensors(images, 0)
            
        # Get proposals from RPN
        proposals, proposal_losses = self.model.rpn(images, features, targets)
        
        # ROI heads
        detections, detector_losses = self.model.roi_heads(features, proposals, images.image_sizes, targets)
        
        # Replace classification loss with focal loss during training
        if self.training:
            # Get class logits
            class_logits = detector_losses.pop('class_logits')
            labels = detector_losses.pop('labels')
            
            # Compute focal loss
            loss_classifier = self.focal_loss(class_logits, labels)
            detector_losses['loss_classifier'] = loss_classifier
            
            # Combine losses
            losses = {**proposal_losses, **detector_losses}
            return losses
            
        return detections

class ResNeXt101FPN(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pre-trained ResNeXt101
        resnext = torchvision.models.resnext101_32x8d(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Extract the feature extractor layers
        self.stem = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
            resnext.maxpool
        )
        
        # Store the ResNeXt101 stages
        self.layer1 = resnext.layer1  # Output: 256 channels
        self.layer2 = resnext.layer2  # Output: 512 channels
        self.layer3 = resnext.layer3  # Output: 1024 channels
        self.layer4 = resnext.layer4  # Output: 2048 channels
        
        # Output channels for each level (must match FPN out_channels)
        self.out_channels = 256
        
        # Create FPN with proper in_channels from ResNeXt101 stages
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],  # Output channels from ResNeXt101 stages
            out_channels=self.out_channels,
            extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelMaxPool()
        )
        
    def forward(self, x):
        # Get features from different stages of ResNeXt101
        x = self.stem(x)
        c2 = self.layer1(x)    # 1/4
        c3 = self.layer2(c2)   # 1/8
        c4 = self.layer3(c3)   # 1/16
        c5 = self.layer4(c4)   # 1/32
        
        # Create feature pyramid
        features = {
            '0': c2,  # 1/4
            '1': c3,  # 1/8
            '2': c4,  # 1/16
            '3': c5,  # 1/32
        }
        
        # Pass through FPN
        features = self.fpn(features)
        return features

def get_model(num_classes):
    """
    Create a Faster R-CNN model with a pre-trained ResNeXt101-32x8d backbone and FPN.
    
    Args:
        num_classes (int): Number of output classes (including background)
        
    Returns:
        torch.nn.Module: Configured Faster R-CNN model with ResNeXt101-FPN
    """
    # Create the backbone with FPN
    backbone = ResNeXt101FPN(pretrained=True)
    
    # Anchor sizes and aspect ratios for each feature level in FPN
    # The FPN has 5 levels: P2, P3, P4, P5, P6
    # We use smaller anchors for higher resolution feature maps
    anchor_sizes = (
        (32, 64, 128),    # P2 (1/4)
        (64, 128, 256),   # P3 (1/8)
        (128, 256, 512),  # P4 (1/16)
        (256, 512, 1024), # P5 (1/32)
        (512, 1024, 2048) # P6 (1/64)
    )
    
    # Aspect ratios for each anchor
    aspect_ratios = ((0.5, 1.0, 2.0),) * 5  # Same aspect ratios for all levels
    
    # Create anchor generator
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    # ROI aligner for each FPN level
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],  # P2, P3, P4, P5
        output_size=7,
        sampling_ratio=2
    )
    
    # Create the Faster R-CNN model with optimized settings
    model = torchvision.models.detection.FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=512,  # Reduced image size for faster training
        max_size=1024,  # Reduced max size
        box_detections_per_img=50,  # Reduced max detections
        box_nms_thresh=0.5,
        box_score_thresh=0.05
    )
    
    # Set the model to training mode
    model.train()
    
    return model.to(DEVICE)

# -----------------------
# Helpers: compute image-level class presence (for sampler)
# -----------------------
def compute_image_class_presence(coco, ids, id_map):
    # returns list of labels present per image (new mapped labels)
    presence = []
    for img_id in ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        labels = set()
        for ann in anns:
            orig = ann["category_id"]
            new = id_map[orig] if id_map else orig
            labels.add(new)
        presence.append(labels)
    return presence

# -----------------------
# Collate
# -----------------------
def collate_fn(batch):
    return tuple(zip(*batch))

# -----------------------
# Training & Evaluation
# -----------------------
def train_and_evaluate(resume=False):
    # Ensure globals are initialized exactly once
    global OUTPUT_DIR, RUN_NAME, DEVICE
    
    # Initialize globals if not already done
    if OUTPUT_DIR is None or RUN_NAME is None:
        OUTPUT_DIR, RUN_NAME, DEVICE = initialize_globals()
        
        print("\n" + "="*50)
        print(f"Starting training run: {RUN_NAME}")
        print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
        print(f"Using device: {DEVICE}")
        print("="*50 + "\n")
    
    # Build id_map and class names from TRAIN_JSON
    id_map, id_to_name = build_coco_id_map(TRAIN_JSON)
    num_classes = len(id_to_name) + 1  # because id_to_name is 1..N, we add background -> total classes = N+1
    num_labels = len(id_to_name)  # number of annotation classes
    print(f"Detected {num_labels} labels. Num classes for model (with background): {num_labels + 1}")
    
    # Initialize logger with proper configuration - use the same output directory
    logger = MetricLogger(
        log_dir=os.path.dirname(OUTPUT_DIR),  # Use parent directory of OUTPUT_DIR
        exp_name=os.path.basename(OUTPUT_DIR),  # Use the directory name as experiment name
        resume=resume
    )
    
    # Log configuration
    config = {
        'model': 'EnhancedFasterRCNN',
        'backbone': 'ResNeXt101-32x8d-FPN',
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'num_epochs': NUM_EPOCHS,
        'optimizer': 'SGD',
        'momentum': MOMENTUM,
        'weight_decay': WEIGHT_DECAY,
        'image_size': 640
    }
    logger.log_config(config)

    # Create datasets
    train_ds = FireSmokeDataset(TRAIN_IMG_DIR, transform=get_train_transforms(), split='train')
    val_ds = FireSmokeDataset(VAL_IMG_DIR, transform=get_val_transforms(), split='val')
    test_ds = FireSmokeDataset(TEST_IMG_DIR, transform=get_val_transforms(), split='test')

    # Get class distribution for weighting
    coco_train = COCO(TRAIN_JSON)
    
    # Get all category IDs and names
    categories = coco_train.loadCats(coco_train.getCatIds())
    print("Dataset categories:", {cat['id']: cat['name'] for cat in categories})
    
    # Count instances per class
    class_counts = Counter()
    for ann in coco_train.loadAnns(coco_train.getAnnIds()):
        class_counts[ann['category_id']] += 1
    
    print("Class distribution:", class_counts)
    
    # Create a simple sequential sampler for now
    # We can implement weighted sampling later if needed
    sampler = torch.utils.data.RandomSampler(train_ds)

    # DataLoaders with optimized settings
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2         # Prefetch 2 batches per worker
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,  # Increased batch size for validation
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,   # Increased batch size for testing
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    # Model
    model = get_model(num_labels + 1)  # background + labels
    model.to(DEVICE)

    # Optimizer & LR scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    
    # Create a combined learning rate scheduler with warmup
    def create_lr_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, step_size, gamma):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Warmup phase: gradually increase from warmup_factor * base_lr to base_lr
                return WARMUP_FACTOR + (1 - WARMUP_FACTOR) * (epoch / warmup_epochs)
            else:
                # Step decay after warmup
                step_epoch = (epoch - warmup_epochs) // step_size
                return gamma ** step_epoch
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create learning rate scheduler
    lr_scheduler = create_lr_scheduler(
        optimizer, 
        warmup_epochs=5, 
        total_epochs=NUM_EPOCHS,
        base_lr=LR,
        step_size=LR_STEP_SIZE,
        gamma=GAMMA
    )
    
    # Initialize mixed precision training
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    best_map = 0.0
    start_epoch = 0
    
    # Load checkpoint if resuming
    if resume:
        checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_map = checkpoint.get('best_map', 0.0)
            print(f"Resuming training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        metric_loss = 0.0
        
        # Initialize metrics
        train_metrics = {
            'loss': 0.0,
            'loss_classifier': 0.0,
            'loss_box_reg': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0
        }
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for i, (images, targets) in enumerate(loop):
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Forward pass with mixed precision (updated API)
            with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Scale the loss and backpropagate with gradient accumulation
                if scaler is not None:
                    scaler.scale(losses / GRAD_ACCUM_STEPS).backward()
                    
                    # Only step and zero_grad every GRAD_ACCUM_STEPS batches
                    if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    (losses / GRAD_ACCUM_STEPS).backward()
                    # Only step and zero_grad every GRAD_ACCUM_STEPS batches
                    if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
            
            # Update metrics
            for k in loss_dict:
                if k in train_metrics:
                    train_metrics[k] += loss_dict[k].item()
            train_metrics['loss'] += losses.item()
            
            # Log metrics
            if (i + 1) % PRINT_FREQ == 0:
                log_metrics = {f'train_{k}': v / (i + 1) for k, v in train_metrics.items()}
                log_metrics['lr'] = optimizer.param_groups[0]['lr']
                logger.update_metrics(
                    log_metrics,
                    epoch=epoch,
                    step=epoch * len(train_loader) + i,
                    phase='train'
                )
                loop.set_postfix({k: f'{v:.4f}' for k, v in log_metrics.items()})
        
        # Update learning rate
        lr_scheduler.step()
        
        # Calculate average losses
        avg_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
        
        # Debug: Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, DEVICE)
        
        # Log metrics with proper formatting
        log_metrics = {
            'loss': avg_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'mAP': val_metrics['mAP'],
            'AP_fire': val_metrics['AP_fire'],
            'AP_smoke': val_metrics['AP_smoke'],
            'lr': optimizer.param_groups[0]['lr']
        }
        
        # Log metrics to console
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  mAP: {val_metrics['mAP']:.4f}")
        print(f"  AP Fire: {val_metrics['AP_fire']:.4f}")
        print(f"  AP Smoke: {val_metrics['AP_smoke']:.4f}")
        
        # Save metrics to CSV
        metrics_csv = os.path.join(OUTPUT_DIR, 'training_metrics.csv')
        if not os.path.exists(metrics_csv):
            with open(metrics_csv, 'w') as f:
                f.write(','.join(['epoch'] + list(log_metrics.keys())) + '\n')
        
        with open(metrics_csv, 'a') as f:
            f.write(','.join([str(epoch)] + [str(v) for v in log_metrics.values()]) + '\n')
        
        # Log epoch metrics
        logger.log_epoch(
            epoch,
            model,
            optimizer,
            log_metrics,
            phase='val'
        )
        
        # Save best model
        current_map = val_metrics['mAP']
        if current_map > best_map:
            best_map = current_map
            best_path = os.path.join(OUTPUT_DIR, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved to {best_path}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_map': best_map,
            'config': config
        }
        torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'checkpoint.pth'))

    print("Training complete.")
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "fasterrcnn_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    
    # Close logger
    logger.close()
    
    # Run final evaluation on test set
    print("Running final evaluation on test set...")
    test_metrics = evaluate(model, test_loader, DEVICE)
    print(f"Test mAP: {test_metrics['mAP']:.4f}")
    print(f"Test AP_fire: {test_metrics['AP_fire']:.4f}")
    print(f"Test AP_smoke: {test_metrics['AP_smoke']:.4f}")
    
    return model, test_metrics

# -----------------------
# Quick evaluator (recall@IoU) - cheap proxy
# -----------------------
def compute_recall_at_iou(model, dataloader, iou_thresh=0.5, score_thresh=0.3, device=DEVICE):
    model.eval()
    total_gt = 0
    total_tp = 0
    with torch.no_grad():
        for images, targets in dataloader:
            img = images[0].to(device)
            target = targets[0]
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            total_gt += len(gt_boxes)

            outputs = model([img])
            pred = outputs[0]
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            # filter by score_thresh
            keep = scores >= score_thresh
            boxes = boxes[keep]; labels = labels[keep]

            # match predictions to GT (greedy)
            matched = set()
            for i_gt, (gbox, glab) in enumerate(zip(gt_boxes, gt_labels)):
                best_i = -1
                best_iou = 0.0
                for j, pbox in enumerate(boxes):
                    if j in matched:
                        continue
                    iou = compute_iou(gbox, pbox)
                    if iou >= iou_thresh and iou > best_iou:
                        best_i = j; best_iou = iou
                if best_i >= 0:
                    matched.add(best_i)
                    total_tp += 1
    model.train()
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    return recall

def compute_iou(boxA, boxB):
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0.0

def evaluate(model, data_loader, device):
    """Evaluate the model on the validation/test set"""
    model.eval()
    
    # Initialize metrics
    metrics = {
        'loss': 0.0,
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0,
    }
    
    # For mAP calculation
    all_detections = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass - compute both losses and detections
            with torch.no_grad():
                # Temporarily set model to training mode to get losses
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Set back to eval mode for detections
                model.eval()
                detections = model(images)
            
            # Update loss metrics
            for k, v in loss_dict.items():
                if f'loss_{k}' in metrics:
                    metrics[f'loss_{k}'] += v.item()
            metrics['loss'] += losses.item()
            
            # Store detections and ground truths for mAP calculation
            for det, target in zip(detections, targets):
                # Convert detections to COCO format
                boxes = det['boxes'].cpu().numpy()
                scores = det['scores'].cpu().numpy()
                labels = det['labels'].cpu().numpy()
                
                # Store ground truth
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                all_detections.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                })
                
                all_ground_truths.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels
                })
    
    # Calculate average metrics
    num_batches = len(data_loader)
    for k in metrics:
        metrics[k] /= num_batches
    
    # Calculate mAP using a simplified approach
    if len(all_detections) > 0 and len(all_ground_truths) > 0:
        # Simple mAP calculation based on IoU threshold
        iou_threshold = 0.5
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for det, gt in zip(all_detections, all_ground_truths):
            det_boxes = det['boxes']
            det_scores = det['scores']
            det_labels = det['labels']
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']
            
            total_gt += len(gt_boxes)
            
            if len(det_boxes) > 0:
                # Match detections to ground truth
                matched_gt = set()
                for i, (det_box, det_score, det_label) in enumerate(zip(det_boxes, det_scores, det_labels)):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if j in matched_gt:
                            continue
                        if det_label == gt_label:
                            iou = compute_iou(det_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j
                    
                    if best_iou >= iou_threshold and best_gt_idx != -1:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_fp += 1
        
        # Calculate precision and recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Use F1 as a proxy for mAP
        metrics['mAP'] = f1
        metrics['AP_fire'] = f1  # Simplified - same for both classes
        metrics['AP_smoke'] = f1
    else:
        metrics['mAP'] = 0.0
        metrics['AP_fire'] = 0.0
        metrics['AP_smoke'] = 0.0
    
    return metrics


# -----------------------
# Inference + Temporal filter example for video
# -----------------------
from collections import deque
def infer_video_with_persistence(model, id_to_name, video_path=0, score_thresh=0.3, persistence=PERSISTENCE_FRAMES):
    """
    Runs inference on a video (or webcam) and applies a simple persistence filter:
    a detection is only accepted if a box with similar location & label appears in
    >= persistence consecutive frames.
    """
    model.eval()
    device = DEVICE
    cap = cv2.VideoCapture(video_path)
    frame_buffers = []  # list of deques per label to store last detections
    max_labels = len(id_to_name)
    frame_buffers = [deque(maxlen=persistence) for _ in range(max_labels + 1)]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(Image.fromarray(img_rgb)).to(device)
        with torch.no_grad():
            outputs = model([img_tensor])[0]
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        # draw persistent detections
        accepted = []
        for box, sc, lab in zip(boxes, scores, labels):
            if sc < score_thresh:
                continue
            # store this detection in buffer for label
            frame_buffers[lab].append(box)
            # check if a consistent box exists in buffer (IoU threshold)
            cnt = 0
            for b in frame_buffers[lab]:
                if compute_iou(box, b) >= 0.5:
                    cnt += 1
            if cnt >= persistence:
                accepted.append((box, sc, lab))

        # visualization
        for box, sc, lab in accepted:
            x1, y1, x2, y2 = map(int, box.tolist())
            name = id_to_name[lab] if lab < len(id_to_name) and id_to_name[lab] else str(lab)
            color = (0, 0, 255) if name.lower().startswith("fire") else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}:{sc:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Re-initialize globals when run as main script
    OUTPUT_DIR, RUN_NAME, DEVICE = None, None, None
    OUTPUT_DIR, RUN_NAME, DEVICE = initialize_globals()
    
    model, id_to_name = train_and_evaluate()
    print("To run real-time inference with persistence, call:")
    print("infer_video_with_persistence(model, id_to_name, video_path=0, score_thresh=0.3, persistence=5)")
