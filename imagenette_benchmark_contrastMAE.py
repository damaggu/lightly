# -*- coding: utf-8 -*-
"""
Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)
You can download the ImageNette dataset from here: https://github.com/fastai/imagenette

Code has been tested on a V100 GPU with 16GBytes of video memory.

Code to reproduce the benchmark results:

Results (5.3.2022):
------------------------------------------------------------------------------------------
| Model         | Batch Size | Epochs |  KNN Test Accuracy |       Time | Peak GPU Usage |
------------------------------------------------------------------------------------------
| BarlowTwins   |        256 |    200 |              0.587 |   86.2 Min |      4.0 GByte |
| BYOL          |        256 |    200 |              0.619 |   88.6 Min |      4.3 GByte |
| DCL (*)       |        256 |    200 |              0.762 |   53.3 Min |      4.3 GByte |
| DCLW (*)      |        256 |    200 |              0.755 |   53.7 Min |      4.3 GByte |
| DINO (Res18)  |        256 |    200 |              0.736 |   86.5 Min |      4.1 GByte |
| MSN (ViT-S)   |        256 |    200 |              0.741 |   92.7 Min |     16.3 GByte |
| Moco          |        256 |    200 |              0.727 |   87.3 Min |      4.3 GByte |
| NNCLR         |        256 |    200 |              0.726 |   86.8 Min |      4.2 GByte |
| SimCLR        |        256 |    200 |              0.771 |   82.2 Min |      3.9 GByte |
| SimSiam       |        256 |    200 |              0.669 |   78.6 Min |      3.9 GByte |
| SMoG          |        128 |    200 |              0.698 |  220.9 Min |     14.3 GByte |
| SwaV          |        256 |    200 |              0.748 |   77.6 Min |      4.0 GByte |
------------------------------------------------------------------------------------------
| BarlowTwins   |        256 |    800 |              0.789 |  330.9 Min |      4.0 GByte |
| BYOL          |        256 |    800 |              0.851 |  332.7 Min |      4.3 GByte |
| DCL (*)       |        256 |    800 |              0.816 |  213.1 Min |      4.3 GByte |
| DCLW (*)      |        256 |    800 |              0.827 |  213.1 Min |      4.3 GByte |
| DINO (Res18)  |        256 |    800 |              0.881 |  613.9 Min |      6.7 GByte |
| MSN (ViT-S)   |        256 |    800 |              0.834 |  376.1 Min |     16.3 GByte |
| Moco          |        256 |    800 |              0.832 |  322.8 Min |      4.2 GByte |
| NNCLR         |        256 |    800 |              0.848 |  341.4 Min |      4.2 GByte |
| SimCLR        |        256 |    800 |              0.858 |  324.8 Min |      3.9 GByte |
| SimSiam       |        256 |    800 |              0.852 |  316.0 Min |      3.9 GByte |
| SwaV          |        256 |    800 |              0.899 |  554.7 Min |      6.6 GByte |
------------------------------------------------------------------------------------------

(*): Different runtime and memory requirements due to different hardware settings
and pytorch version. Runtime and memory requirements are comparable to SimCLR
with the default settings.

"""
import copy
import io
import math
import os

import sys

sys.path.append(os.path.join(os.getcwd(), "lightly"))
from lightly.utils import BenchmarkModule
from lightly.models.modules import masked_autoencoder

import time
import lightly
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from kornia.feature import DenseSIFTDescriptor
from lightly.models import modules
from lightly.models.modules import heads
from torchvision.transforms import Normalize, Compose

from modified_items import MAEBackbone, MAEDecoder, learned_token_mask
from lightly.models import utils

from lightly.utils import scheduler
from pytorch_lightning.loggers.wandb import WandbLogger
from kornia import filters
from torch.nn import functional as F
from pl_bolts.optimizers.lars import LARS
from sklearn.cluster import KMeans
import os
from torchvision.models.vision_transformer import _vision_transformer
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# import display


# simple arg parser
import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument('--run_name', type=str, default=None)

run_name = parser.parse_args().run_name

# try out inat pytorch dataloader
#
# from torchvision.datasets import INaturalist
#
# test = INaturalist(
#     root='./datasets/inat_pytorch',
#     version='2021_train',
#     download=True,
# )
# print('done')


# wandb offline
# os.environ['WANDB_MODE'] = 'offline'

import wandb

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")
eli = False
dist = False
test = False
args = {}
args["dataset"] = "imagenette"

if args["dataset"] == "cifar10" or args["dataset"] == "imagenette":
    # input_size = 128
    input_size = 224
elif args["dataset"] in ["iNat2021mini", "inat_birds"]:
    input_size = 224
elif args["dataset"] in ["ChestMNIST", "RetinaMNIST", "BreastMNIST"]:
    # input_size = 28
    input_size = 224
else:
    raise ValueError("Invalid dataset name")

args["input_size"] = input_size
args['flatten'] = True
args["num_workers"] = 6
args["memory_bank_size"] = 4096
if eli:
    args["batch_size"] = 4096
    args["max_epochs"] = 1000
    args["val_epoch"] = 50
else:
    args["max_epochs"] = 800
    args["val_epoch"] = 10
    if input_size == 224:
        args["batch_size"] = 128 if dist else 64
    else:
        args["batch_size"] = 4096 if dist else 2048

args['MAE_baseLR'] = 0.0015
args['accumulate_grad_batches'] = 8
args["effective_bs"] = args["batch_size"] * args['accumulate_grad_batches']

if input_size == 224:
    args["ft_batch_size"] = 256 if dist else 128
else:
    args["ft_batch_size"] = 4096 if dist else 2048


gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
if dist:
    args["gpus"] = gpus
    args['batch_size'] = args['batch_size'] * gpus
    args['ft_batch_size'] = args['ft_batch_size'] * gpus


args["warmup_epochs"] = 10
args["mae_masking_ratio"] = 0.75
args["msn_masking_ratio"] = 0.15
args["patch_size"] = 16
args["do_probing"] = False
args["do_kNN"] = True
args["do_medmnist"] = False
args["knn_k"] = 200
args["knn_t"] = 0.1
args["n_runs"] = 1
if args["dataset"] in ["ChestMNIST", "RetinaMNIST", "BreastMNIST"]:
    args["do_medmnist"] = True
    args["ft_batch_size"] = 8192 if dist else 4096
    args["max_epochs"] = 50
    args["val_epoch"] = 5
    # mae_masking_ratio = 0.5
    # msn_masking_ratio = 0.15
    # patch_size = 2
args["epochs_medmnist"] = 50
args["lr_medmnist"] = 0.1
args["gamma_medmnist"] = 0.1
args["milestones_medmnist"] = [
    0.5 * args["epochs_medmnist"],
    0.75 * args["epochs_medmnist"],
]
args["weight_decay"] = 0
args["blr"] = 0.1
args["lr"] = args["blr"] * args["ft_batch_size"] / 256
args["epochs"] = 100
args["clip_grad"] = 1.0
args["accum_iter"] = 1
args["model_dim"] = 512
args["is_3d"] = False
args["min_lr"] = 0.00001

if test:
    args["num_workers"] = 0
    args["max_epochs"] = 2
    args["val_epoch"] = 1
    args["warmup_epochs"] = 0
    args["epochs_medmnist"] = 2
    args["epochs"] = 2
    if input_size == 28:
        args["batch_size"] = 16
        args["ft_batch_size"] = 1024
    else:
        args["batch_size"] = 2
        args["ft_batch_size"] = 2
        args["mae_masking_ratio"] = 0.75
        # args["model_dim"] = 384
        args["model_dim"] = 512
        args["patch_size"] = 16

lr_factor = args["effective_bs"] / 256  # scales the learning rate linearly with batch size

# msn_aug_mode = 'v9'
msn_aug_mode = "v0"
# byol_mode = 'v3'
byol_mode = "v0"
# args["dataset"]  = 'cifar10'
# args["dataset"]  = 'imagenette'
# args["dataset"]  = 'iNat2021mini'
# args["dataset"]  = 'RetinaMNIST'
# args["dataset"]  = 'BreastMNIST'
project_name = args["dataset"] + "_benchmark_correctedK"
log_model = True

#### linear probing args

# Set to True to enable Distributed Data Parallel training.
distributed = dist

# Set to True to enable Synchronized Batch Norm (requires distributed=True).
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = dist

# Set to True to gather features from all gpus before calculating
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all
# gpus, otherwise only features from the same gpu are used.
gather_distributed = dist

# benchmark

# use a GPU if available

# gpus = 0

if distributed:
    distributed_backend = "ddp"
    # reduce batch size for distributed training
    batch_size = args["batch_size"] // gpus
else:
    distributed_backend = None
    # limit to single gpu if not using distributed training
    gpus = min(gpus, 1)

# The dataset structure should be like this:

normalize_transform = torchvision.transforms.Normalize(
    mean=lightly.data.collate.imagenet_normalize["mean"],
    std=lightly.data.collate.imagenet_normalize["std"],
)
inv_normalize = torchvision.transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
)

# Use SimCLR augmentations
if args["dataset"] == "imagenette":

    if input_size == 128:
        collate_fn = lightly.data.SimCLRCollateFunction(
            input_size=input_size,
        )
        if byol_mode == "v1" or byol_mode == "v2" or byol_mode == "v3":
            # import Normalize from torchvision transforms
            collate_fn = lightly.data.SimCLRCollateFunction(
                input_size=input_size,
                normalize={
                    "mean": (0.48145466, 0.4578275, 0.40821073),
                    "std": (0.26862954, 0.26130258, 0.27577711),
                },
            )

        # Multi crop augmentation for SwAV
        swav_collate_fn = lightly.data.SwaVCollateFunction(
            crop_sizes=[128, 64],
            crop_counts=[2, 6],  # 2 crops @ 128x128px and 6 crops @ 64x64px
        )

        # Multi crop augmentation for DINO, additionally, disable blur for cifar10
        dino_collate_fn = lightly.data.DINOCollateFunction(
            global_crop_size=128,
            local_crop_size=64,
        )

        # Two crops for SMoG
        smog_collate_function = lightly.data.collate.SMoGCollateFunction(
            crop_sizes=[128, 128],
            crop_counts=[1, 1],
            crop_min_scales=[0.2, 0.2],
            crop_max_scales=[1.0, 1.0],
        )
        # Collate function passing geometrical transformation for VICRegL
        vicregl_collate_fn = lightly.data.VICRegLCollateFunction(
            global_crop_size=128, local_crop_size=64, global_grid_size=4, local_grid_size=2
        )
        msn_collate_fn = lightly.data.MSNCollateFunction(random_size=128, focal_size=64)

        vqgan_collate_fn = lightly.data.MAECollateFunction(normalize=None, input_size=128)

        # No additional augmentations for the test set
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(input_size),
                torchvision.transforms.CenterCrop(128),
                torchvision.transforms.ToTensor(),
                normalize_transform,
            ]
        )
    elif input_size == 224:
        collate_fn = lightly.data.SimCLRCollateFunction(
            input_size=input_size,
        )
        if byol_mode == "v1" or byol_mode == "v2" or byol_mode == "v3":
            # import Normalize from torchvision transforms
            collate_fn = lightly.data.SimCLRCollateFunction(
                input_size=input_size,
                normalize={
                    "mean": (0.48145466, 0.4578275, 0.40821073),
                    "std": (0.26862954, 0.26130258, 0.27577711),
                },
            )
        swav_collate_fn = lightly.data.SwaVCollateFunction(
            crop_sizes=[224, 96],
            crop_counts=[2, 6],  # 2 crops @ 224x224px and 6 crops @ 96x96px
        )
        dinocollate_fn = lightly.data.DINOCollateFunction(
            global_crop_size=224,
            local_crop_size=96,
        )
        smog_collate_function = lightly.data.collate.SMoGCollateFunction(
            crop_sizes=[224, 224],
            crop_counts=[1, 1],
            crop_min_scales=[0.2, 0.2],
            crop_max_scales=[1.0, 1.0],
        )
        vicregl_collate_fn = lightly.data.VICRegLCollateFunction(
            global_crop_size=224, local_crop_size=96, global_grid_size=7, local_grid_size=3
        )
        msn_collate_fn = lightly.data.MSNCollateFunction(random_size=224, focal_size=96)
        vqgan_collate_fn = lightly.data.MAECollateFunction(normalize=None, input_size=224)
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(input_size),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize_transform,
            ]
        )


elif args["dataset"] == "cifar10":
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=32,
        gaussian_blur=0.0,
    )

    # Multi crop augmentation for SwAV, additionally, disable blur for cifar10
    swav_collate_fn = lightly.data.SwaVCollateFunction(
        crop_sizes=[32],
        crop_counts=[2],  # 2 crops @ 32x32px
        crop_min_scales=[0.14],
        gaussian_blur=0,
    )

    # Multi crop augmentation for DINO, additionally, disable blur for cifar10
    dino_collate_fn = lightly.data.DINOCollateFunction(
        global_crop_size=32,
        n_local_views=0,
        gaussian_blur=(0, 0, 0),
    )

    # Two crops for SMoG
    smog_collate_function = lightly.data.collate.SMoGCollateFunction(
        crop_sizes=[32, 32],
        crop_counts=[1, 1],
        gaussian_blur_probs=[0.0, 0.0],
        crop_min_scales=[0.2, 0.2],
        crop_max_scales=[1.0, 1.0],
    )
elif args["dataset"] in ["iNat2021mini", "inat_birds"]:
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_size,
        gaussian_blur=0.0,  # from eli's paper
    )

    # Multi crop augmentation for SwAV
    swav_collate_fn = lightly.data.SwaVCollateFunction(
        crop_sizes=[224, 96],  # from paper
        crop_counts=[2, 6],  # 2 crops @ 128x128px and 6 crops @ 64x64px
    )

    # Multi crop augmentation for DINO, additionally, disable blur for cifar10
    dino_collate_fn = lightly.data.DINOCollateFunction(
        global_crop_size=224,
        local_crop_size=96,
    )

    # Two crops for SMoG
    smog_collate_function = lightly.data.collate.SMoGCollateFunction(
        crop_sizes=[128, 128],
        crop_counts=[1, 1],
        crop_min_scales=[0.2, 0.2],
        crop_max_scales=[1.0, 1.0],
    )
    # Collate function passing geometrical transformation for VICRegL
    vicregl_collate_fn = lightly.data.VICRegLCollateFunction(
        global_crop_size=224, local_crop_size=96, global_grid_size=4, local_grid_size=2
    )
    msn_collate_fn = lightly.data.MSNCollateFunction(random_size=224, focal_size=96)
    # No additional augmentations for the test set
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize_transform,
        ]
    )
elif args["dataset"] in ["medmnist", "ChestMNIST", "RetinaMNIST"]:
    if input_size == 224:
        collate_fn = lightly.data.SimCLRCollateFunction(
            input_size=input_size,
            gaussian_blur=0.0,  # from eli's paper
        )

        # Multi crop augmentation for SwAV
        swav_collate_fn = lightly.data.SwaVCollateFunction(
            crop_sizes=[224, 96],  # from paper
            crop_counts=[2, 6],  # 2 crops @ 128x128px and 6 crops @ 64x64px
        )

        # Multi crop augmentation for DINO, additionally, disable blur for cifar10
        dino_collate_fn = lightly.data.DINOCollateFunction(
            global_crop_size=224,
            local_crop_size=96,
        )

        # Two crops for SMoG
        smog_collate_function = lightly.data.collate.SMoGCollateFunction(
            crop_sizes=[128, 128],
            crop_counts=[1, 1],
            crop_min_scales=[0.2, 0.2],
            crop_max_scales=[1.0, 1.0],
        )
        # Collate function passing geometrical transformation for VICRegL
        vicregl_collate_fn = lightly.data.VICRegLCollateFunction(
            global_crop_size=224, local_crop_size=96, global_grid_size=4, local_grid_size=2
        )
        msn_collate_fn = lightly.data.MSNCollateFunction(random_size=224, focal_size=96)
        # No additional augmentations for the test set
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                # torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize_transform,
            ]
        )
    elif input_size == 28:
        collate_fn = lightly.data.SimCLRCollateFunction(
            input_size=28,
            gaussian_blur=0.0,
        )

        # Multi crop augmentation for SwAV
        swav_collate_fn = lightly.data.SwaVCollateFunction(
            crop_sizes=[28, 12],
            crop_counts=[2, 6]  # 2 crops @ 128x128px and 6 crops @ 64x64px
        )

        # Multi crop augmentation for DINO, additionally, disable blur for cifar10
        dino_collate_fn = lightly.data.DINOCollateFunction(
            global_crop_size=28,
            local_crop_size=12,
        )

        # Two crops for SMoG
        smog_collate_function = lightly.data.collate.SMoGCollateFunction(
            crop_sizes=[16, 16],
            crop_counts=[1, 1],
            crop_min_scales=[0.2, 0.2],
            crop_max_scales=[1.0, 1.0],
        )
        # Collate function passing geometrical transformation for VICRegL
        vicregl_collate_fn = lightly.data.VICRegLCollateFunction(
            global_crop_size=28, local_crop_size=12, global_grid_size=4, local_grid_size=2
        )
        msn_collate_fn = lightly.data.MSNCollateFunction(random_size=28, focal_size=12)
        # No additional augmentations for the test set
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(input_size),
                # torchvision.transforms.CenterCrop(28),
                torchvision.transforms.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        raise NotImplementedError

#  Single crop augmentation for MAE
mae_collate_fn = lightly.data.MAECollateFunction()

if args["dataset"] == "imagenette":
    path_to_train = "./datasets/imagenette2-160/train/"
    path_to_test = "./datasets/imagenette2-160/val/"
elif args["dataset"] == "cifar10":
    path_to_train = "./datasets/cifar10/train/"
    path_to_test = "./datasets/cifar10/test/"
elif args["dataset"] == "iNat2021mini":
    path_to_train = "./datasets/inat/train/train_mini/"
    path_to_test = "./datasets/inat/val/"
elif args["dataset"] == "inat_birds":
    path_to_train = "./datasets/inat/birds_train/"
    path_to_test = "./datasets/inat/birds_val/"
elif args["dataset"] in ["ChestMNIST", "RetinaMNIST"]:
    import medmnist
    import torchvision.transforms as T

    root_dir_train = f"./datasets/medmnist/{args['dataset']}/train"
    if not os.path.exists(root_dir_train):
        os.makedirs(root_dir_train)
    if args["dataset"] == "ChestMNIST":
        train_dataset = medmnist.ChestMNIST(
            as_rgb=True,
            split="train",
            download=True,
            root=root_dir_train,
        )
    elif args["dataset"] == "RetinaMNIST":
        train_dataset = medmnist.RetinaMNIST(
            as_rgb=True,
            split="train",
            download=True,
            root=root_dir_train,
        )
    else:
        raise NotImplementedError
    root_dir_test = f"./datasets/medmnist/{args['dataset']}/test"
    if not os.path.exists(root_dir_test):
        os.makedirs(root_dir_test)
    if args["dataset"] == "ChestMNIST":
        test_dataset = medmnist.ChestMNIST(
            as_rgb=True,
            split="test",
            download=True,
            root=root_dir_test,
        )
    elif args["dataset"] == "RetinaMNIST":
        test_dataset = medmnist.RetinaMNIST(
            as_rgb=True,
            split="test",
            download=True,
            root=root_dir_test,
        )
    root_dir_val = f"./datasets/medmnist/{args['dataset']}/val"
    if not os.path.exists(root_dir_val):
        os.makedirs(root_dir_val)
    if args["dataset"] == "ChestMNIST":
        val_dataset = medmnist.ChestMNIST(
            as_rgb=True,
            split="val",
            download=True,
            root=root_dir_val,
        )
    elif args["dataset"] == "RetinaMNIST":
        val_dataset = medmnist.RetinaMNIST(
            as_rgb=True,
            split="val",
            download=True,
            root=root_dir_val,
        )
    else:
        raise NotImplementedError

    dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
        train_dataset,
        transform=T.Compose([T.Resize(224)]) if input_size == 224 else None,
    )
    dataset_train_probing = lightly.data.LightlyDataset.from_torch_dataset(
        copy.deepcopy(train_dataset), transform=test_transforms
    )
    dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(
        copy.deepcopy(train_dataset), transform=test_transforms
    )
    dataset_test = lightly.data.LightlyDataset.from_torch_dataset(
        test_dataset, transform=test_transforms
    )
else:
    raise ValueError("Unknown dataset name")

if args["dataset"] not in ["medmnist", "ChestMNIST", "RetinaMNIST"]:
    dataset_train_ssl = lightly.data.LightlyDataset(input_dir=path_to_train)
    dataset_train_probing = lightly.data.LightlyDataset(
        input_dir=path_to_train, transform=test_transforms
    )
    # we use test transformations for getting the feature for kNN on train data
    dataset_train_kNN = lightly.data.LightlyDataset(
        input_dir=path_to_train, transform=test_transforms
    )
    dataset_test = lightly.data.LightlyDataset(
        input_dir=path_to_test, transform=test_transforms
    )

try:
    # get the number of classes from the dataset
    classes = len(dataset_train_ssl.dataset.info["label"])
except:
    classes = len(dataset_train_ssl.dataset.classes)

args["num_classes"] = classes


def show_image(s, im=0, inv_normalize=False, times_255=False):
    # plot im1 first image
    if inv_normalize:
        im1 = inv_normalize(s)
    else:
        im1 = copy.deepcopy(s)
    im1 = im1.permute(0, 2, 3, 1)
    im1 = im1.detach().cpu().numpy()
    # im1 = (im1 + 1) / 2
    if times_255:
        im1 = im1 * 255
    im1 = im1.astype(np.uint8)
    im1 = im1[im]
    plt.imshow(im1)
    plt.show()


def get_data_loaders(
        batch_size_train_ssl: int,
        batch_size_train_probing: int,
        batch_size_train_kNN: int,
        batch_size_test: int,
        model,
):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    col_fn = collate_fn
    if model == SwaVModel:
        col_fn = swav_collate_fn
    elif model == DINOModel:
        col_fn = dino_collate_fn
    elif model == MAEModel:
        col_fn = mae_collate_fn
    elif model == SimMIMModel:
        col_fn = mae_collate_fn
    elif model == MSNModel:
        col_fn = msn_collate_fn
    elif model == SMoGModel:
        col_fn = smog_collate_function
    elif model == VICRegLModel:
        col_fn = vicregl_collate_fn
    elif model == vqganMAEModel:
        col_fn = vqgan_collate_fn
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size_train_ssl,
        shuffle=True,
        collate_fn=col_fn,
        drop_last=True,
        num_workers=args["num_workers"],
    )
    dataloader_train_probing = torch.utils.data.DataLoader(
        dataset_train_probing,
        batch_size=batch_size_train_probing,
        shuffle=True,
        # collate_fn=col_fn,
        collate_fn=None,
        drop_last=True,
        num_workers=args["num_workers"],
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size_train_kNN,
        shuffle=False,
        drop_last=False,
        num_workers=args["num_workers"],
        collate_fn=None,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size_test,
        shuffle=False,
        drop_last=False,
        num_workers=args["num_workers"],
        collate_fn=None,
    )

    return (
        dataloader_train_ssl,
        dataloader_train_probing,
        dataloader_train_kNN,
        dataloader_test,
    )


def load_vqgan_model(config_path, checkpoint_path):
    import sys

    sys.path.append("./vqgan/taming_transformers")
    sys.path.append("./vqgan/")
    import taming_transformers as taming
    from taming.models import cond_transformer, vqgan
    from taming import modules
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    if config.model.target == "taming.models.vqgan.VQModel":
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)

    elif config.model.target == "taming.models.vqgan.GumbelVQ":
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)

    elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f"unknown model type: {config.model.target}")
    del model.loss
    return model


class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        # create a ResNet backbone and remove the classification head
        num_splits = 0 if sync_batchnorm else 8
        # TODO: Add split batch norm to the resnet model
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        # create a moco model based on ResNet
        self.projection_head = heads.MoCoProjectionHead(feature_dim, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1, memory_bank_size=args["memory_bank_size"]
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x1_, shuffle = utils.batch_shuffle(x1_, distributed=distributed)
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            x1_ = utils.batch_unshuffle(x1_, shuffle, distributed=distributed)
            return x0_, x1_

        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*step(x0, x1))
        loss_2 = self.criterion(*step(x1, x0))

        loss = 0.5 * (loss_1 + loss_2)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(
            self.projection_head.parameters()
        )
        optim = torch.optim.SGD(
            params,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class SimCLRModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.NTXentLoss(gather_distributed=gather_distributed)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        lr = 6e-2
        if eli:
            lr = 0.075
        optim = torch.optim.SGD(
            self.parameters(), lr=lr * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        if eli:
            self.warmup_epochs = 10
            cosine_scheduler = scheduler.CosineWarmupScheduler(
                optim, self.warmup_epochs, args["max_epochs"]
            )
        else:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, args["max_epochs"]
            )
        return [optim], [cosine_scheduler]


def CLIP_embedding(frames, device, batch_size=64):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    res = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i: i + batch_size]
            inputs = processor(
                text=["a"] * len(batch),
                images=[a for a in batch],
                return_tensors="pt",
                padding=True,
            ).to(device)
            outputs = model(**inputs)
            res.append(outputs["image_embeds"])
            print("batch: ", i, " of ", len(frames))

    frames = torch.cat(res, dim=0)

    print("done")
    return frames


class SLIPModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.NTXentLoss(gather_distributed=gather_distributed)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch

        # clip part
        clip_0 = CLIP_embedding(x0, self.device)
        clip_1 = CLIP_embedding(x1, self.device)
        loss_clip = self.criterion(clip_0, clip_1)

        # simclr
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss_simclr = self.criterion(z0, z1)

        loss = loss_clip + loss_simclr

        self.log("train_loss_simclr", loss_simclr)
        self.log("train_loss_clip", loss_clip)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class SequentialSLIPModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.NTXentLoss(gather_distributed=gather_distributed)

    def set_backbone(self, backbone):
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch

        z0 = self.forward(x0)
        z1 = self.forward(x1)

        # clip part
        clip_0 = CLIP_embedding(x0, self.device)
        clip_1 = CLIP_embedding(x1, self.device)

        loss_clip = self.criterion(clip_0, z0) + self.criterion(clip_1, z1)

        loss = loss_clip

        self.log("train_loss_clip", loss_clip)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class SimSiamModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimSiamProjectionHead(feature_dim, 2048, 2048)
        self.prediction_head = heads.SimSiamPredictionHead(2048, 512, 2048)
        self.criterion = lightly.loss.NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,  # no lr-scaling, results in better training stability
            momentum=0.9,
            weight_decay=5e-4,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class BarlowTwinsModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = heads.BarlowTwinsProjectionHead(feature_dim, 2048, 2048)

        self.criterion = lightly.loss.BarlowTwinsLoss(
            gather_distributed=gather_distributed
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class BYOLModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        # resnet = torchvision.models.resnet18()
        resnet = torchvision.models.resnet50()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        # create a byol model based on ResNet
        self.projection_head = heads.BYOLProjectionHead(feature_dim, 4096, 256)
        self.prediction_head = heads.BYOLPredictionHead(256, 4096, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.NegativeCosineSimilarity()

        if byol_mode == "v1" or byol_mode == "v2" or byol_mode == "v3":
            import clip

            self.clip_model, self.preprocess = clip.load("ViT-B/16", device=self.device)
            utils.deactivate_requires_grad(self.clip_model)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p, y

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z, y

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        utils.update_momentum(
            self.projection_head, self.projection_head_momentum, m=0.99
        )
        (x0, x1), _, _ = batch
        p0, py0 = self.forward(x0)
        z0, zy0 = self.forward_momentum(x0)
        p1, py1 = self.forward(x1)
        z1, zy1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        if byol_mode == "v1":
            x0_clip = self.clip_model.encode_image(x0)
            x1_clip = self.clip_model.encode_image(x1)
            loss += 0.5 * (self.criterion(py0, x0_clip) + self.criterion(py1, x1_clip))
        if byol_mode == "v2":
            x0_clip = self.clip_model.encode_image(x0)
            x1_clip = self.clip_model.encode_image(x1)
            loss += (
                            0.5 * (self.criterion(py0, x1_clip) + self.criterion(py1, x0_clip))
                    ) * 0.1
        if byol_mode == "v3":
            x0_clip = self.clip_model.encode_image(x0)
            x1_clip = self.clip_model.encode_image(x1)
            loss += (
                            0.5 * (self.criterion(py0, x0_clip) + self.criterion(py1, x1_clip))
                    ) * 0.1
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        params = (
                list(self.backbone.parameters())
                + list(self.projection_head.parameters())
                + list(self.prediction_head.parameters())
        )
        optim = torch.optim.SGD(
            params,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class NNCLRModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.NNCLRProjectionHead(feature_dim, 2048, 256)
        self.prediction_head = heads.NNCLRPredictionHead(256, 4096, 256)

        self.criterion = lightly.loss.NTXentLoss()
        self.memory_bank = modules.NNMemoryBankModule(size=4096)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class SwaVModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = heads.SwaVProjectionHead(feature_dim, 2048, 128)
        self.prototypes = heads.SwaVPrototypes(128, 3000)  # use 3000 prototypes

        self.criterion = lightly.loss.SwaVLoss(
            sinkhorn_gather_distributed=gather_distributed
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)

    def training_step(self, batch, batch_idx):
        # normalize the prototypes so they are on the unit sphere
        self.prototypes.normalize()

        # the multi-crop dataloader returns a list of image crops where the
        # first two items are the high resolution crops and the rest are low
        # resolution crops
        multi_crops, _, _ = batch
        multi_crop_features = [self.forward(x) for x in multi_crops]

        # split list of crop features into high and low resolution
        high_resolution_features = multi_crop_features[:2]
        low_resolution_features = multi_crop_features[2:]

        # calculate the SwaV loss
        loss = self.criterion(high_resolution_features, low_resolution_features)

        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=1e-3 * lr_factor,
            weight_decay=1e-6,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class DINOModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        # resnet = torchvision.models.resnet18(pretrained=False)
        # pretrained resnet 18
        # resnet = torchvision.models.resnet18(pretrained=False)
        # resnet = torchvision.models.resnet18()
        resnet = torchvision.models.resnet50()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=True
        )
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=True
        )

        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = lightly.loss.DINOLoss(output_dim=2048)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.99)
        utils.update_momentum(self.head, self.teacher_head, m=0.99)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        param = list(self.backbone.parameters()) + list(self.head.parameters())
        optim = torch.optim.SGD(
            param,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class DCL(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.DCLLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class DCLW(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.DCLWLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


# class MAEModel(BenchmarkModule):
#     def __init__(self, dataloader_kNN, num_classes):
#         super().__init__(dataloader_kNN, num_classes)
#
#         decoder_dim = 512
#         my_patch_size = 16
#         vit = _vision_transformer(
#                     # patch_size=self.patch_size,
#                     patch_size=16,
#                     num_layers=12,
#                     num_heads=12,
#                     hidden_dim=768,
#                     mlp_dim=3072,
#                     progress = True,
#                     weights=None,
#         )
#
#         self.warmup_epochs = 40 if args["max_epochs"] >= 800 else 20
#         self.mask_ratio = masking_ratio
#         # self.patch_size = vit.patch_size
#         # self.patch_size = patch_size
#         self.patch_size = my_patch_size
#         self.sequence_length = vit.seq_length
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
#         self.backbone = MAEBackbone.from_vit(vit)
#         self.decoder = MAEDecoder(
#             seq_length=vit.seq_length,
#             num_layers=1,
#             num_heads=16,
#             embed_input_dim=vit.hidden_dim,
#             hidden_dim=decoder_dim,
#             mlp_dim=decoder_dim * 4,
#             out_dim=my_patch_size**2 * 3,
#             dropout=0,
#             attention_dropout=0,
#         )
#         self.criterion = nn.MSELoss()
#
#     def forward_encoder(self, images, idx_keep=None):
#         return self.backbone.encode(images, idx_keep)
#
#     def forward_decoder(self, x_encoded, idx_keep, idx_mask):
#         # build decoder input
#         batch_size = x_encoded.shape[0]
#         x_decode = self.decoder.embed(x_encoded)
#         x_masked = utils.repeat_token(
#             self.mask_token, (batch_size, self.sequence_length)
#         )
#         x_masked = utils.set_at_index(x_masked, idx_keep, x_decode)
#
#         # decoder forward pass
#         x_decoded = self.decoder.decode(x_masked)
#
#         # predict pixel values for masked tokens
#         x_pred = utils.get_at_index(x_decoded, idx_mask)
#         x_pred = self.decoder.predict(x_pred)
#         return x_pred
#
#     def training_step(self, batch, batch_idx):
#         images, _, _ = batch
#
#         batch_size = images.shape[0]
#         idx_keep, idx_mask = utils.random_token_mask(
#             size=(batch_size, self.sequence_length),
#             mask_ratio=self.mask_ratio,
#             device=images.device,
#         )
#         #TODO: check why [0] is needed
#         x_encoded = self.forward_encoder(images, idx_keep)[0]
#         x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)
#
#         # get image patches for masked tokens
#         patches = utils.patchify(images, self.patch_size)
#         # must adjust idx_mask for missing class token
#         target = utils.get_at_index(patches, idx_mask - 1)
#
#         loss = self.criterion(x_pred, target)
#         self.log('train_loss_ssl', loss)
#         return loss
#
#     def configure_optimizers(self):
#         optim = torch.optim.AdamW(
#             self.parameters(),
#             lr=1.5e-4 * lr_factor,
#             weight_decay=0.05,
#             betas=(0.9, 0.95),
#         )
#         cosine_with_warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, self.scale_lr)
#         return [optim], [cosine_with_warmup_scheduler]
#
#     def scale_lr(self, epoch):
#         if epoch < self.warmup_epochs:
#             return epoch / self.warmup_epochs
#         else:
#             return 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (args["max_epochs"] - self.warmup_epochs)))


class MAEModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )

        decoder_dim = 512
        # vit = torchvision.models.vit_b_32(pretrained=False)

        self.warmup_epochs = 40 if args["max_epochs"] >= 800 else 20
        self.mask_ratio = args["mae_masking_ratio"]
        self.patch_size = args["patch_size"]
        self.sequence_length = (args["input_size"] // self.patch_size) ** 2 + 1
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        # self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.backbone = masked_autoencoder.MAEBackbone(
            image_size=input_size,
            patch_size=self.patch_size,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=768 * 4,
        )
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=self.sequence_length,
            num_layers=4, #TODO: tryout vals here
            num_heads=16,
            embed_input_dim=768,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=self.patch_size ** 2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode)

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        images, _, _ = batch

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)



        # target_img = utils.set_at_index(patches, idx_mask - 1, x_pred)

        loss = self.criterion(x_pred, target)
        self.log("train_loss_ssl", loss)
        if batch_idx == 0:
            # empty patch
            # target_img = utils.set_at_index(patches, idx_mask - 1, torch.zeros_like(patches))
            target_img = utils.set_at_index(patches, idx_mask - 1, torch.zeros_like(patches[:, :idx_mask.shape[1], :]))
            reconstructed_img = utils.set_at_index(target_img, idx_mask - 1, x_pred)

            # test = utils.unpatchify(x=patches, patch_size=self.patch_size)
            # test2 = utils.unpatchify(x=x_pred, patch_size=self.patch_size)
            test_target_img = utils.unpatchify(x=target_img, patch_size=self.patch_size)[0]
            reconstructed_img_unpatched = utils.unpatchify(x=reconstructed_img, patch_size=self.patch_size)[0]
            orginal_img_unpatched = utils.unpatchify(x=patches, patch_size=self.patch_size)[0]

            # show_image(test_target_img, 1, inv_normalize=inv_normalize, times_255=True)
            # show_image(reconstructed_img_unpatched, 1, inv_normalize=inv_normalize, times_255=True)
            # show_image(orginal_img_unpatched, 1, inv_normalize=inv_normalize, times_255=True)

            # orginial, target, reconstructed next to each other
            concat_images = torch.cat((orginal_img_unpatched, test_target_img, reconstructed_img_unpatched), dim=2)
            # show_image(torch.cat((orginal_img_unpatched, test_target_img, reconstructed_img_unpatched), dim=2), 1, inv_normalize=inv_normalize, times_255=True)

            concat_images = concat_images.unsqueeze(0)
            concat_images = inv_normalize(concat_images)
            concat_images = concat_images.permute(0, 2, 3, 1)
            concat_images = concat_images.detach().cpu().numpy()
            concat_images = concat_images * 255
            concat_images = concat_images.astype(np.uint8)
            concat_images = concat_images[0]

            plt.figure()
            plt.imshow(concat_images)
            plt.xlabel('batch')
            plt.ylabel('loss')
            plt.title(f'Loss vs. Batches, epoch {self.current_epoch}')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)

            self.logger.log_image(key='reconstruction',
                                  images=[image]
                                  )

            del buf
            del image
            del concat_images
            del orginal_img_unpatched
            del reconstructed_img_unpatched
            del test_target_img


        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=args['MAE_baseLR'] * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, args["max_epochs"]
        )
        return [optim], [cosine_scheduler]


class vqganMAEModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )

        decoder_dim = 512
        vit = torchvision.models.vit_b_32(pretrained=False)

        self.warmup_epochs = 40 if args["max_epochs"] >= 800 else 20
        # self.mask_ratio = 0.75
        self.mask_ratio = 0.25
        # self.patch_size = vit.patch_size
        self.patch_size = 2
        # self.sequence_length = vit.seq_length
        self.sequence_length = 16
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        # self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.backbone = masked_autoencoder.vqganMAEBackbone(
            image_size=8,  # TODO: rm hardcode
            patch_size=2,  # TODO: check patch sizes
            num_layers=6,
            num_heads=4,
            hidden_dim=256,
            mlp_dim=256 * 4,
        )
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=self.sequence_length,
            num_layers=1,
            num_heads=16,
            # embed_input_dim=vit.hidden_dim,
            embed_input_dim=256,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            # out_dim=self.patch_size ** 2 * 3,
            out_dim=self.patch_size ** 2 * 256,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()

        self.vqganmodel = load_vqgan_model(
            config_path="./vqgan/model.yaml", checkpoint_path="./vqgan/last.ckpt"
        )
        self.vqganmodel.eval()
        self.vqgan_batch_size = 64

        from transformers import ViTConfig, ViTModel

        # Initializing a ViT vit-base-patch16-224 style configuration
        configuration = ViTConfig(
            num_channels=256,
            image_size=14,
            patch_size=4,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            encoder_stride=2,
        )
        self.hidden_vit = ViTModel(configuration)

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode)

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def images_to_codes(self, images):
        codes = []
        for i in range(0, images.shape[0], self.vqgan_batch_size):
            quant, emb_loss, info = self.vqganmodel.encode(
                images[i: i + self.vqgan_batch_size]
            )
            codes.append(quant)
        codes = torch.cat(codes, dim=0)
        # TODO:use post-quantization convolutions????
        if False:
            codes = self.model.post_quant_conv(codes)
        return codes

    def training_step(self, batch, batch_idx):
        # (im1, im2), _, _ = batch
        images, _, _ = batch

        images = self.images_to_codes(images)

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=1.5e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, args["max_epochs"]
        )
        return [optim], [cosine_scheduler]


class contrastMAEModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes, contrastive_type="simclr"):
        super().__init__(dataloader_kNN, num_classes)

        decoder_dim = 512
        vit = torchvision.models.vit_b_32(pretrained=False)

        self.warmup_epochs = 40 if args["max_epochs"] >= 800 else 20
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = MAEBackbone.from_vit(vit)
        self.decoder = MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size ** 2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion_reconstruction = nn.MSELoss()
        self.contrastive_type = contrastive_type
        feature_dim = 768

        if self.contrastive_type == "simclr":
            self.projection_head = heads.SimCLRProjectionHead(
                feature_dim, feature_dim, 128
            )
            self.contrastive_criterion = lightly.loss.NTXentLoss()

            self.collate_fn = lightly.data.SimCLRCollateFunction(
                input_size=feature_dim,
            )

        elif self.contrastive_type == "byol":
            self.projection_head = heads.BYOLProjectionHead(feature_dim, feature_dim)
            self.projection_head = heads.BYOLProjectionHead(feature_dim, 4096, 256)
            self.prediction_head = heads.BYOLPredictionHead(256, 4096, 256)

            self.backbone_momentum = copy.deepcopy(self.backbone)
            self.projection_head_momentum = copy.deepcopy(self.projection_head)

            utils.deactivate_requires_grad(self.backbone_momentum)
            utils.deactivate_requires_grad(self.projection_head_momentum)

            self.contrastive_criterion = lightly.loss.NegativeCosineSimilarity()
        elif self.contrastive_type == "moco":
            self.projection_head = heads.MoCoProjectionHead(feature_dim, 2048, 128)
            self.backbone_momentum = copy.deepcopy(self.backbone)
            self.projection_head_momentum = copy.deepcopy(self.projection_head)
            utils.deactivate_requires_grad(self.backbone_momentum)
            utils.deactivate_requires_grad(self.projection_head_momentum)

            # create our loss with the optional memory bank
            self.contrastive_criterion = lightly.loss.NTXentLoss(
                temperature=0.1, memory_bank_size=args["memory_bank_size"]
            )
        else:
            raise NotImplementedError

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode)

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        if self.contrastive_type == "byol":
            utils.update_momentum(self.backbone, self.backbone_momentum, m=0.99)
            utils.update_momentum(
                self.projection_head, self.projection_head_momentum, m=0.99
            )

            p0 = self.prediction_head(
                self.projection_head(self.backbone(x0).flatten(start_dim=1))
            )
            z0 = self.projection_head_momentum(
                self.backbone_momentum(x0).flatten(start_dim=1)
            ).detach()
            p1 = self.prediction_head(
                self.projection_head(self.backbone(x1).flatten(start_dim=1))
            )
            z1 = self.projection_head_momentum(
                self.backbone_momentum(x1).flatten(start_dim=1)
            ).detach()

            loss_contrastive = 0.5 * (
                    self.contrastive_criterion(p0, z1) + self.contrastive_criterion(p1, z0)
            )
            # x_encoded = self.forward_encoder(x)
            # x_encoded = self.projection_head(x_encoded)
            # x_encoded = x_encoded.reshape(batch_size, 2, -1).mean(dim=1)
            # x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)
        elif self.contrastive_type == "moco":
            # update momentum
            utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
            utils.update_momentum(
                self.projection_head, self.projection_head_momentum, 0.99
            )

            def step(x0_, x1_):
                x1_, shuffle = utils.batch_shuffle(x1_, distributed=distributed)
                x0_ = self.backbone(x0_).flatten(start_dim=1)
                x0_ = self.projection_head(x0_)

                x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
                x1_ = self.projection_head_momentum(x1_)
                x1_ = utils.batch_unshuffle(x1_, shuffle, distributed=distributed)
                return x0_, x1_

            # We use a symmetric loss (model trains faster at little compute overhead)
            # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
            loss_1 = self.contrastive_criterion(*step(x0, x1))
            loss_2 = self.contrastive_criterion(*step(x1, x0))

            loss_contrastive = 0.5 * (loss_1 + loss_2)
            # self.log('train_loss_ssl', loss)

        elif self.contrastive_type == "simclr":
            z0 = self.forward(
                self.projection_head(self.backbone(x0).flatten(start_dim=1))
            )
            z1 = self.forward(
                self.projection_head(self.backbone(x1).flatten(start_dim=1))
            )

            loss_contrastive = self.contrastive_criterion(z0, z1)
            # self.log('train_loss_ssl', loss)

        x = torch.cat([x0, x1], dim=0)
        batch_size = x.shape[0]

        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=x.device,
        )

        x_encoded = self.forward_encoder(x, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(x, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss_reconstruction = self.criterion_reconstruction(x_pred, target)

        # loss = loss_contrastive + loss_reconstruction
        # loss as average of both losses
        loss = 0.5 * (loss_contrastive + loss_reconstruction)

        self.log("train_loss_ssl", loss)
        self.log("train_loss_contrastive", loss_contrastive)
        self.log("train_loss_reconstruction", loss_reconstruction)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=1.5e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_with_warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, self.scale_lr
        )
        return [optim], [cosine_with_warmup_scheduler]

    def scale_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs
        else:
            return 0.5 * (
                    1.0
                    + math.cos(
                math.pi
                * (epoch - self.warmup_epochs)
                / (args["max_epochs"] - self.warmup_epochs)
            )
            )


# class MSNModel(BenchmarkModule):
#     def __init__(self, dataloader_kNN, num_classes):
#         super().__init__(dataloader_kNN, num_classes)
#
#         self.warmup_epochs = 15
#         #  ViT small configuration (ViT-S/16)
#         self.mask_ratio = msn_masking_ratio
#         self.backbone = MAEBackbone(
#             image_size=224,
#             patch_size=16,
#             num_layers=12,
#             num_heads=6,
#             hidden_dim=384,
#             mlp_dim=384 * 4,
#         )
#         self.projection_head = heads.MSNProjectionHead(384)
#
#         self.anchor_backbone = copy.deepcopy(self.backbone)
#         self.anchor_projection_head = copy.deepcopy(self.projection_head)
#
#         utils.deactivate_requires_grad(self.backbone)
#         utils.deactivate_requires_grad(self.projection_head)
#
#         self.prototypes = nn.Linear(384, 1024, bias=False).weight
#         self.criterion = lightly.loss.MSNLoss()
#
#         # self.mask = torch.ones(3, 128, 128, dtype=torch.float, requires_grad=True)
#         # self.mask = torch.rand(128, 64, dtype=torch.float, requires_grad=True)
#         import torch
#         from torchvision import transforms
#         self.freeze_mask_model = True
#         self.pretrained_mask_model = False
#         self.mask = None
#         self.maskmodel_backbone = torchvision.models.mobilenet_v3_small(
#             pretrained=True if self.pretrained_mask_model else False
#         ).features
#         if self.freeze_mask_model:
#             for param in self.maskmodel_backbone.parameters():
#                 param.requires_grad = False
#         self.maskmodel_head = nn.Sequential(
#             # nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(576, 64),
#             nn.ReLU(),
#             nn.LayerNorm(64),
#         )
#         # self.maskmodel_decoder = nn.Sequential(
#
#     def training_step(self, batch, batch_idx):
#         utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
#         utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)
#         loss = 0
#
#         views, _, _ = batch
#         views = [view.to(self.device, non_blocking=True) for view in views]
#         targets = views[0]
#         anchors = views[1]
#         anchors_focal = torch.concat(views[2:], dim=0)
#
#         targets_out = self.backbone(targets)
#         targets_out = self.projection_head(targets_out)
#         anchors_out = self.encode_masked(anchors)
#         anchors_out = self.anchor_projection_head(anchors_out)
#         anchors_focal_out = self.encode_masked(anchors_focal)
#         anchors_focal_out = self.anchor_projection_head(anchors_focal_out)
#         anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)
#         if msn_aug_mode == 'v1':
#             sobel_anchors = filters.sobel(anchors)
#             sobel_anchors_out = self.encode_masked(sobel_anchors)
#             anchors_out = torch.cat([anchors_out, sobel_anchors_out], dim=0)
#         elif msn_aug_mode == 'v2':
#             out_anchors = []
#             for i in range(5):
#                 # TODO: make this learnable?
#                 # TODO: try other ftures of kornia
#                 # TODO: try with random numbers for the kernel, but limited numbers and not too big and dont apply to image
#                 # also try with bigger image in this case
#                 kernel = torch.randn(3, 3, 3, 3).to(self.device)
#                 test = F.conv2d(anchors, kernel, padding=1)
#                 out_anchors.append(test)
#             out_anchors = torch.cat(out_anchors, dim=0)
#             out_anchors_out = self.encode_masked(out_anchors)
#             anchors_out = torch.cat([anchors_out, out_anchors_out], dim=0)
#         elif msn_aug_mode == 'v3':
#             # shuffle anchors on dimension 1
#             permutation = torch.randperm(anchors_out.size(1))
#             # Shuffle the tensor's second dimension according to the permutation
#             shuffled_tensor = anchors_out[:, permutation]
#             anchors_out = torch.cat([anchors_out, shuffled_tensor], dim=0)
#         elif msn_aug_mode == 'v4':
#             # embed targets_blocks, anchors_blocks, anchors_focal_blocks
#             _, targets_blocks = self.backbone.forward_blocks(targets)
#             _, anchors_blocks = self.encode_blocks(anchors)
#             _, anchors_focal_blocks = self.encode_blocks(anchors_focal)
#
#             targets_blocks = [self.projection_head(block) for block in targets_blocks]
#             anchors_blocks = [self.anchor_projection_head(block) for block in anchors_blocks]
#             anchors_focal_blocks = [self.anchor_projection_head(block) for block in anchors_focal_blocks]
#             targets_blocks = torch.cat(targets_blocks, dim=0)
#             anchors_blocks = torch.cat(anchors_blocks, dim=0)
#             anchors_focal_blocks = torch.cat(anchors_focal_blocks, dim=0)
#             anchors_blocks = torch.cat([anchors_blocks, anchors_focal_blocks], dim=0)
#             # anchors_out = torch.cat([anchors_out, targets_blocks, anchors_blocks], dim=0)
#         elif msn_aug_mode == 'v5':
#             _, targets_blocks = self.backbone.forward_blocks(targets)
#             _, anchors_blocks = self.encode_blocks(anchors)
#             _, anchors_focal_blocks = self.encode_blocks(anchors_focal)
#
#             targets_blocks = [self.projection_head(block) for block in targets_blocks]
#             anchors_blocks = [self.anchor_projection_head(block) for block in anchors_blocks]
#             anchors_focal_blocks = [self.anchor_projection_head(block) for block in anchors_focal_blocks]
#             idx = 0
#             for target_block, anchor_block, anchor_focal_block in zip(targets_blocks, anchors_blocks, anchors_focal_blocks):
#                 anchors_block_out = torch.cat([anchor_block, anchor_focal_block], dim=0)
#                 loss += self.criterion(anchors_block_out, target_block, self.prototypes.data) * 0.5 ** idx
#                 idx += 1
#
#         elif msn_aug_mode == 'v6':
#             _, targets_blocks = self.backbone.forward_blocks(targets)
#             _, anchors_blocks = self.encode_blocks(anchors)
#             _, anchors_focal_blocks = self.encode_blocks(anchors_focal)
#
#             targets_blocks = [self.projection_head(block) for block in targets_blocks]
#             anchors_blocks = [self.anchor_projection_head(block) for block in anchors_blocks]
#             anchors_focal_blocks = [self.anchor_projection_head(block) for block in anchors_focal_blocks]
#             idx = 0
#             for target_block, anchor_block, anchor_focal_block in zip(targets_blocks, anchors_blocks, anchors_focal_blocks):
#                 anchors_block_out = torch.cat([anchor_block, anchor_focal_block], dim=0)
#                 loss += self.criterion(anchors_block_out, target_block, self.prototypes.data) * (1 - 0.5 ** idx)
#                 idx += 1
#
#         elif msn_aug_mode == 'v7':
#             _, targets_blocks = self.backbone.forward_blocks(targets)
#             _, anchors_blocks = self.encode_blocks(anchors)
#             _, anchors_focal_blocks = self.encode_blocks(anchors_focal)
#
#             targets_blocks = [self.projection_head(block) for block in targets_blocks]
#             anchors_blocks = [self.anchor_projection_head(block) for block in anchors_blocks]
#             anchors_focal_blocks = [self.anchor_projection_head(block) for block in anchors_focal_blocks]
#             idx = 0
#             num_blocks = len(targets_blocks)
#             for target_block, anchor_block, anchor_focal_block in zip(targets_blocks, anchors_blocks, anchors_focal_blocks):
#                 anchors_block_out = torch.cat([anchor_block, anchor_focal_block], dim=0)
#                 loss += self.criterion(anchors_block_out, target_block, self.prototypes.data) * (idx+1/num_blocks)
#                 idx += 1
#
#         elif msn_aug_mode == 'v8':
#             _, targets_blocks = self.backbone.forward_blocks(targets)
#             _, anchors_blocks = self.encode_blocks(anchors)
#             _, anchors_focal_blocks = self.encode_blocks(anchors_focal)
#
#             targets_blocks = [self.projection_head(block) for block in targets_blocks]
#             anchors_blocks = [self.anchor_projection_head(block) for block in anchors_blocks]
#             anchors_focal_blocks = [self.anchor_projection_head(block) for block in anchors_focal_blocks]
#             idx = 0
#             num_blocks = len(targets_blocks)
#             for target_block, anchor_block, anchor_focal_block in zip(targets_blocks, anchors_blocks, anchors_focal_blocks):
#                 anchors_block_out = torch.cat([anchor_block, anchor_focal_block], dim=0)
#                 for target_block_2, anchor_block_2, anchor_focal_block_2 in zip(targets_blocks, anchors_blocks, anchors_focal_blocks):
#                     if anchor_block_2 is anchor_block:
#                         continue
#                     anchors_block_out = torch.cat([anchors_block_out, anchor_block_2, anchor_focal_block_2], dim=0)
#                 loss += self.criterion(anchors_block_out, target_block, self.prototypes.data) * (idx+1/num_blocks)
#                 idx += 1
#
#         elif msn_aug_mode == 'v9':
#             _, targets_blocks = self.backbone.forward_blocks(targets)
#             _, anchors_blocks = self.encode_blocks(anchors)
#             _, anchors_focal_blocks = self.encode_blocks(anchors_focal)
#
#             targets_blocks = [self.projection_head(block) for block in targets_blocks]
#             anchors_blocks = [self.anchor_projection_head(block) for block in anchors_blocks]
#             anchors_focal_blocks = [self.anchor_projection_head(block) for block in anchors_focal_blocks]
#             idx = 0
#             num_blocks = len(targets_blocks)
#             for target_block, anchor_block, anchor_focal_block in zip(targets_blocks, anchors_blocks, anchors_focal_blocks):
#                 anchors_block_out = torch.cat([anchor_block, anchor_focal_block], dim=0)
#                 for target_block_2, anchor_block_2, anchor_focal_block_2 in zip(targets_blocks, anchors_blocks, anchors_focal_blocks):
#                     loss += self.criterion(anchors_block_out, target_block_2, self.prototypes.data) * (idx+1/num_blocks)
#                 idx += 1
#
#         elif msn_aug_mode == 'v10':
#             # add noise to anchors
#             anchors_out = anchors_out + torch.randn_like(anchors_out) * 0.1
#
#         loss += self.criterion(anchors_out, targets_out, self.prototypes.data)
#         if msn_aug_mode == 'v4':
#             loss += self.criterion(anchors_blocks, targets_blocks, self.prototypes.data)
#         self.log('train_loss_ssl', loss)
#         return loss
#
#     def encode_masked(self, anchors, mask=None):
#         batch_size, _, _, width = anchors.shape
#         seq_length = (width // self.anchor_backbone.patch_size) ** 2
#         if mask is None:
#             idx_keep, _ = learned_token_mask(
#                 size=(batch_size, seq_length),
#                 mask_ratio=self.mask_ratio,
#                 device=self.device,
#                 mask = self.mask,
#             )
#         else:
#             idx_keep, _ = utils.random_token_mask(
#                 size=(batch_size, seq_length),
#                 mask_ratio=self.mask_ratio,
#                 device=self.device,
#             )
#         out = self.anchor_backbone(anchors, idx_keep)
#         return self.anchor_projection_head(out)
#
#     def encode_blocks(self, anchors):
#         batch_size, _, _, width = anchors.shape
#         seq_length = (width // self.anchor_backbone.patch_size) ** 2
#         idx_keep, _ = utils.random_token_mask(
#             size=(batch_size, seq_length),
#             mask_ratio=self.mask_ratio,
#             device=self.device,
#         )
#         out, blocks_out = self.anchor_backbone.forward_blocks(anchors, idx_keep)
#         return out, blocks_out
#
#     def configure_optimizers(self):
#         params = [
#             *list(self.anchor_backbone.parameters()),
#             *list(self.anchor_projection_head.parameters()),
#             self.prototypes,
#         ]
#         optim = torch.optim.AdamW(
#             params=params,
#             lr=1.5e-4 * lr_factor,
#             weight_decay=0.05,
#             betas=(0.9, 0.95),
#         )
#         cosine_scheduler = scheduler.CosineWarmupScheduler(optim, self.warmup_epochs, args["max_epochs"])
#         return [optim], [cosine_scheduler]
class MSNModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )

        self.warmup_epochs = 15
        # ViT small configuration (ViT-S/16)
        self.mask_ratio = args["msn_masking_ratio"]
        self.backbone = masked_autoencoder.MAEBackbone(
            image_size=input_size,
            patch_size=args["patch_size"],
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=384 * 4,
        )
        self.projection_head = heads.MSNProjectionHead(384)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight
        self.criterion = lightly.loss.MSNLoss()

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views, _, _ = batch
        views = [view.to(self.device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.backbone(targets)
        targets_out = self.projection_head(targets_out)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
        self.log("train_loss_ssl", loss)
        return loss

    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=self.device,
        )
        out = self.anchor_backbone(anchors, idx_keep)
        return self.anchor_projection_head(out)

    def configure_optimizers(self):
        params = [
            *list(self.anchor_backbone.parameters()),
            *list(self.anchor_projection_head.parameters()),
            self.prototypes,
        ]
        optim = torch.optim.AdamW(
            params=params,
            lr=1.5e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, args["max_epochs"]
        )
        return [optim], [cosine_scheduler]


from sklearn.cluster import KMeans


class SMoGModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        # create a model based on ResNet
        self.projection_head = heads.SMoGProjectionHead(512, 2048, 128)
        self.prediction_head = heads.SMoGPredictionHead(128, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # smog
        self.n_groups = 300
        memory_bank_size = 10000
        self.memory_bank = lightly.loss.memory_bank.MemoryBankModule(
            size=memory_bank_size
        )
        # create our loss
        group_features = torch.nn.functional.normalize(
            torch.rand(self.n_groups, 128), dim=1
        ).to(self.device)
        self.smog = heads.SMoGPrototypes(group_features=group_features, beta=0.99)
        self.criterion = nn.CrossEntropyLoss()

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def _reset_group_features(self):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        features = self.memory_bank.bank
        group_features = self._cluster_features(features.t())
        self.smog.set_group_features(group_features)

    def _reset_momentum_weights(self):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

    def training_step(self, batch, batch_idx):
        if self.global_step > 0 and self.global_step % 300 == 0:
            # reset group features and weights every 300 iterations
            self._reset_group_features()
            self._reset_momentum_weights()
        else:
            # update momentum
            utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
            utils.update_momentum(
                self.projection_head, self.projection_head_momentum, 0.99
            )

        (x0, x1), _, _ = batch

        if batch_idx % 2:
            # swap batches every second iteration
            x0, x1 = x1, x0

        x0_features = self.backbone(x0).flatten(start_dim=1)
        x0_encoded = self.projection_head(x0_features)
        x0_predicted = self.prediction_head(x0_encoded)
        x1_features = self.backbone_momentum(x1).flatten(start_dim=1)
        x1_encoded = self.projection_head_momentum(x1_features)

        # update group features and get group assignments
        assignments = self.smog.assign_groups(x1_encoded)
        group_features = self.smog.get_updated_group_features(x0_encoded)
        logits = self.smog(x0_predicted, group_features, temperature=0.1)
        self.smog.set_group_features(group_features)

        loss = self.criterion(logits, assignments)

        # use memory bank to periodically reset the group features with k-means
        self.memory_bank(x0_encoded, update=True)

        return loss

    def configure_optimizers(self):
        params = (
                list(self.backbone.parameters())
                + list(self.projection_head.parameters())
                + list(self.prediction_head.parameters())
        )
        optim = torch.optim.SGD(
            params,
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-6,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args["max_epochs"])
        return [optim], [cosine_scheduler]


class SimMIMModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )

        vit = torchvision.models.vit_b_32(pretrained=False)
        self.warmup_epochs = 40 if args["max_epochs"] >= 800 else 20
        decoder_dim = 384
        self.mask_ratio = args["mae_masking_ratio"]
        self.patch_size = args["patch_size"]
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # same backbone as MAE
        self.backbone = masked_autoencoder.MAEBackbone(
            image_size=input_size,
            patch_size=self.patch_size,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=384 * 4,
        )

        # the decoder is a simple linear layer
        self.decoder = nn.Linear(384, self.patch_size ** 2 * 3)

        # L1 loss as paper suggestion
        self.criterion = nn.L1Loss()

    def forward_encoder(self, images, batch_size, idx_mask):
        # pass all the tokens to the encoder, both masked and non masked ones
        tokens = self.backbone.images_to_tokens(images, prepend_class_token=True)
        tokens_masked = utils.mask_at_index(tokens, idx_mask, self.mask_token)
        return self.backbone.encoder(tokens_masked)

    def forward_decoder(self, x_encoded):
        return self.decoder(x_encoded)

    def training_step(self, batch, batch_idx):
        images, _, _ = batch

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        # Encoding...
        x_encoded = self.forward_encoder(images, batch_size, idx_mask)
        x_encoded_masked = utils.get_at_index(x_encoded, idx_mask)

        # Decoding...
        x_out = self.forward_decoder(x_encoded_masked)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)

        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_out, target)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=8e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.999),
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, args["max_epochs"]
        )
        return [optim], [cosine_scheduler]


class VICRegModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = heads.BarlowTwinsProjectionHead(512, 2048, 2048)
        self.criterion = lightly.loss.VICRegLoss()
        self.warmup_epochs = 40 if args["max_epochs"] >= 800 else 20

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        # Training diverges without LARS
        optim = LARS(
            self.parameters(),
            lr=0.3 * lr_factor,
            weight_decay=1e-4,
            momentum=0.9,
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, args["max_epochs"]
        )
        return [optim], [cosine_scheduler]


class VICRegLModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()

        # The train_backbone variable is introduced in order to fit with the
        # structure of BenchmarkModule. During training, train_backbone is used
        # to extract local and global features. Durig evaluation, backbone is used
        # to evaluate global features.
        self.train_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = heads.BarlowTwinsProjectionHead(512, 2048, 2048)
        self.local_projection_head = heads.VicRegLLocalProjectionHead(512, 128, 128)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.criterion = lightly.loss.VICRegLLoss(alpha=0.75, num_matches=(16, 4))
        self.backbone = nn.Sequential(self.train_backbone, self.average_pool)
        self.warmup_epochs = 40 if args["max_epochs"] >= 800 else 20

    def forward(self, x):
        x = self.train_backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        y_local = x.permute(0, 2, 3, 1)  # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)
        return z, z_local

    def training_step(self, batch, batch_index):
        (view_global, view_local, grid_global, grid_local), _, _ = batch
        z_global, z_global_local_features = self.forward(view_global)
        z_local, z_local_local_features = self.forward(view_local)
        loss = self.criterion(
            z_global=z_global,
            z_local=z_local,
            z_global_local_features=z_global_local_features,
            z_local_local_features=z_local_local_features,
            grid_global=grid_global,
            grid_local=grid_local,
        )
        return loss

    def configure_optimizers(self):
        # Training diverges without LARS
        optim = LARS(
            self.parameters(),
            lr=0.3 * lr_factor,
            weight_decay=1e-4,
            momentum=0.9,
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, args["max_epochs"]
        )
        return [optim], [cosine_scheduler]


class TiCoModel(BenchmarkModule):
    def __init__(
            self, dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
    ):
        super().__init__(
            dataloader_kNN, dataloader_train_ssl, dataloader_test, args, num_classes
        )

        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = heads.TiCoProjectionHead(512, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.TiCoLoss()
        self.warmup_epochs = 40 if args["max_epochs"] >= 800 else 20

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        momentum = scheduler.cosine_schedule(self.current_epoch, 10, 0.996, 1)
        utils.update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        utils.update_momentum(
            self.projection_head, self.projection_head_momentum, m=momentum
        )
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        z0 = self.forward(x0)
        z1 = self.forward_momentum(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=0.3 * lr_factor,
            weight_decay=1e-4,
            momentum=0.9,
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, args["max_epochs"]
        )
        return [optim], [cosine_scheduler]


# models = [
#     # MSNModel,
#     # MSNContrastModel,
#     # MSNContrastSimCLRModel,
#
#     # MAEModel, # disabled by default because MAE uses larger images with size 224
#     # MSNModel, # disabled by default because MSN uses larger images with size 224
#     # SimMIMModel, # disabled by default because SimMIM uses larger images with size 224
#
#     # BarlowTwinsModel,
#     BYOLModel,
#     # DCL,
#     # DCLW,
#     DINOModel,
#     # MocoModel,
#     # NNCLRModel,
#     # SimCLRModel,
#     # SimSiamModel,
#     SwaVModel,
#     # SMoGModel,
#     TiCoModel,
#     # VICRegModel,
#     VICRegLModel,
# ]

models = [
    # vqganMAEModel,
    # SLIPModel,
    # vqganMAEModel,
    # SequentialSLIPModel,

    # SimMIMModel,

    # SimSiamModel,
    # SwaVModel,
    # DINOModel,
    # BYOLModel, # bs 256; ft 128
    MAEModel, # bs 256; ft 64
    # MSNModel,
    # SimCLRModel,
    # TiCoModel,
    # VICRegLModel,
]
bench_results = dict()

contrastive_types = [
    "byol",
    "moco",
    "simclr",
]

idx = 0
experiment_version = None
# loop through configurations and train models
for BenchmarkModel in models:
    model_name = BenchmarkModel.__name__.replace("Model", "")
    for contrastive_type in contrastive_types:
        if not "contrast" in model_name and contrastive_type in ["moco", "simclr"]:
            continue
        runs = []

        if model_name == 'MAE' or model_name == 'MSN' or model_name == 'SimMIM':
            # args update model dim
            args.update({"model_dim": 384})
            args.update({"flatten": False})

        # if model_name == 'SimCLR':
        #     # args update model dim
        #     args.update({"model_dim": 1024})

        if "contrast" in model_name:
            model_name = model_name + contrastive_type
        for seed in range(args["n_runs"]):
            if "MSN" in model_name:
                wandb_logger = WandbLogger(
                    project=project_name,
                    entity="maggu",
                    name=f"{model_name}--_{msn_aug_mode}_224_{args['msn_masking_ratio']}_training--{seed}",
                    log_model=log_model,
                )
            else:
                wandb_logger = WandbLogger(
                    project=project_name,
                    entity="maggu",
                    name=f"{model_name}--training--{seed}" + run_name,
                    log_model=log_model,
                )
            # get every key val of args
            wandb_logger.log_hyperparams(args)

            pl.seed_everything(seed)

            (
                dataloader_train_ssl,
                dataloader_train_probing,
                dataloader_train_kNN,
                dataloader_test,
            ) = get_data_loaders(
                batch_size_train_ssl=args["batch_size"],
                batch_size_train_kNN=args["ft_batch_size"],
                batch_size_train_probing=args["ft_batch_size"],
                batch_size_test=args["ft_batch_size"],
                model=BenchmarkModel,
            )
            if "contrast" in model_name:
                benchmark_model = BenchmarkModel(
                    dataloader_train_kNN,
                    dataloader_train_probing,
                    dataloader_test,
                    args=args,
                    num_classes=classes,
                    contrastive_type=contrastive_type,
                )
            elif 'Sequential' in model_name:

                simclr_model = SimCLRModel(
                    dataloader_train_kNN,
                    dataloader_train_probing,
                    dataloader_test,
                    args=args,
                    num_classes=classes,
                )
                trainer = pl.Trainer(
                    max_epochs=args["max_epochs"],
                    gpus=gpus,
                    default_root_dir=logs_root_dir,
                    strategy=distributed_backend,
                    sync_batchnorm=sync_batchnorm,
                    logger=wandb_logger,
                    check_val_every_n_epoch=args["val_epoch"],
                    # accelerator="cpu",
                    # num_processes=0,
                    # callbacks=[checkpoint_callback]
                )
                start = time.time()
                trainer.fit(
                    simclr_model,
                    train_dataloaders=dataloader_train_ssl,
                    val_dataloaders=dataloader_test,
                )

                benchmark_model = BenchmarkModel(
                    dataloader_train_kNN,
                    dataloader_train_probing,
                    dataloader_test,
                    args=args,
                    num_classes=classes,
                )
                benchmark_model.set_backbone(simclr_model.backbone)
                print("done")

            else:
                benchmark_model = BenchmarkModel(
                    dataloader_train_kNN,
                    dataloader_train_probing,
                    dataloader_test,
                    args=args,
                    num_classes=classes,
                )

            # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
            # If multiple runs are specified a subdirectory for each run is created.
            sub_dir = model_name if args["n_runs"] <= 1 else f"{model_name}/run{seed}"
            # logger = TensorBoardLogger(
            #     save_dir=os.path.join(logs_root_dir, args["dataset"] ),
            #     name='',
            #     sub_dir=sub_dir,
            #     version=experiment_version,
            # )
            # if experiment_version is None:
            #     # Save results of all models under same version directory
            #     experiment_version = logger.version
            # checkpoint_callback = pl.callbacks.ModelCheckpoint(
            #     dirpath=os.path.join(logger.log_dir, 'checkpoints')
            # )

            wandb_logger.watch(benchmark_model, log_graph=False, log="all", log_freq=5)

            # trainer = pl.Trainer(
            #     args["max_epochs"]=args["max_epochs"],
            #     accelerator="cpu",
            #     default_root_dir=logs_root_dir,
            #     strategy="dp",
            #     num_processes=0,
            # )
            trainer = pl.Trainer(
                max_epochs=args["max_epochs"],
                gpus=gpus,
                default_root_dir=logs_root_dir,
                strategy=distributed_backend,
                sync_batchnorm=sync_batchnorm,
                logger=wandb_logger,
                check_val_every_n_epoch=args["val_epoch"],
                limit_train_batches=1 if test else None,
                limit_val_batches=1 if test else None,
                accumulate_grad_batches = args["accumulate_grad_batches"],
                # accelerator="cpu",
                # num_processes=0,
                # callbacks=[checkpoint_callback]
            )
            start = time.time()
            trainer.fit(
                benchmark_model,
                train_dataloaders=dataloader_train_ssl,
                val_dataloaders=dataloader_test,
            )

            end = time.time()
            run = {
                "model": model_name,
                "batch_size": args["batch_size"],
                "epochs": args["max_epochs"],
                "max_accuracy": benchmark_model.max_accuracy,
                "runtime": end - start,
                "gpu_memory_usage": torch.cuda.max_memory_allocated(),
                "seed": seed,
            }
            runs.append(run)
            print(run)

            # delete model and trainer + free up cuda memory
            del benchmark_model
            del trainer
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            wandb_logger.experiment.finish()
            # del wandb_logger

        bench_results[model_name] = runs

if False:
    #  print results table
    header = (
        f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
        f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    idx = 0
    for model, results in bench_results.items():
        wandb.init(
            project=project_name,
            entity="maggu",
            # settings=wandb.Settings(start_method="thread"),
            save_code=False,
            name=f"{model}--results",
        )

        idx += 1
        if idx == 5:
            break
        runtime = np.array([result["runtime"] for result in results])
        runtime = runtime.mean() / 60  # convert to min
        accuracy = np.array([result["max_accuracy"] for result in results])
        gpu_memory_usage = np.array([result["gpu_memory_usage"] for result in results])
        gpu_memory_usage = gpu_memory_usage.max() / (1024 ** 3)  #  convert to gbyte

        if len(accuracy) > 1:
            accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
        else:
            accuracy_msg = f"{accuracy.mean():>18.3f}"

        print(
            f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
            f"| {accuracy_msg} | {runtime:>6.1f} Min "
            f"| {gpu_memory_usage:>8.1f} GByte |",
            flush=True,
        )

        wandb.log(
            {
                "model": model,
                "batch_size": batch_size,
                "epochs": max_epochs,
                "max_accuracy": accuracy.mean(),
                "runtime": runtime,
                "gpu_memory_usage": gpu_memory_usage,
            }
        )
        wandb.finish()
        time.sleep(6)
    print("-" * len(header))
