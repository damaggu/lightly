### finetune a model from a checkpoint

import os
import pickle
import sys

import PIL
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import create_transform
import torchvision.transforms as T

from wandb import api

import lightly
import wandb
import copy

# wandb offline
# os.environ['WANDB_MODE'] = 'offline'
from imagenette_benchmark_contrastMAE import MAEModel, BYOLModel, get_data_loaders
from lightly.utils.benchmarking import evaluate_model_linear_probing

# a = wandb.restore("model.pt", run_path="username/project/run_id")


logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")
eli = False
dist = False
test = False
args = {}
args["dataset"] = "imagenette"

if args["dataset"] == "cifar10" or args["dataset"] == "imagenette":
    # input_size = 224
    input_size = 112
elif args["dataset"] in ["iNat2021mini", "inat_birds"]:
    # input_size = 224
    input_size = 112
elif args["dataset"] in ["ChestMNIST", "RetinaMNIST", "BreastMNIST"]:
    input_size = 28
    # input_size = 224
else:
    raise ValueError("Invalid dataset name")

args["input_size"] = input_size
args['flatten'] = True
args["num_workers"] = 3
args["memory_bank_size"] = 4096
if eli:
    args["batch_size"] = 4096
    args["max_epochs"] = 1000
    args["val_epoch"] = 50
else:
    args["max_epochs"] = 800
    args["val_epoch"] = 10
    if input_size == 224:
        args["batch_size"] = 128 if dist else 128
    else:
        args["batch_size"] = 4096 if dist else 128
args['MAE_collate_type'] = 'normal'
args['MAE_baseLR'] = 1.5e-4
args['accumulate_grad_batches'] = 8
args["effective_bs"] = args["batch_size"] * args['accumulate_grad_batches']

ratio = input_size / 224
# args["ft_batch_size"] = 256 if dist else 1024
# args["ft_batch_size"] = 4096
args["ft_batch_size"] = 16
args["warmup_epochs"] = 20
args["mae_masking_ratio"] = 0.75
args["patch_size"] = 16
args["patch_size"] = int(args["patch_size"] * ratio)

# vit settings
args["vit_name"] = "vit_base"

if args["vit_name"] == "vit_base":
    args["vit_dim"] = 768
    args["vit_depth"] = 12
    args["vit_heads"] = 12
    args["vit_mlp_dim"] = 4 * args["vit_dim"]
    args["vit_decoder_dim"] = 512
    args["vit_decoder_layers"] = 4
    args["vit_decoder_heads"] = 8
elif args["vit_name"] == "vit_small":
    args["vit_dim"] = 384
    args["vit_depth"] = 8
    args["vit_heads"] = 6
    args["vit_mlp_dim"] = 4 * args["vit_dim"]
    args["vit_decoder_dim"] = 256
    args["vit_decoder_layers"] = 2
    args["vit_decoder_heads"] = 4
elif args["vit_name"] == "vit_tiny":
    args["vit_dim"] = 192
    args["vit_depth"] = 4
    args["vit_heads"] = 3
    args["vit_mlp_dim"] = 4 * args["vit_dim"]
    args["vit_decoder_dim"] = 128
    args["vit_decoder_layers"] = 1
    args["vit_decoder_heads"] = 2
elif args["vit_name"] == "deprecated_vit_tiny":
    args["vit_dim"] = 384
    args["vit_depth"] = 12
    args["vit_heads"] = 8
    args["vit_mlp_dim"] = 4 * args["vit_dim"]
    args["vit_decoder_dim"] = 512
    args["vit_decoder_layers"] = 1
    args["vit_decoder_heads"] = 16
elif args["vit_name"] == "deprecated_vit_base":
    args["vit_dim"] = 768
    args["vit_depth"] = 12
    args["vit_heads"] = 12
    args["vit_mlp_dim"] = 4 * args["vit_dim"]
    args["vit_decoder_dim"] = 512
    args["vit_decoder_layers"] = 1
    args["vit_decoder_heads"] = 16
else:
    raise ValueError("Invalid vit name")

gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
if dist:
    args["gpus"] = gpus
    args['batch_size'] = args['batch_size'] * gpus
    args['ft_batch_size'] = args['ft_batch_size'] * gpus

args["msn_masking_ratio"] = 0.15
args['finetune_only'] = True
args["do_probing"] = False
args["do_kNN"] = True
args["do_medmnist"] = False
args["knn_k"] = 200
args["knn_t"] = 0.1
args["n_runs"] = 1
if args["dataset"] in ["ChestMNIST", "RetinaMNIST", "BreastMNIST"]:
    args["do_medmnist"] = True
    args["ft_batch_size"] = 8192 if dist else 4096
    # args["max_epochs"] = 50
    # args["val_epoch"] = 5
    # mae_masking_ratio = 0.5
    # msn_masking_ratio = 0.15
    # patch_size = 2
args["epochs_medmnist"] = 100
args["lr_medmnist"] = 0.1
args["gamma_medmnist"] = 0.1
args["milestones_medmnist"] = [
    0.5 * args["epochs_medmnist"],
    0.75 * args["epochs_medmnist"],
]
args["weight_decay"] = 0
args["blr"] = 0.1
args["accum_iter"] = 4
args['effective_bs'] = args['ft_batch_size'] * args['accum_iter']
args['ft_lr_factor'] = args["ft_batch_size"] / 256
args["lr"] = args["blr"] * args["ft_lr_factor"]
args["epochs"] = 300
args["clip_grad"] = 1.0
args["model_dim"] = 2048
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
project_name = args["dataset"] + "fine_tune"
log_model = 'all'

# benchmark

# use a GPU if available


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

local_patch_size = int(input_size * ratio)
ratio2 = 256 / 224
resize_size = int(input_size * ratio2)

if args["dataset"] in ["iNat2021mini", "inat_birds"]:
    # imagenet
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_size,
    )
else:
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_size,
        gaussian_blur=0.0,  # from eli's paper
    )

# Multi crop augmentation for SwAV
swav_collate_fn = lightly.data.SwaVCollateFunction(
    crop_sizes=[input_size, local_patch_size],
    crop_counts=[2, 6],  # 2 crops @ 128x128px and 6 crops @ 64x64px
)

# Multi crop augmentation for DINO, additionally, disable blur for cifar10
dino_collate_fn = lightly.data.DINOCollateFunction(
    global_crop_size=input_size,
    local_crop_size=local_patch_size,
)

# Two crops for SMoG
smog_collate_function = lightly.data.collate.SMoGCollateFunction(
    crop_sizes=[input_size, local_patch_size],
    crop_counts=[1, 1],
    crop_min_scales=[0.2, 0.2],
    crop_max_scales=[1.0, 1.0],
)
# Collate function passing geometrical transformation for VICRegL
vicregl_collate_fn = lightly.data.VICRegLCollateFunction(
    global_crop_size=input_size, local_crop_size=local_patch_size, global_grid_size=4, local_grid_size=2
)
msn_collate_fn = lightly.data.MSNCollateFunction(random_size=input_size, focal_size=local_patch_size)
# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(resize_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        normalize_transform,
    ]
)

vqgan_collate_fn = lightly.data.MAECollateFunction(normalize=None, input_size=input_size)


args['color_jitter'] = 0.4
args['aa'] = 'v0'
args['reprob'] = 0.25
args['remode'] = 'pixel'
args['recount'] = 1

def build_transform(is_train, args):
    mean = lightly.data.collate.imagenet_normalize["mean"]
    std = lightly.data.collate.imagenet_normalize["std"]
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args['input_size'],
            is_training=True,
            color_jitter=args['color_jitter'],
            auto_augment=args['aa'],
            interpolation='bicubic',
            re_prob=args['reprob'],
            re_mode=args['remode'],
            re_count=args['recount'],
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args['input_size'] <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args['input_size'] / crop_pct)
    t.append(
        # T.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        T.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(T.CenterCrop(args['input_size']))

    t.append(T.ToTensor())
    t.append(T.Normalize(mean, std))
    return T.Compose(t)


train_transform = build_transform(is_train=True, args=args)
test_transform = build_transform(is_train=False, args=args)

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
        transform=train_transform,
    )
    dataset_train_probing = lightly.data.LightlyDataset.from_torch_dataset(
        copy.deepcopy(train_dataset), transform=test_transforms
    )
    dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(
        copy.deepcopy(train_dataset), transform=test_transforms
    )
    dataset_test = lightly.data.LightlyDataset.from_torch_dataset(
        test_dataset, transform=test_transform
    )
else:
    raise ValueError("Unknown dataset name")

if args["dataset"] not in ["medmnist", "ChestMNIST", "RetinaMNIST"]:
    dataset_train_ssl = lightly.data.LightlyDataset(input_dir=path_to_train, transform=train_transform)
    dataset_train_probing = lightly.data.LightlyDataset(
        input_dir=path_to_train, transform=train_transform
    )
    # we use test transformations for getting the feature for kNN on train data
    dataset_train_kNN = lightly.data.LightlyDataset(
        input_dir=path_to_train, transform=test_transform
    )
    dataset_test = lightly.data.LightlyDataset(
        input_dir=path_to_test, transform=test_transform
    )

try:
    # get the number of classes from the dataset
    classes = len(dataset_train_ssl.dataset.info["label"])
except:
    classes = len(dataset_train_ssl.dataset.classes)

args["num_classes"] = classes

# load a model from wandb


# a = wandb.restore(
#     model_path,
#     replace=True,
#     root="./temp",
# )

run = wandb.init()
args['run_name'] = "sa3rkof5"
args['model_name'] = 'MAE'
args['load_pretrained'] = False
args['linear_probing'] = False
args["model_dim"] = args["vit_dim"]

artifact = run.use_artifact('maggu/imagenette_benchmark_Tiny/model-' + args['run_name'] + ':v58', type='model')
artifact_dir = artifact.download()
model_artifact = torch.load(artifact_dir + '/model.ckpt')
# get config from run

api = wandb.Api()
run = api.run("maggu/imagenette_benchmark_Tiny/" + args['run_name'])
config = run.config
state_dict = model_artifact['state_dict']

dataloader_train_ssl = torch.utils.data.DataLoader(
    dataset_train_ssl,
    batch_size=args["ft_batch_size"],
    shuffle=True,
    # collate_fn=col_fn,
    drop_last=True,
    num_workers=args["num_workers"],
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args["ft_batch_size"],
    shuffle=False,
    drop_last=False,
    num_workers=args["num_workers"],
    collate_fn=None,
)

if args['model_name'] == "BYOL":
    model = BYOLModel(
        dataloader_train_ssl,
        dataloader_train_ssl,
        dataloader_test,
        args=args,
        num_classes=classes,
    )
elif args['model_name'] == "MAE":

    model = MAEModel(dataloader_train_ssl,
                     dataloader_train_ssl,
                     dataloader_test,
                     args=args,
                     num_classes=classes,
                     )
else:
    raise ValueError("Unknown model name")

#
# model.load_from_checkpoint(torch.load(artifact_dir + '/model.ckpt'))

if args['load_pretrained']:
    model.load_state_dict(state_dict)
    args["model_dim"] = run.config["model_dim"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if the model is MAEModel
if model._get_name() == 'MAEModel' and args['load_pretrained'] and args['linear_probing']:
    save_heads = copy.deepcopy(model.backbone.heads)
    # self.backbone.heads = nn.Identity()
    del model.backbone.heads

# do new wandb init for linear probing
wandb.init(
    project="imagenette_benchmark_eval",
    entity="maggu",
    # settings=wandb.Settings(start_method="thread"),
    save_code=False,
    name=args['run_name'] + "_linear_probing",
)
wandb.config.update(args)
wandb.watch(model, log="all")

max_accuracy, acc1, _, _ = evaluate_model_linear_probing(model.backbone, model.dataloader_train_ssl,
                                                         model.dataloader_test, device, args,
                                                         addition_model=None, linear_probing=args['linear_probing'])
