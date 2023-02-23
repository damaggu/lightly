""" Helper modules for benchmarking SSL models """
import copy
import io

import numpy as np
# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from fb_MAE.engine_finetune import train_one_epoch, evaluate
from pl_bolts.optimizers.lars import LARS
from torch._six import inf


# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
# linear probing code adapted from FB research

def knn_predict(feature: torch.Tensor,
                feature_bank: torch.Tensor,
                feature_labels: torch.Tensor,
                num_classes: int,
                knn_k: int = 200,
                knn_t: float = 0.1) -> torch.Tensor:
    """Run kNN predictions on features based on a feature bank

    This method is commonly used to monitor performance of self-supervised
    learning methods.

    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature: 
            Tensor of shape [N, D] for which you want predictions
        feature_bank: 
            Tensor of a database of features used for kNN
        feature_labels: 
            Labels for the features in our feature_bank
        num_classes: 
            Number of classes (e.g. `10` for CIFAR-10)
        knn_k: 
            Number of k neighbors used for kNN
        knn_t: 
            Temperature parameter to reweights similarities for kNN

    Returns:
        A tensor containing the kNN predictions

    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     targets_bank,
        >>>     num_classes=10,
        >>> )
    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(
        feature.size(0), -1), dim=-1, index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(
        0) * knn_k, num_classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(
        0), -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def evaluate_model_linear_probing(
        model,
        data_loader_train,
        data_loader_val,
        device,
        args,
        train_transform=None,
        val_transform=None,
        addition_model=None,
):
    linear_layer = nn.Linear(args["model_dim"], args["num_classes"], bias=True)
    linear_layer.weight.data.normal_(mean=0.0, std=0.01)
    linear_layer.bias.data.zero_()

    # get model name
    # print(model)

    if args['flatten']:
        model.head = nn.Sequential(nn.Flatten(start_dim=1), torch.nn.BatchNorm1d(args["model_dim"], affine=False, eps=1e-6),
                                   linear_layer)
    else:
        model.head = nn.Sequential(torch.nn.BatchNorm1d(args["model_dim"], affine=False, eps=1e-6), linear_layer)
        # model.head = linear_layer

    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    # print("number of params (M): %.2f" % (n_parameters / 1.0e6))
    print("number of params: %.2f" % (n_parameters))

    # TODO: needed?
    # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    eff_batch_size = args["batch_size"]

    if args["lr"] is None:  # only base_lr is specified
        args["lr"] = args["blr"] * eff_batch_size / 256

    print("base lr: %.2e" % (args["lr"] * 256 / eff_batch_size))
    print("actual lr: %.2e" % args["lr"])

    print("accumulate grad iterations: %d" % args["accum_iter"])
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    #     model_without_ddp = model.module

    optimizer = LARS(
        model.head.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay'],
        # momentum=0.9,
    )

    print(optimizer)
    loss_scaler = NativeScalerWithGradNormCount()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # TODO: check if needed
    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    test_stats = evaluate(data_loader_val, model, device, args, transform=val_transform, addition_model=addition_model)
    print(
        f"Accuracy of the network on the {len(data_loader_val)} test images: {test_stats['acc1']:.1f}%"
    )

    print(f"Start training for {args['epochs']} epochs")
    # start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(0, args["epochs"]):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args["clip_grad"],
            transform=train_transform,
            # mixup_fn,
            # log_writer=log_writer,
            args=args,
            addition_model=addition_model,
        )
        # if args.output_dir:
        #     misc.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device, args=args, transform=val_transform)
        print(
            f"Accuracy of the network on the {len(data_loader_val)} test images: {test_stats['acc1']:.1f}%"
        )

    for _, p in model.named_parameters():
        p.requires_grad = True

    return max_accuracy, test_stats["acc1"], test_stats["acc5"], test_stats["loss"]


class BenchmarkModule(LightningModule):
    """A PyTorch Lightning Module for automated kNN callback

    At the end of every training epoch we create a feature bank by feeding the
    `dataloader_kNN` passed to the module through the backbone.
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the
    feature_bank features from the train data.

    We can access the highest test accuracy during a kNN prediction 
    using the `max_accuracy` attribute.

    Attributes:
        backbone:
            The backbone model used for kNN validation. Make sure that you set the
            backbone when inheriting from `BenchmarkModule`.
        max_accuracy:
            Floating point number between 0.0 and 1.0 representing the maximum
            test accuracy the benchmarked model has achieved.
        dataloader_kNN:
            Dataloader to be used after each training epoch to create feature bank.
        num_classes:
            Number of classes. E.g. for cifar10 we have 10 classes. (default: 10)
        knn_k:
            Number of nearest neighbors for kNN
        knn_t:
            Temperature parameter for kNN

    Examples:
        >>> class SimSiamModel(BenchmarkingModule):
        >>>     def __init__(dataloader_kNN, num_classes):
        >>>         super().__init__(dataloader_kNN, num_classes)
        >>>         resnet = lightly.models.ResNetGenerator('resnet-18')
        >>>         self.backbone = nn.Sequential(
        >>>             *list(resnet.children())[:-1],
        >>>             nn.AdaptiveAvgPool2d(1),
        >>>         )
        >>>         self.resnet_simsiam = 
        >>>             lightly.models.SimSiam(self.backbone, num_ftrs=512)
        >>>         self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        >>>
        >>>     def forward(self, x):
        >>>         self.resnet_simsiam(x)
        >>>
        >>>     def training_step(self, batch, batch_idx):
        >>>         (x0, x1), _, _ = batch
        >>>         x0, x1 = self.resnet_simsiam(x0, x1)
        >>>         loss = self.criterion(x0, x1)
        >>>         return loss
        >>>     def configure_optimizers(self):
        >>>         optim = torch.optim.SGD(
        >>>             self.resnet_simsiam.parameters(), lr=6e-2, momentum=0.9
        >>>         )
        >>>         return [optim]
        >>>
        >>> model = SimSiamModel(dataloader_train_kNN)
        >>> trainer = pl.Trainer()
        >>> trainer.fit(
        >>>     model,
        >>>     train_dataloader=dataloader_train_ssl,
        >>>     val_dataloaders=dataloader_test
        >>> )
        >>> # you can get the peak accuracy using
        >>> print(model.max_accuracy)

    """

    def __init__(self,
                 dataloader_kNN: DataLoader,
                 dataloader_train_ssl: DataLoader,
                 dataloader_test: DataLoader,
                 args: dict,
                 num_classes: int,
                 knn_k: int = 200,
                 knn_t: float = 0.1):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.dataloader_train_ssl = dataloader_train_ssl
        self.dataloader_test = dataloader_test
        self.args = args
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t

        # create dummy param to keep track of the device the model is using
        self.dummy_param = nn.Parameter(torch.empty(0))

    def training_epoch_end(self, outputs):
        # update feature bank at the end of each training epoch
        if self.args['do_kNN'] == True:
            self.backbone.eval()
            self.feature_bank = []
            self.targets_bank = []
            with torch.no_grad():
                for data in self.dataloader_kNN:
                    img, target, _ = data
                    img = img.to(self.dummy_param.device)
                    target = target.to(self.dummy_param.device)
                    if self._get_name() == 'vqganMAEModel':
                        img = self.images_to_codes(img)
                    feature = self.backbone(img).squeeze()
                    feature = F.normalize(feature, dim=1)
                    self.feature_bank.append(feature)
                    self.targets_bank.append(target)
            self.feature_bank = torch.cat(
                self.feature_bank, dim=0).t().contiguous()
            self.targets_bank = torch.cat(
                self.targets_bank, dim=0).t().contiguous()
            self.backbone.train()

    def validation_step(self, batch, batch_idx):
        print('validation step')
        # we can only do kNN predictions once we have a feature bank
        if self.args['do_kNN'] == True:
            if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
                images, targets, _ = batch
                if self._get_name() == 'vqganMAEModel':
                    images = self.images_to_codes(images)
                feature = self.backbone(images).squeeze()
                feature = F.normalize(feature, dim=1)
                pred_labels = knn_predict(
                    feature,
                    self.feature_bank,
                    self.targets_bank,
                    self.num_classes,
                    self.knn_k,
                    self.knn_t
                )
                num = images.size()
                top1 = (pred_labels[:, 0] == targets).float().sum()
                return (num, top1)

    def validation_epoch_end(self, outputs):
        current_epoch = self.current_epoch
        device = self.dummy_param.device
        if outputs:
            total_num = torch.Tensor([0]).to(device)
            total_top1 = torch.Tensor([0.]).to(device)
            for (num, top1) in outputs:
                total_num += num[0]
                total_top1 += top1

            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(total_num)
                dist.all_reduce(total_top1)

            acc = float(total_top1.item() / total_num.item())
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            self.log('kNN_accuracy', acc * 100.0, prog_bar=True)
            print(f"Current kNN accuracy: {acc * 100.0:.2f} %")
            # also perform linear probing
            # with torch.no_grad():

        if self.args['do_probing']:
            torch.set_grad_enabled(True)

            # if the model is MAEModel
            if self._get_name() == 'MAEModel':
                save_heads = copy.deepcopy(self.backbone.heads)
                # self.backbone.heads = nn.Identity()
                del self.backbone.heads

            max_accuracy, acc1, _, _ = evaluate_model_linear_probing(self.backbone, self.dataloader_train_ssl,
                                                                     self.dataloader_test, device, self.args,
                                                                     addition_model=None)
            torch.set_grad_enabled(False)
            self.log('linear_probing_accuracy1', acc1, prog_bar=True)
            print(f"Current linear probing accuracy1: {acc1:.2f} %")
            # remove model.head from the backbone
            del self.backbone.head
            if self._get_name() == 'MAEModel':
                self.backbone.heads = save_heads

        if self.args['do_medmnist']:
            torch.set_grad_enabled(True)
            print('uy')
            import medmnist

            task = self.dataloader_test.dataset.dataset.info['task']
            data_flag = self.dataloader_test.dataset.dataset.flag

            # val_evaluator = medmnist.Evaluator(data_flag, 'val
            test_evaluator = medmnist.Evaluator(data_flag, 'test', root=self.dataloader_test.dataset.dataset.root)
            # evaluators = {'val': val_evaluator, 'test': test_evaluator}

            linear_layer = nn.Linear(self.args["model_dim"], self.args["num_classes"], bias=True)
            linear_layer.weight.data.normal_(mean=0.0, std=0.01)
            linear_layer.bias.data.zero_()

            self.backbone.head = nn.Sequential(nn.Flatten(start_dim=1),
                                               torch.nn.BatchNorm1d(self.args["model_dim"], affine=False, eps=1e-6),
                                               linear_layer)

            for _, p in self.backbone.named_parameters():
                p.requires_grad = False
            for _, p in self.backbone.head.named_parameters():
                p.requires_grad = True

            self.backbone.to(self.dummy_param.device)

            if task == "multi-label, binary-class":
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(self.backbone.head.parameters(), lr=self.args['lr_medmnist'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args['milestones_medmnist'],
                                                             gamma=self.args['gamma_medmnist'])

            print('Training the linear head for 10 epochs...')
            total_loss = []
            losses = []
            for epoch in range(self.args['epochs_medmnist']):
                scheduler.step()
                for batch_idx, (inputs, targets, _) in tqdm(enumerate(self.dataloader_train_ssl)):
                    optimizer.zero_grad()
                    inputs = inputs.to(self.dummy_param.device)
                    outputs = self.backbone(inputs).squeeze()
                    # outputs = F.normalize(outputs, dim=1)

                    if task == 'multi-label, binary-class':
                        targets = targets.to(torch.float32).to(self.dummy_param.device)
                        loss = criterion(outputs, targets)
                    else:
                        targets = torch.squeeze(targets, 1).long().to(self.dummy_param.device)
                        loss = criterion(outputs, targets)

                    # log the loss for current_epoch

                    total_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                losses.append(np.mean(total_loss))

            def plot_loss(losses, epoch):
                plt.figure()
                plt.plot(losses)
                plt.xlabel('batch')
                plt.ylabel('loss')
                plt.title(f'Loss vs. Batches, epoch {epoch}')
                # plt to PIL image
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image = Image.open(buf)
                return image

            # log a plot of the loss for current_epoch using matplotlib
            self.logger.log_image(key='loss vs. epochs',
                                  images=[plot_loss(losses, self.current_epoch)]
                                  )

            self.backbone.eval()
            y_score = torch.tensor([]).to(device)
            with torch.no_grad():
                for batch_idx, (inputs, targets, _) in enumerate(self.dataloader_test):
                    inputs = inputs.to(self.dummy_param.device)
                    outputs = self.backbone(inputs).squeeze()
                    # outputs = F.normalize(outputs, dim=1)

                    if task == 'multi-label, binary-class':
                        targets = targets.to(torch.float32).to(self.dummy_param.device)
                        loss = criterion(outputs, targets)
                        m = nn.Sigmoid()
                        outputs = m(outputs).to(self.dummy_param.device)
                    else:
                        targets = torch.squeeze(targets, 1).long().to(self.dummy_param.device)
                        loss = criterion(outputs, targets)
                        m = nn.Softmax(dim=1)
                        outputs = m(outputs).to(self.dummy_param.device)
                        targets = targets.float().resize_(len(targets), 1)

                    y_score = torch.cat((y_score, outputs), 0)

                y_score = y_score.detach().cpu().numpy()
                auc, acc = test_evaluator.evaluate(y_score)
                self.log('medmnist_auc', auc, prog_bar=True)
                self.log('medmnist_acc', acc, prog_bar=True)

            epoch_loss = sum(total_loss) / len(total_loss)
            for _, p in self.backbone.named_parameters():
                p.requires_grad = True
            torch.set_grad_enabled(False)
            del self.backbone.head

            # if the backbone is a vit model, we visualize the attention maps
            if self.backbone.__class__.__name__ == 'VisionTransformer':
                self.backbone.eval()
                for batch_idx, (inputs, targets, _) in enumerate(self.dataloader_test):
                    inputs = inputs.to(self.dummy_param.device)
                    outputs = self.backbone(inputs)
                    break

            self.backbone.train()
