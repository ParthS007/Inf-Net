# -*- coding: utf-8 -*-

"""Preview
Code for NestedUNet (UNet++) training on COVID-19 Lung Infection Segmentation from CT Scans
with GroupNorm (without DP) and optional Morphological Operations

Created on 2025-11-07 (@author: Parth Shandilya)
"""

import torch
import os
import argparse
from datetime import datetime
from Code.utils.dataloader_LungInf import get_loader
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F

# Opacus for GroupNorm conversion
from opacus.validators import ModuleValidator

# Morphology
from kornia.morphology import opening, closing, dilation, erosion

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def apply_kornia_morphology_binary(pred_mask, operation="both", kernel_size=3):
    """Apply morphology to binary predictions"""
    choices = ["open", "close", "both", "none", "dilation", "erosion"]
    if operation not in choices:
        raise ValueError("Operation must be one of 'open', 'close', 'both', or 'none'.")

    kernel = torch.ones(kernel_size, kernel_size).to(pred_mask.device)
    if operation == "dilation":
        refined_mask = dilation(pred_mask, kernel)
    elif operation == "open":
        refined_mask = opening(pred_mask, kernel)
    elif operation == "close":
        refined_mask = closing(pred_mask, kernel)
    elif operation == "erosion":
        refined_mask = erosion(pred_mask, kernel)
    elif operation == "both":
        refined_mask = opening(pred_mask, kernel)
        refined_mask = closing(refined_mask, kernel)
    elif operation == "none":
        refined_mask = pred_mask
    else:
        raise ValueError("open", "close", "both", "none", "dilation", "erosion")
    return refined_mask


def joint_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, opt):
    model.train()
    # ---- multi-scale training ----
    # For faster testing, use only scale 1.0, otherwise use [0.75, 1, 1.25]
    if hasattr(opt, "fast_test") and opt.fast_test:
        size_rates = [1.0]  # Single scale for faster testing
    else:
        size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = images.to(opt.device)
            gts = gts.to(opt.device)

            # ---- rescaling the inputs ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )
                gts = F.interpolate(
                    gts,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )

            # ---- forward ----
            pred = model(images)

            # ---- Handle deep supervision output (list) ----
            if isinstance(pred, list):
                # If deep supervision is enabled, use the final output (last element)
                pred = pred[-1]

            # ---- Apply morphology if enabled ----
            if opt.enable_morphology:
                pred = apply_kornia_morphology_binary(
                    pred,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )

            # ---- loss function ----
            loss = joint_loss(pred, gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        # ---- train logging ----
        if i % 5 == 0 or i == total_step:
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}".format(
                    datetime.now(),
                    epoch,
                    opt.epoch,
                    i,
                    total_step,
                    loss_record.show(),
                )
            )

    # ---- save model ----
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(save_path, f"NestedUNet-{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print("[Saving Snapshot:]", checkpoint_path)


def build_snapshot_path(opt):
    """Build the snapshot save path based on configuration"""
    if opt.is_pseudo and (not opt.is_semi):
        base_path = "NestedUNet_Pseudo"
    elif (not opt.is_pseudo) and opt.is_semi:
        base_path = "Semi-NestedUNet"
    elif (not opt.is_pseudo) and (not opt.is_semi):
        # Determine model type
        if opt.enable_morphology:
            model_type = "NestedUNet_Morph_GroupNorm"
            morph_dir = opt.morph_operation
            batch_dir = f"batch_{opt.batchsize}"
            run_dir = f"run_{opt.run}"
            base_path = os.path.join(model_type, morph_dir, batch_dir, run_dir)
        else:
            model_type = "NestedUNet_GroupNorm"
            batch_dir = f"batch_{opt.batchsize}"
            run_dir = f"run_{opt.run}"
            base_path = os.path.join(model_type, batch_dir, run_dir)
    else:
        # Custom save path
        base_path = opt.train_save

    save_path = os.path.join("./Snapshots/save_weights", base_path)
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument("--epoch", type=int, default=100, help="epoch number")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=24, help="training batch size")
    parser.add_argument(
        "--trainsize", type=int, default=352, help="set the size of training sample"
    )
    parser.add_argument(
        "--clip", type=float, default=0.5, help="gradient clipping margin"
    )
    parser.add_argument(
        "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
    )
    parser.add_argument(
        "--decay_epoch", type=int, default=50, help="every n epochs decay learning rate"
    )
    parser.add_argument(
        "--is_thop",
        type=bool,
        default=False,
        help="whether calculate FLOPs/Params (Thop)",
    )
    parser.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="choose which GPU device you want to use",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers in dataloader. In windows, set num_workers=0",
    )
    # model parameters
    parser.add_argument(
        "--n_classes", type=int, default=1, help="binary segmentation when n_classes=1"
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="input channels (3 for RGB CT images)",
    )
    parser.add_argument(
        "--deep_supervision",
        type=bool,
        default=False,
        help="enable deep supervision for NestedUNet",
    )
    # training dataset
    parser.add_argument(
        "--train_path",
        type=str,
        default="./Dataset/TrainingSet/LungInfection-Train/Doctor-label",
    )
    parser.add_argument(
        "--is_semi",
        type=bool,
        default=False,
        help="if True, you will turn on the mode of `Semi-NestedUNet`",
    )
    parser.add_argument(
        "--is_pseudo",
        type=bool,
        default=False,
        help="if True, you will train the model on pseudo-label",
    )
    parser.add_argument(
        "--train_save",
        type=str,
        default=None,
        help="If you use custom save path, please edit `--is_semi=True` and `--is_pseudo=True`",
    )
    parser.add_argument(
        "--run", type=int, default=1, help="the training iteration number"
    )

    # Morphology arguments
    parser.add_argument(
        "--enable_morphology",
        action="store_true",
        help="Enable morphological operations during training",
    )
    parser.add_argument(
        "--morph_operation",
        type=str,
        default="close",
        choices=["open", "close", "dilation", "erosion", "both"],
        help="Morphological operation to apply",
    )
    parser.add_argument(
        "--morph_kernel_size",
        type=int,
        default=3,
        help="Morphological kernel size (must be odd)",
    )
    parser.add_argument(
        "--fast_test",
        action="store_true",
        help="Fast test mode: disable multi-scale training for faster execution",
    )

    opt = parser.parse_args()

    # ---- setup device ----
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- build models ----
    if opt.device == "cuda":
        torch.cuda.set_device(opt.gpu_device)

    print("Model loading: NestedUNet (UNet++) with GroupNorm")
    from Code.model_lung_infection.InfNet_NestedUNet_GroupNorm import (
        NestedUNet_GroupNorm,
    )

    model = NestedUNet_GroupNorm(
        input_channels=opt.in_channels,
        num_classes=opt.n_classes,
        deep_supervision=opt.deep_supervision,
    ).to(opt.device)

    # ---- Convert BatchNorm to GroupNorm (same as DP training) ----
    print("Converting BatchNorm to GroupNorm for fair comparison with DP models")
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
        print("Model architecture converted to GroupNorm")
    else:
        print("Model already GroupNorm-compatible")

    # Move model back to device after conversion
    model = model.to(opt.device)

    # ---- load pre-trained weights ----
    # NestedUNet doesn't have pre-trained weights for COVID dataset
    print("Not loading weights from weights file (training from scratch)")

    # ---- calculate FLOPs and Params ----
    if opt.is_thop:
        from Code.utils.utils import CalParams

        x = torch.randn(1, opt.in_channels, opt.trainsize, opt.trainsize).to(opt.device)
        CalParams(model, x)

    # ---- Create optimizer ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    # ---- Load data ----
    image_root = "{}/Imgs/".format(opt.train_path)
    gt_root = "{}/GT/".format(opt.train_path)
    edge_root = "{}/Edge/".format(opt.train_path)

    train_loader = get_loader(
        image_root,
        gt_root,
        edge_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=opt.num_workers,
    )
    total_step = len(train_loader)

    # ---- Build save path ----
    save_path = build_snapshot_path(opt)

    # ---- Print training info ----
    morph_info = (
        f"Morphology: {opt.morph_operation}"
        if opt.enable_morphology
        else "Morphology: Disabled"
    )
    print(
        "#" * 50,
        "\nStart Training (NestedUNet-GroupNorm)\n"
        "Input Channels: {}\n"
        "Output Classes: {}\n"
        "Deep Supervision: {}\n"
        "Batch Size: {}\n"
        "{}\n"
        "Architecture: GroupNorm (no DP)\n"
        "Save Path: {}\n"
        "Run: {}\n"
        "{}\n".format(
            opt.in_channels,
            opt.n_classes,
            opt.deep_supervision,
            opt.batchsize,
            morph_info,
            save_path,
            opt.run,
            opt,
        ),
        "#" * 50,
    )

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, save_path, opt)
