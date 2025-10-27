# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
with optional Differential Privacy and Morphological Operations
submit to Transactions on Medical Imaging, 2020.

1st Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
2nd Version: Fix some bugs caused by THOP on 2020-06-10 (@author: Ge-Peng Ji)
3rd Version: Add differential privacy training on 2025-10-12 (@author: Parth Shandilya)
4th Version: Unified DP+Morph with batch sizes and epsilon on 2025-10-23
"""

import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from Code.utils.dataloader_LungInf import get_loader
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np

# Differential Privacy
from opacus.privacy_engine import PrivacyEngine
from opacus.validators import ModuleValidator

# Morphology
from kornia.morphology import opening, closing, dilation, erosion

# Improve CUDA memory behavior to reduce fragmentation (harmless if re-set)
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


def train(
    train_loader, model, optimizer, epoch, save_path, privacy_engine=None, opt=None
):
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = (
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
    )
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = images.to(opt.device)
            gts = gts.to(opt.device)
            edges = edges.to(opt.device)

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
                edges = F.interpolate(
                    edges,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )

            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = (
                model(images)
            )

            # ---- Apply morphology if enabled ----
            if opt.enable_morphology:
                lateral_map_5 = apply_kornia_morphology_binary(
                    lateral_map_5,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )
                lateral_map_4 = apply_kornia_morphology_binary(
                    lateral_map_4,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )
                lateral_map_3 = apply_kornia_morphology_binary(
                    lateral_map_3,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )
                lateral_map_2 = apply_kornia_morphology_binary(
                    lateral_map_2,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )

            # ---- loss function ----
            loss5 = joint_loss(lateral_map_5, gts)
            loss4 = joint_loss(lateral_map_4, gts)
            loss3 = joint_loss(lateral_map_3, gts)
            loss2 = joint_loss(lateral_map_2, gts)
            loss1 = BCE(lateral_edge, edges)
            loss = loss1 + loss2 + loss3 + loss4 + loss5

            # ---- backward ----
            loss.backward()
            # clip_gradient(optimizer, opt.clip)
            # ---- optimizer step ----
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)

        # ---- train logging ----
        if i % 20 == 0 or i == total_step:
            epsilon = 0.0
            if privacy_engine:
                epsilon = privacy_engine.get_epsilon(delta=opt.delta)
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-edge: {:.4f}, "
                "lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}, epsilon: {:0.4f}]".format(
                    datetime.now(),
                    epoch,
                    opt.epoch,
                    i,
                    total_step,
                    loss_record1.show(),
                    loss_record2.show(),
                    loss_record3.show(),
                    loss_record4.show(),
                    loss_record5.show(),
                    epsilon,
                )
            )

    # ---- save model ----
    os.makedirs(save_path, exist_ok=True)

    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(save_path, f"Inf-Net-{epoch+1}.pth")
        if opt.enable_privacy:
            torch.save(model._module.state_dict(), checkpoint_path)
        print("[Saving Snapshot:]", checkpoint_path)
        if privacy_engine:
            epsilon = privacy_engine.get_epsilon(delta=opt.delta)
            print(f"[Privacy Budget]: ε = {epsilon:.2f} (δ = {opt.delta})")


def build_snapshot_path(opt):
    """Build the snapshot save path based on configuration"""
    if opt.is_pseudo and (not opt.is_semi):
        base_path = "Inf-Net_Pseudo"
    elif (not opt.is_pseudo) and opt.is_semi:
        base_path = "Semi-Inf-Net"
    elif (not opt.is_pseudo) and (not opt.is_semi):
        # Determine model type and structure
        # Base: Inf-Net_DP or Inf-Net_DP_Morph
        model_prefix = "Inf-Net_DP"

        # Add morphology info if enabled
        if opt.enable_morphology:
            model_type = f"{model_prefix}_Morph"
            morph_dir = opt.morph_operation
            batch_dir = f"batch_{opt.batchsize}"
            run_dir = f"run_{opt.run}"
            base_path = os.path.join(model_type, morph_dir, batch_dir, run_dir)
        else:
            model_type = model_prefix
            batch_dir = f"batch_{opt.batchsize}"
            run_dir = f"run_{opt.run}"
            base_path = os.path.join(model_type, batch_dir, run_dir)

        noise_multiplier = f"noise_multiplier_{opt.noise_multiplier}"
        base_path = os.path.join(base_path, noise_multiplier)
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
        default=4,
        help="number of workers in dataloader. In windows, set num_workers=0",
    )
    # model parameters
    parser.add_argument(
        "--net_channel",
        type=int,
        default=32,
        help="internal channel numbers in the Inf-Net, default=32, try larger for better accuracy",
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="binary segmentation when n_classes=1"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="Res2Net50",
        help="change different backbone, choice: VGGNet16, ResNet50, Res2Net50",
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
        help="if True, you will turn on the mode of `Semi-Inf-Net`",
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

    # Privacy related arguments
    parser.add_argument(
        "--enable_privacy",
        action="store_true",
        help="Enable differential privacy training",
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=1.1,
        help="Noise multiplier for DP-SGD",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Target privacy parameter (delta)",
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
        choices=["open", "close", "dilation", "erosion", "both", "none"],
        help="Morphological operation to apply",
    )
    parser.add_argument(
        "--morph_kernel_size",
        type=int,
        default=3,
        help="Morphological kernel size (must be odd)",
    )

    opt = parser.parse_args()

    # ---- setup device ----
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- build models ----
    if opt.device == "cuda":
        torch.cuda.set_device(opt.gpu_device)

    if opt.backbone == "Res2Net50":
        print("Backbone loading: Res2Net50")
        from Code.model_lung_infection.InfNet_Res2Net import Inf_Net
    elif opt.backbone == "ResNet50":
        print("Backbone loading: ResNet50")
        from Code.model_lung_infection.InfNet_ResNet import Inf_Net
    elif opt.backbone == "VGGNet16":
        print("Backbone loading: VGGNet16")
        from Code.model_lung_infection.InfNet_VGGNet import Inf_Net
    else:
        raise ValueError("Invalid backbone parameters: {}".format(opt.backbone))

    model = Inf_Net(channel=opt.net_channel, n_class=opt.n_classes).to(opt.device)

    print("Freezing unused branches for DP compatibility...")
    if opt.enable_privacy:
        if opt.backbone == "Res2Net50":
            if hasattr(model.resnet, "avgpool"):
                for param in model.resnet.avgpool.parameters():
                    param.requires_grad = False
            if hasattr(model.resnet, "fc"):
                for param in model.resnet.fc.parameters():
                    param.requires_grad = False
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable: {trainable_params:,} / {total_params:,} parameters")

    # ---- load pre-trained weights ----
    if opt.is_semi and opt.backbone == "Res2Net50":
        print("Loading weights from weights file trained on pseudo label")
        model.load_state_dict(
            torch.load("./Snapshots/save_weights/Inf-Net_Pseduo/Inf-Net_pseudo_100.pth")
        )
    else:
        print("Not loading weights from weights file")

    # ---- calculate FLOPs and Params ----
    if opt.is_thop:
        from Code.utils.utils import CalParams

        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).to(opt.device)
        CalParams(model, x)

    # ---- load training sub-modules ----
    BCE = torch.nn.BCEWithLogitsLoss()

    # ---- Fix model for differential privacy BEFORE creating optimizer ----
    if opt.enable_privacy:
        try:
            ModuleValidator.validate(model, strict=True)
            print("Model is compatible with differential privacy")
        except Exception as e:
            print(f"Validator raised issues with model. Attempting auto-fix...")
            print(f"Original error: {type(e).__name__}")
            model = ModuleValidator.fix(model)
            model = model.to(opt.device)
            try:
                ModuleValidator.validate(model, strict=True)
                print("Model successfully fixed for differential privacy")
            except Exception as validation_error:
                print(
                    f"Warning: Model may still have compatibility issues: {validation_error}"
                )

    # ---- Create optimizer AFTER model fix ----
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

    # ---- Setup privacy engine ----
    privacy_engine = None
    if opt.enable_privacy:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=opt.epoch,
            noise_multiplier=opt.noise_multiplier,
            max_grad_norm=opt.max_grad_norm,
        )

    # ---- Print training info ----
    privacy_info = (
        f"DP Enabled (ε budget, δ={opt.delta})"
        if opt.enable_privacy
        else "DP: Disabled"
    )
    morph_info = (
        f"Morphology: {opt.morph_operation}"
        if opt.enable_morphology
        else "Morphology: Disabled"
    )

    print(
        "#" * 70,
        "\nStart Training (Inf-Net-{})\n"
        "Backbone: {}\n"
        "Batch Size: {}\n"
        "Privacy: {}\n"
        "{}\n"
        "Save Path: {}\n"
        "Run: {}\n"
        "{}\n".format(
            opt.backbone,
            opt.backbone,
            opt.batchsize,
            privacy_info,
            morph_info,
            save_path,
            opt.run,
            opt,
        ),
        "#" * 70,
    )

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, save_path, privacy_engine, opt)
