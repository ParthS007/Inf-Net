# -*- coding: utf-8 -*-

"""Preview
Code for testing UNet and NestedUNet models on COVID-19 Lung Infection Segmentation
Automatically finds latest snapshots and saves results respecting current structure

Created on 2025-11-07 (@author: Parth Shandilya)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import imageio
import glob
from Code.utils.dataloader_LungInf import test_dataset
from opacus.validators import ModuleValidator


def find_latest_snapshot(snapshot_dir, model_name):
    """
    Find the latest (highest epoch) snapshot in a directory
    
    Args:
        snapshot_dir: Directory containing snapshots
        model_name: Model name prefix (e.g., 'UNet', 'NestedUNet')
    
    Returns:
        Path to latest snapshot or None if not found
    """
    if not os.path.exists(snapshot_dir):
        return None
    
    # Find all snapshots matching pattern: ModelName-*.pth
    pattern = os.path.join(snapshot_dir, f"{model_name}-*.pth")
    snapshots = glob.glob(pattern)
    
    if not snapshots:
        return None
    
    # Extract epoch numbers and find maximum
    def get_epoch(path):
        filename = os.path.basename(path)
        # Extract number from "ModelName-XX.pth"
        try:
            epoch = int(filename.replace(f"{model_name}-", "").replace(".pth", ""))
            return epoch
        except ValueError:
            return -1
    
    # Sort by epoch number and return latest
    latest_snapshot = max(snapshots, key=get_epoch)
    latest_epoch = get_epoch(latest_snapshot)
    
    print(f"Found latest snapshot: {os.path.basename(latest_snapshot)} (epoch {latest_epoch})")
    return latest_snapshot, latest_epoch


def build_model_path(opt):
    """Build the model path based on configuration (matching snapshot structure)"""
    base_path = "./Snapshots/save_weights"
    
    if opt.model_type == "UNet_GroupNorm":
        # Structure: UNet_GroupNorm/batch_X/run_Y/UNet-Z.pth
        snapshot_dir = os.path.join(
            base_path,
            "UNet_GroupNorm",
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
        )
        model_name = "UNet"
    elif opt.model_type == "UNet_Morph_GroupNorm":
        # Structure: UNet_Morph_GroupNorm/morph_op/batch_X/run_Y/UNet-Z.pth
        snapshot_dir = os.path.join(
            base_path,
            "UNet_Morph_GroupNorm",
            opt.morph_operation,
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
        )
        model_name = "UNet"
    elif opt.model_type == "NestedUNet_GroupNorm":
        # Structure: NestedUNet_GroupNorm/batch_X/run_Y/NestedUNet-Z.pth
        snapshot_dir = os.path.join(
            base_path,
            "NestedUNet_GroupNorm",
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
        )
        model_name = "NestedUNet"
    elif opt.model_type == "NestedUNet_Morph_GroupNorm":
        # Structure: NestedUNet_Morph_GroupNorm/morph_op/batch_X/run_Y/NestedUNet-Z.pth
        snapshot_dir = os.path.join(
            base_path,
            "NestedUNet_Morph_GroupNorm",
            opt.morph_operation,
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
        )
        model_name = "NestedUNet"
    else:
        # Custom path
        if opt.pth_path:
            return opt.pth_path, None
        else:
            raise ValueError(f"Unknown model_type: {opt.model_type}")
    
    # Find latest snapshot if epoch not specified
    if opt.epoch is None or opt.epoch == -1:
        result = find_latest_snapshot(snapshot_dir, model_name)
        if result is None:
            raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")
        model_path, latest_epoch = result
        return model_path, latest_epoch
    else:
        # Use specified epoch
        model_path = os.path.join(snapshot_dir, f"{model_name}-{opt.epoch}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Snapshot not found: {model_path}")
        return model_path, opt.epoch


def build_result_path(opt, epoch):
    """Build the result save path based on configuration (matching snapshot structure)"""
    base_path = "./Results/Lung_infection_segmentation"
    
    if opt.model_type == "UNet_GroupNorm":
        # Structure: UNet_GroupNorm/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "UNet_GroupNorm",
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
        )
    elif opt.model_type == "UNet_Morph_GroupNorm":
        # Structure: UNet_Morph_GroupNorm/morph_op/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "UNet_Morph_GroupNorm",
            opt.morph_operation,
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
        )
    elif opt.model_type == "NestedUNet_GroupNorm":
        # Structure: NestedUNet_GroupNorm/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "NestedUNet_GroupNorm",
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
        )
    elif opt.model_type == "NestedUNet_Morph_GroupNorm":
        # Structure: NestedUNet_Morph_GroupNorm/morph_op/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "NestedUNet_Morph_GroupNorm",
            opt.morph_operation,
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
        )
    else:
        # Custom path
        result_path = opt.save_path if opt.save_path else base_path
    
    # Create directory
    os.makedirs(result_path, exist_ok=True)
    return result_path


def load_model(model_path, model_type, device="cuda"):
    """
    Load model with correct architecture based on model type
    
    Args:
        model_path: Path to checkpoint file
        model_type: One of 'UNet_GroupNorm', 'NestedUNet_GroupNorm', etc.
        device: Device to load model on
    
    Returns:
        model: Loaded model
    """
    # Import appropriate model
    if "UNet" in model_type and "Nested" not in model_type:
        from Code.model_lung_infection.InfNet_UNet_GroupNorm import UNet_GroupNorm
        model = UNet_GroupNorm(
            in_channels=3,
            out_channels=1,
            init_features=32,
        )
    elif "NestedUNet" in model_type:
        from Code.model_lung_infection.InfNet_NestedUNet_GroupNorm import (
            NestedUNet_GroupNorm,
        )
        model = NestedUNet_GroupNorm(
            input_channels=3,
            num_classes=1,
            deep_supervision=False,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Convert to GroupNorm (already done in training, but ensure compatibility)
    print("Converting BatchNorm to GroupNorm for compatibility")
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
        print("Model architecture converted to GroupNorm")
    else:
        print("Model already GroupNorm-compatible")
    
    # Move to device
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    
    # Clean state dict (remove THOP keys, wrapper prefixes)
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # Skip THOP keys
        if k.endswith(("total_ops", "total_params")):
            continue
        # Remove wrapper prefixes
        if k.startswith("module."):
            new_key = k[len("module.") :]
        elif k.startswith("_module."):
            new_key = k[len("_module.") :]
        else:
            new_key = k
        filtered_state_dict[new_key] = v
    
    # Load state dict
    try:
        model.load_state_dict(filtered_state_dict, strict=True)
        print("Checkpoint loaded successfully")
    except RuntimeError as e:
        print(f"Warning: Could not load with strict=True: {e}")
        print("Falling back to strict=False")
        model.load_state_dict(filtered_state_dict, strict=False)
    
    model.eval()
    return model


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testsize", type=int, default=352, help="testing size"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./Dataset/TestingSet/LungInfection-Test/",
        help="Path to test data",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[
            "UNet_GroupNorm",
            "UNet_Morph_GroupNorm",
            "NestedUNet_GroupNorm",
            "NestedUNet_Morph_GroupNorm",
        ],
        help="Model type (must match training configuration)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        required=True,
        help="Batch size used during training",
    )
    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="Run number used during training",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to test (default: latest available)",
    )
    parser.add_argument(
        "--morph_operation",
        type=str,
        default=None,
        choices=["open", "close", "dilation", "erosion", "both"],
        help="Morphological operation (required if model_type includes Morph)",
    )
    parser.add_argument(
        "--pth_path",
        type=str,
        default=None,
        help="Custom path to weights file (overrides model_type-based path)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Custom path to save results (overrides default structure)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )
    
    opt = parser.parse_args()
    
    # Validate morph_operation for Morph models
    if "Morph" in opt.model_type and opt.morph_operation is None:
        raise ValueError(
            f"morph_operation is required for model_type '{opt.model_type}'"
        )
    
    print(
        "#" * 50,
        "\nStart Testing\n"
        f"Model Type: {opt.model_type}\n"
        f"Batch Size: {opt.batchsize}\n"
        f"Run: {opt.run}\n"
        f"Epoch: {'Latest' if opt.epoch is None else opt.epoch}\n"
        f"{'Morph Operation: ' + opt.morph_operation if opt.morph_operation else ''}\n"
        f"{opt}\n",
        "#" * 50,
    )
    
    # Build model path
    try:
        model_path, epoch = build_model_path(opt)
        print(f"Using model: {model_path}")
        print(f"Epoch: {epoch}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load model
    device = opt.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    model = load_model(model_path, opt.model_type, device)
    
    # Build result path
    result_path = build_result_path(opt, epoch)
    print(f"Results will be saved to: {result_path}")
    
    # Load test data
    image_root = "{}/Imgs/".format(opt.data_path)
    test_loader = test_dataset(image_root, opt.testsize)
    print(f"Testing on {test_loader.size} images...")
    
    # Test loop
    with torch.no_grad():
        for i in range(test_loader.size):
            image, name = test_loader.load_data()
            image = image.to(device)
            
            # Forward pass
            pred = model(image)
            
            # Handle deep supervision output (list) for NestedUNet
            if isinstance(pred, list):
                pred = pred[-1]  # Use final output
            
            # Process prediction
            res = pred.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)
            
            # Save result
            save_path = os.path.join(result_path, name)
            imageio.imwrite(save_path, res)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{test_loader.size} images")
    
    print(f"\nTest Done! Results saved to: {result_path}")


if __name__ == "__main__":
    inference()

