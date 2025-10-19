# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import imageio
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import test_dataset


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsize", type=int, default=352, help="testing size")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./Dataset/TestingSet/LungInfection-Test/",
        help="Path to test data",
    )
    parser.add_argument(
        "--pth_path",
        type=str,
        default="./Snapshots/save_weights/Inf-Net/DP/2/Inf-Net-2.pth",
        help="Path to weights file. If `semi-sup`, edit it to `Semi-Inf-Net/Semi-Inf-Net-100.pth`",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./Results/Lung_infection_segmentation/Inf-Net/DP/2",
        help="Path to save the predictions. if `semi-sup`, edit it to `Semi-Inf-Net`",
    )
    parser.add_argument("--run", type=int, help="the raining iteartion number")
    parser.add_argument("--use_cpu", action="store_true", help="use CPU instead of GPU")
    opt = parser.parse_args()

    print(
        "#" * 20,
        "\nStart Testing (Inf-Net)\n{}\nThis code is written for 'Inf-Net: Automatic COVID-19 Lung "
        "Infection Segmentation from CT Scans', 2020, TMI.\n"
        "----\nPlease cite the paper if you use this code and dataset. "
        "And any questions feel free to contact me "
        "via E-mail (gepengai.ji@gamil.com)\n----\n".format(opt),
        "#" * 20,
    )

    # Determine device
    device = torch.device(
        "cpu" if opt.use_cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Using device: {device}")

    model = Network()

    # Load the state dict
    state_dict = torch.load(opt.pth_path, map_location=device)

    # Remove DataParallel wrapper prefix and THOP keys
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # Skip THOP keys
        if k.endswith(("total_ops", "total_params")):
            continue
        # Remove '_module.' prefix if present (from DataParallel)
        new_key = k.replace("_module.", "") if k.startswith("_module.") else k
        filtered_state_dict[new_key] = v

    # Use strict=False to allow missing running_mean/running_var buffers
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()

    image_root = "{}/Imgs/".format(opt.data_path)
    # gt_root = '{}/GT/'.format(opt.data_path)
    test_loader = test_dataset(image_root, opt.testsize)
    if opt.run:
        test_saving_path = os.path.join(opt.save_path, str(opt.run))
    else:
        test_saving_path = opt.save_path
    os.makedirs(test_saving_path, exist_ok=True)

    for i in range(test_loader.size):
        image, name = test_loader.load_data()

        image = image.to(device)

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = (
            model(image)
        )

        res = lateral_map_2
        # res = F.upsample(res, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)  # Convert to uint8 for image saving
        save_path = os.path.join(test_saving_path, name)
        imageio.imwrite(save_path, res)

    print("Test Done!")


if __name__ == "__main__":
    inference()
