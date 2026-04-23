# -*- coding: utf-8 -*-
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a MedSAM checkpoint to SAM checkpoint format for inference."
    )
    parser.add_argument(
        "--sam-ckpt-path",
        default="/data/sam_vit_b_01ec64.pth",
        help="Path to the base SAM checkpoint.",
    )
    parser.add_argument(
        "--medsam-ckpt-path",
        default="/data/mask-training-2025-05-31/MedSAM-ExASPIM-3brains-20250531-1654/medsam_model_best.pth",
        help="Path to the MedSAM training checkpoint.",
    )
    parser.add_argument(
        "--save-path",
        default="/results/medsam_best.pth",
        help="Path where the converted checkpoint should be saved.",
    )
    parser.add_argument(
        "--multi-gpu-ckpt",
        action="store_true",
        help="Set when the MedSAM checkpoint was trained with multi-GPU and uses module-prefixed keys.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sam_ckpt = torch.load(args.sam_ckpt_path)
    medsam_ckpt = torch.load(args.medsam_ckpt_path)
    sam_keys = sam_ckpt.keys()
    for key in sam_keys:
        if not args.multi_gpu_ckpt:
            sam_ckpt[key] = medsam_ckpt["model"][key]
        else:
            sam_ckpt[key] = medsam_ckpt["model"]["module." + key]

    torch.save(sam_ckpt, args.save_path)


if __name__ == "__main__":
    main()
