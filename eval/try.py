import os
import argparse
from evaluate_iou_loc import main as eval_main

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ramen_try using ramen as GT"
    )
    parser.add_argument(
        "--feat_dir",
        type=str,
        default="../output",
        help="Directory of rendered features (for ramen_try)",
    )
    parser.add_argument(
        "--ae_ckpt_dir",
        type=str,
        default="../autoencoder/ckpt",
        help="Directory of autoencoder checkpoints",
    )
    parser.add_argument(
        "--json_folder",
        type=str,
        required=True,
        help="Path to lerf_ovs/label (GT annotations of ramen)",
    )
    parser.add_argument(
        "--encoder_dims",
        nargs="+",
        type=int,
        default=[256, 128, 64, 32, 3],
    )
    parser.add_argument(
        "--decoder_dims",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 256, 512],
    )
    # 允许指定 ramen_try 的输出子目录名，默认 ramen_try_3
    parser.add_argument(
        "--pred_scene",
        type=str,
        default="ramen_try_3",
        help="Predicted scene name (rendered features dir under feat_dir)",
    )
    # 允许指定 ramen 的 AE 名字，默认 ramen
    parser.add_argument(
        "--gt_scene",
        type=str,
        default="ramen",
        help="GT scene name (for AE ckpt and labels)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 构造传给原 evaluate_iou_loc.main 的参数对象
    class EvalArgs:
        pass

    eval_args = EvalArgs()
    # 预测结果来自 ramen_try
    eval_args.dataset_name = args.pred_scene
    eval_args.feat_dir = args.feat_dir

    # AE ckpt 和 label 都用 ramen
    eval_args.ae_ckpt_dir = args.ae_ckpt_dir
    eval_args.encoder_dims = args.encoder_dims
    eval_args.decoder_dims = args.decoder_dims
    eval_args.json_folder = args.json_folder

    # 在原脚本中，通常 dataset_name 也用于选择 label json，
    # 这里需要显式告诉内部：label 来自 ramen
    # 如果 evaluate_iou_loc.py 里支持单独的 GT 名字，可以传；否则可以通过 json_folder 区分
    # 假设它只看 json_folder，就不再额外区分 GT 名

    # 调用原评测入口
    eval_main(eval_args)

if __name__ == "__main__":
    main()