import os
from argparse import ArgumentParser


if __name__ == '__main__':

    root_dir = "/home/kdhsimplepro/kdhsimplepro/AI/BirdDetection/"

    parser = ArgumentParser()

    parser.add_argument("--model_name", default="model", type=str, help="model name")

    parser.add_argument("--data", required=True, type=str, help="path of data.yaml")

    parser.add_argument("--name", required=True, type=str, help="folder name")

    parser.add_argument("--iou_threshold", default=0.5, type=float, help="iou threshold")

    args = parser.parse_args()

    os.system(
        f'python3 {os.path.join(root_dir, "yolov5", "val.py")} --data {args.data} --weights {os.path.join(root_dir, "yolov5", "runs", "train", args.model_name, "weights", "best.pt")} --task test --name {args.name} --iou-thres {args.iou_threshold}'
    )