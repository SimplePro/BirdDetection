import os
from argparse import ArgumentParser


if __name__ == '__main__':

    root_dir = "/home/kdhsimplepro/kdhsimplepro/AI/BirdDetection/"

    parser = ArgumentParser()

    parser.add_argument("--batch", default=16, type=int, help="batch size")

    parser.add_argument("--epochs", default=50, type=int, help="epochs")

    parser.add_argument("--model_size", default="m", type=str, help="model size (x, s, m, l)")

    parser.add_argument("--model_name", default="model", type=str, help="model name")

    args = parser.parse_args()

    os.system(
        f'python3 {os.path.join(root_dir, "yolov5", "train.py")} --img 256 --batch {args.batch} --epochs {args.epochs} --data {os.path.join(root_dir, "dataset", "data.yaml")} --cfg {os.path.join(root_dir, "yolov5", "models", f"yolov5{args.model_size}.yaml")} --weights {f"yolov5{args.model_size}.pt"} --name {args.model_name}'
    )