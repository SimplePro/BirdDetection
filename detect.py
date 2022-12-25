import os
from argparse import ArgumentParser


if __name__ == '__main__':

    root_dir = "/home/kdhsimplepro/kdhsimplepro/AI/BirdDetection/"

    parser = ArgumentParser()

    parser.add_argument("--model_name", default="model", type=str, help="model name")

    parser.add_argument("--source", required=True, type=str, help="test images source")

    parser.add_argument("--name", required=True, type=str, help="folder name")

    parser.add_argument("--conf", default=0.5, type=float, help="confidence score")

    args = parser.parse_args()

    os.system(
        f'python3 {os.path.join(root_dir, "yolov5", "detect.py")} --source {args.source} --weights {os.path.join(root_dir, "yolov5", "runs", "train", args.model_name, "weights", "best.pt")} --name {args.name} --conf {args.conf}'
    )