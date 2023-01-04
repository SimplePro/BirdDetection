import os
import argparse

if __name__ == '__main__':

    root_dir = "/home/kdhsimplepro/kdhsimplepro/AI/BirdDetection/"

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="model_name")

    parser.add_argument("--model_size", default="m", type=str, help="model_size")

    parser.add_argument("--epochs", default=100, type=int, help="epochs")

    args = parser.parse_args()

    # train
    os.system(
        f"python3 {os.path.join(root_dir, 'yolov5', 'train.py')} --img 256 --batch 16 --epochs {args.epochs} \
        --data {os.path.join(root_dir, 'original_dataset', 'data.yaml')} --name {args.model_name} --cfg {os.path.join(root_dir, 'yolov5', 'models', f'yolov5{args.model_size}.yaml')}"
    )

    # valid
    os.system(
        f"python3 {os.path.join(root_dir, 'yolov5', 'val.py')} --img 256 --data {os.path.join(root_dir, 'original_dataset', 'data.yaml')} \
        --weights {os.path.join(root_dir, 'yolov5', 'runs', 'train', args.model_name, 'weights', 'best.pt')} --name val_{args.model_name}"
    )

    # test
    os.system(
        f"python3 {os.path.join(root_dir, 'yolov5', 'val.py')} --img 256 --data {os.path.join(root_dir, 'original_dataset', 'data.yaml')} \
        --weights {os.path.join(root_dir, 'yolov5', 'runs', 'train', args.model_name, 'weights', 'best.pt')} --name test_{args.model_name} --task test"
    )
    
    # detect
    os.system(
        f"python3 {os.path.join(root_dir, 'yolov5', 'detect.py')} --source test_images \
        --weights {os.path.join(root_dir, 'yolov5', 'runs', 'train', args.model_name, 'weights', 'best.pt')} --name test_images_{args.model_name}"
    )