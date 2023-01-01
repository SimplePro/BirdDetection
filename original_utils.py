import torch
import torch.nn.functional as F
from torchvision import transforms

import albumentations as A
from random import uniform, randint, choices, shuffle

import numpy as np

from tqdm import tqdm

from PIL import Image
from PIL import ImageDraw

import os

from glob import glob

import pickle


def draw_bbox(img, bboxes):
    image = transforms.ToPILImage()(img)
    _, H, W = img.shape

    draw = ImageDraw.Draw(image)
    
    for x, y, w, h in bboxes:
        xmin = (x - (w/2)) * W
        ymin = (y - (h/2)) * H
        xmax = (x + (w/2)) * W
        ymax = (y + (h/2)) * H

        draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=3)
    
    return image



def yolo2coco(bbox, img_size=256):
        x, y, w, h = bbox

        return [
            int(min(1, max(0, (x - w/2))) * img_size),
            int(min(1, max(0, (y - h/2))) * img_size),
            int(min(1, max(0, (x + w/2))) * img_size),
            int(min(1, max(0, (y + h/2))) * img_size),
        ]

    
def coco2yolo(bbox, img_size=256):
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin, xmax, ymax = xmin / img_size, ymin / img_size, xmax / img_size, ymax / img_size

    return [
        (xmin+xmax)/2, (ymin + ymax)/2, (xmax-xmin), (ymax-ymin)
    ]


class BackgroundCompositor:

    def __init__(
        self,
        bird_images,
        mask_images,
        bboxes_list,
        background_images,
        iof_threshold=0.4, # intersection over front_area
        iob_threshold=0.4, # intersection over back_area
        min_size_of_bird=8,
        max_size_of_bird=256,
        scale_factor_statistics=[(0.25, 0.1), (0.6, 0.2), (1, 0.3)],
        scale_factor_weights=[0.2, 0.3, 0.5]
    ):
    
        self.img_H, self.img_W = 256, 256

        self.bird_images = bird_images
        self.mask_images = mask_images
        self.bboxes_list = bboxes_list
        self.background_images = background_images

        self.iof_threshold = iof_threshold
        self.iob_threshold = iob_threshold

        self.min_size_of_bird = min_size_of_bird
        self.max_size_of_bird = max_size_of_bird
        self.scale_factor_statistics = scale_factor_statistics
        self.scale_factor_weights = scale_factor_weights

        self.croped_bird_images = []
        self.birds_p = []
        self.croped_mask_images = []
        self.croped_bird_class_list = []

        self.crop_bird_and_mask_images()

        self.transformation = A.Compose([
                A.VerticalFlip(p=0.2),
                A.HorizontalFlip(p=0.5),
                A.AdvancedBlur(p=0.3)
            ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    
    def crop_bird_and_mask_images(
        self
    ):
        count_list = [0] * 5
        
        for bird_img, mask_img, bboxes in zip(self.bird_images, self.mask_images, self.bboxes_list):
            
            for c, x, y, w, h in bboxes:
                xmin, ymin, xmax, ymax = (x - w/2), (y - h/2), (x + w/2), (y + h/2)
                xmin, ymin, xmax, ymax = int(xmin * self.img_W), int(ymin * self.img_H), int(xmax * self.img_W), int(ymax * self.img_H)

                self.croped_bird_images.append(
                    bird_img[:, ymin:ymax, xmin:xmax]
                )

                self.croped_mask_images.append(
                    mask_img[:, ymin:ymax, xmin:xmax]
                )

                self.croped_bird_class_list.append(c)

                count_list[int(c)] += 1
        
        class_p = [0.2 / count_list[i] for i in range(5)]

        for i in range(len(self.croped_bird_images)):
            c = self.croped_bird_class_list[i]
            self.birds_p.append(class_p[c])


    def do_transform_img(
        self,
        bird_img,
        mask_img,
        bboxes, # [[class, x, y, w, h] ... ]
    ):

        is_bird = bird_img != None
        is_mask = mask_img != None
        is_bbox = bboxes != None

        mask_img = torch.zeros((1, bird_img.size(1), bird_img.size(2))) if not is_mask else mask_img
        bboxes = [[0, 0.5, 0.5, 0.5, 0.5]] if not is_bbox else bboxes

        transform_result = self.transformation(
            image=bird_img.permute(1, 2, 0).numpy(),
            mask=mask_img.permute(1, 2, 0).numpy(),
            bboxes=[bbox[1:] for bbox in bboxes],
            class_labels=[bbox[0] for bbox in bboxes]
        )

        result = []

        if is_bird:
            result.append(torch.from_numpy(transform_result["image"]).permute(2, 0, 1))
        
        if is_mask:
            result.append(torch.from_numpy(transform_result["mask"]).permute(2, 0, 1))

        if is_bbox:
            result.append([[bboxes[i][0]] + list(bbox) for i, bbox in enumerate(transform_result["bboxes"])])

        return result
        

    def random_resize(
        self,
        bird_img,
        mask_img,
    ):
        bird_H, bird_W = bird_img.size(1), bird_img.size(2)

        [(mean, std)] = choices(self.scale_factor_statistics, weights=self.scale_factor_weights, k=1)

        random_scale_factor = np.random.normal(size=1).item() * std + mean

        min_scale_factor = self.min_size_of_bird / min(bird_H, bird_W)
        max_scale_factor = self.max_size_of_bird / max(bird_H, bird_W)

        random_scale_factor = min(max_scale_factor, max(min_scale_factor, random_scale_factor))
        
        resized_bird_img = F.interpolate(bird_img.unsqueeze(0), scale_factor=random_scale_factor, mode="bilinear").squeeze(0)
        resized_mask_img = F.interpolate(mask_img.unsqueeze(0), scale_factor=random_scale_factor, mode="bilinear").squeeze(0)

        return resized_bird_img, resized_mask_img


    def get_random_position(
        self,
        bird_img,
        background_img
    ):
        bird_H, bird_W = bird_img.size(1), bird_img.size(2)
        background_H, background_W = background_img.size(1), background_img.size(2)

        xmin = randint(0, background_W - bird_W)
        ymin = randint(0, background_H - bird_H)
        
        x = (xmin + bird_W/2) / background_W
        y = (ymin + bird_H/2) / background_H
        w = bird_W / background_W
        h = bird_H / background_H

        return (x, y, w, h)


    def do_augmentation(
        self,
        birds_n=[0, 1, 2, 3],
        birds_n_p=[0.1, 0.4, 0.3, 0.2],
        dataset_n=40000
    ):

        augmentation_bird_images = []
        augmentation_bboxes_list = []

        print("background composition")
        for _ in tqdm(range(dataset_n)):
            random_background = self.background_images[randint(0, len(self.background_images))-1]
            [number_of_birds] = choices(birds_n, weights=birds_n_p, k=1)

            bboxes = []

            pasted_img = random_background.clone()
            id_img = torch.zeros((1, pasted_img.size(1), pasted_img.size(2))).fill_(-1) # iof, iob 를 계산하기 위한 변수.

            for new_imgid in range(number_of_birds):
                [random_idx] = choices(list(range(len(self.croped_bird_images))), k=1, weights=self.birds_p)

                bird_image = self.croped_bird_images[random_idx]
                mask_image = self.croped_mask_images[random_idx]
                class_ = self.croped_bird_class_list[random_idx]
                
                bird_image, mask_image = self.do_transform_img(
                    bird_image, mask_image, bboxes=None,
                )

                bird_image, mask_image = self.random_resize(bird_image, mask_image)

                random_position = self.get_random_position(bird_image, pasted_img)

                its_ok = True

                new_xmin, new_ymin, new_xmax, new_ymax = yolo2coco(random_position)
                _, _, new_w, new_h = random_position
                new_w, new_h = new_w * 256, new_h * 256

                iof = (torch.sum(id_img[:, new_ymin:new_ymax, new_xmin:new_xmax] != -1) / (new_w * new_h)).item()

                if iof > self.iof_threshold:
                    its_ok = False

                for imgid, bbox in enumerate(bboxes):
                    xmin, ymin, xmax, ymax = yolo2coco(bbox[1:])

                    intersection_area = (min(xmax, new_xmax) - max(xmin, new_xmin)) * (min(ymax, new_ymax) - max(ymin, new_ymin))
                    if intersection_area < 0: intersection_area = 0

                    back_area = new_w * new_h
                    intersection_area = torch.sum(id_img[:, ymin:ymax, xmin:xmax] != imgid) + intersection_area
                    iob = (intersection_area / back_area).item()

                    if iob > self.iob_threshold:
                        its_ok = False
                        break

                if its_ok:
                    xmin, ymin, xmax, ymax = yolo2coco(random_position)

                    id_img[:, ymin:ymax, xmin:xmax] = new_imgid
                    pasted_img[:, ymin:ymax, xmin:xmax] = (bird_image * torch.sqrt(mask_image)) + (pasted_img[:, ymin:ymax, xmin:xmax] * (1 - torch.sqrt(mask_image)))

                    bboxes.append([class_, *random_position])

            augmentation_bird_images.append(pasted_img.type(torch.float16))
            augmentation_bboxes_list.append(bboxes)

        return augmentation_bird_images, augmentation_bboxes_list



class Transforms:

    def __init__(
        self,
        bird_images,
        bboxes_list,
    ):


        self.albumentation_transform = A.Compose([
                A.CropAndPad(percent=(-0.2,0.4), p=1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.AdvancedBlur(p=0.3),
                A.RandomFog(p=0.3)
            ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

        self.bird_images = bird_images
        self.bboxes_list = bboxes_list

    
    def transform_function(
        self,
        img,
        bboxes # yolo
    ):

        transform_result = self.albumentation_transform(
            image=img.permute(1, 2, 0).numpy(),
            bboxes=[coco2yolo(yolo2coco(bbox[1:])) for bbox in bboxes],
            class_labels=[bbox[0] for bbox in bboxes]
        )

        return (
            torch.from_numpy(transform_result["image"]).permute(2, 0, 1),
            [
                [bboxes[i][0]] + list(bbox) # yolo
                for i, bbox in enumerate(transform_result["bboxes"])
            ]
        )


    def do_augmentation(
        self,
        augmentation_n
    ):

        augmentation_bird_images = []
        augmentation_bboxes_list = []

        print("simple data sugmentation")
        if augmentation_n > 0:
            for bird_img, bboxes in tqdm(zip(self.bird_images, self.bboxes_list)):
                for _ in range(augmentation_n-1):
                    augmentation_bird_img, augmentation_bboxes = self.transform_function(img=bird_img, bboxes=bboxes)

                    augmentation_bird_images.append(augmentation_bird_img.type(torch.float16))
                    augmentation_bboxes_list.append(augmentation_bboxes)

                augmentation_bird_images.append(bird_img.type(torch.float16))
                augmentation_bboxes_list.append(bboxes)

        return augmentation_bird_images, augmentation_bboxes_list


class AugmentationClass:

    def __init__(
        self,
        bird_images,
        mask_images,
        bboxes_list,
        background_images
    ):
        self.bird_images = bird_images
        self.mask_images = mask_images
        self.bboxes_list = bboxes_list
        self.background_compositor = BackgroundCompositor(
            bird_images=bird_images,
            mask_images=mask_images,
            bboxes_list=bboxes_list,
            background_images=background_images
        )
        self.custom_transform = Transforms(bird_images=bird_images, bboxes_list=bboxes_list)

    
    def do_augmentation(
        self,
        images_dir,
        labels_dir,
        background_compositor_data_n=40000,
        augmentation_n=10
    ):

        background_composed_bird_images, background_composed_bboxes_list = self.background_compositor.do_augmentation(
            dataset_n=background_compositor_data_n,
            birds_n=[0, 1, 2, 3], birds_n_p=[0.1, 0.4, 0.3, 0.2]
        )
        augmentation_bird_images, augmentation_bboxes_list = self.custom_transform.do_augmentation(augmentation_n=augmentation_n)

        for i, (bird_image, bboxes) in enumerate(
            zip(background_composed_bird_images + augmentation_bird_images, background_composed_bboxes_list + augmentation_bboxes_list)
        ):
            img = transforms.ToPILImage()(bird_image)
            img.save(os.path.join(images_dir, f"{i}.jpg"))

            with open(os.path.join(labels_dir, f"{i}.txt"), "w") as f:
                f.write("\n".join([f"{int(c)} {x} {y} {w} {h}" for (c, x, y, w, h) in bboxes]) + "\n")



if __name__ == '__main__':

    is_train_valid_test = 0 # 0: train, 1: valid, 2: test

    dataset_mode = {
        0: "train",
        1: "valid",
        2: "test"
    }[is_train_valid_test]

    background_compositor_data_n = {
        0: 50000,
        1: 3000,
        2: 0
    }[is_train_valid_test]

    augmentation_n = {
        0: 20,
        1: 10,
        2: 1
    }[is_train_valid_test]

    root_dir = "/home/kdhsimplepro/kdhsimplepro/AI/BirdDetection/"

    images_dir = os.path.join(root_dir, "original_dataset", dataset_mode, "images")
    labels_dir = os.path.join(root_dir, "original_dataset", dataset_mode, "labels")

    with open(os.path.join(root_dir, "datasamples", "split_indexes.pickle"), "rb") as f:
        data_split_indexes = pickle.load(f)

    bird_images = []
    mask_images = []
    bboxes_list = []

    print("add dataset in list variable")
    for i in tqdm(range(len(data_split_indexes))):
        if data_split_indexes[i] == is_train_valid_test:

            try:
                bird_path = os.path.join(root_dir, "datasamples", "images", f"{i}.jpg")
                mask_path = os.path.join(root_dir, "datasamples", "mask", f"{i}.jpg")
                bboxes_path = os.path.join(root_dir, "datasamples", "bboxes", f"{i}.txt")

                bird_image = transforms.ToTensor()(Image.open(bird_path).convert("RGB"))
                mask_image = transforms.ToTensor()(Image.open(mask_path).convert("L"))

                bboxes = []
                
                with open(bboxes_path, "r") as f:
                    lines = f.readlines()

                    for line in lines:
                        c, x, y, w, h = map(float, line.split())
                        c = int(c)

                        bboxes.append([c, x, y, w, h])
                
                bird_images.append(bird_image)
                mask_images.append(mask_image)
                bboxes_list.append(bboxes)

            except:
                pass

    if is_train_valid_test in [0, 1]:
        background_images = []

        print("background images")
        for path in tqdm(glob(os.path.join(root_dir, "Landscape", "*.jpg"))):
            background_images.append(
                (transforms.ToTensor()(Image.open(path).convert("RGB").resize((256, 256)))).type(torch.float16)
            )

        augmentation_class = AugmentationClass(bird_images, mask_images, bboxes_list, background_images)

        augmentation_class.do_augmentation(
            images_dir=images_dir,
            labels_dir=labels_dir,
            background_compositor_data_n=background_compositor_data_n,
            augmentation_n=augmentation_n
        )

    elif is_train_valid_test in [2]:
        for i, (bird_image, bboxes) in enumerate(zip(bird_images, bboxes_list)):
            img = transforms.ToPILImage()(bird_image)
            img.save(os.path.join(images_dir, f"{i}.jpg"))

            with open(os.path.join(labels_dir, f"{i}.txt"), "w") as f:
                f.write("\n".join([f"{int(c)} {x} {y} {w} {h}" for (c, x, y, w, h) in bboxes]) + "\n")


    image_paths = glob(os.path.join(root_dir, "original_dataset", dataset_mode, "images", "*.jpg"))

    shuffle(image_paths)

    with open(os.path.join(root_dir, "original_dataset", dataset_mode, f"{dataset_mode}.txt"), "w") as f:
        f.write("\n".join(image_paths) + "\n")