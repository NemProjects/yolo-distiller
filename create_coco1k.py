#!/usr/bin/env python3
"""
COCO-1K Dataset Creator
Creates a balanced subset of COCO dataset with ~1500 images for fast KD validation
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm

class COCO1KCreator:
    def __init__(self, coco_root="/datasets/coco", target_size=1500, output_dir="coco1k"):
        self.coco_root = Path(coco_root)
        self.target_size = target_size
        self.output_dir = Path(output_dir)
        self.images_per_class = target_size // 80  # ~19 images per class

        # COCO paths
        self.train_ann_file = self.coco_root / "annotations/instances_train2017.json"
        self.train_img_dir = self.coco_root / "train2017"
        self.val_ann_file = self.coco_root / "annotations/instances_val2017.json"
        self.val_img_dir = self.coco_root / "val2017"

        # Output paths
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_train_dir = self.output_dir / "train2017"
        self.output_val_dir = self.output_dir / "val2017"
        self.output_ann_dir = self.output_dir / "annotations"

        for dir_path in [self.output_train_dir, self.output_val_dir, self.output_ann_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

    def load_coco_data(self):
        """Load COCO annotations"""
        print("Loading COCO annotations...")
        with open(self.train_ann_file) as f:
            self.train_data = json.load(f)
        with open(self.val_ann_file) as f:
            self.val_data = json.load(f)
        print(f"Loaded {len(self.train_data['images'])} train images, {len(self.val_data['images'])} val images")

    def get_balanced_sample(self):
        """Get balanced sample of images across all classes"""
        print("Creating balanced sample...")

        # Group images by categories for train set
        train_cat_to_imgs = defaultdict(set)
        train_img_id_to_img = {img['id']: img for img in self.train_data['images']}

        for ann in self.train_data['annotations']:
            train_cat_to_imgs[ann['category_id']].add(ann['image_id'])

        # Sample images ensuring each category is represented
        selected_train_imgs = set()
        for cat_id, img_ids in train_cat_to_imgs.items():
            img_list = list(img_ids)
            sample_size = min(self.images_per_class, len(img_list))
            selected = random.sample(img_list, sample_size)
            selected_train_imgs.update(selected)

        # If we need more images, add randomly
        all_train_imgs = set(img['id'] for img in self.train_data['images'])
        remaining_imgs = all_train_imgs - selected_train_imgs
        if len(selected_train_imgs) < self.target_size and remaining_imgs:
            additional_needed = self.target_size - len(selected_train_imgs)
            additional = random.sample(list(remaining_imgs),
                                     min(additional_needed, len(remaining_imgs)))
            selected_train_imgs.update(additional)

        # For validation, take a smaller balanced sample
        val_cat_to_imgs = defaultdict(set)
        val_img_id_to_img = {img['id']: img for img in self.val_data['images']}

        for ann in self.val_data['annotations']:
            val_cat_to_imgs[ann['category_id']].add(ann['image_id'])

        selected_val_imgs = set()
        val_per_class = max(2, self.images_per_class // 4)  # ~5 images per class for val
        for cat_id, img_ids in val_cat_to_imgs.items():
            img_list = list(img_ids)
            sample_size = min(val_per_class, len(img_list))
            selected = random.sample(img_list, sample_size)
            selected_val_imgs.update(selected)

        print(f"Selected {len(selected_train_imgs)} train images, {len(selected_val_imgs)} val images")
        return selected_train_imgs, selected_val_imgs

    def copy_images(self, selected_train_imgs, selected_val_imgs):
        """Copy selected images to output directory"""
        print("Copying images...")

        # Copy train images
        train_img_id_to_img = {img['id']: img for img in self.train_data['images']}
        for img_id in tqdm(selected_train_imgs, desc="Copying train images"):
            img_info = train_img_id_to_img[img_id]
            src = self.train_img_dir / img_info['file_name']
            dst = self.output_train_dir / img_info['file_name']
            if src.exists():
                shutil.copy2(src, dst)

        # Copy val images
        val_img_id_to_img = {img['id']: img for img in self.val_data['images']}
        for img_id in tqdm(selected_val_imgs, desc="Copying val images"):
            img_info = val_img_id_to_img[img_id]
            src = self.val_img_dir / img_info['file_name']
            dst = self.output_val_dir / img_info['file_name']
            if src.exists():
                shutil.copy2(src, dst)

    def create_annotations(self, selected_train_imgs, selected_val_imgs):
        """Create new annotation files for the subset"""
        print("Creating annotation files...")

        # Create train annotations
        new_train_data = {
            'info': self.train_data['info'],
            'licenses': self.train_data['licenses'],
            'categories': self.train_data['categories'],
            'images': [img for img in self.train_data['images'] if img['id'] in selected_train_imgs],
            'annotations': [ann for ann in self.train_data['annotations'] if ann['image_id'] in selected_train_imgs]
        }

        with open(self.output_ann_dir / "instances_train2017.json", 'w') as f:
            json.dump(new_train_data, f)

        # Create val annotations
        new_val_data = {
            'info': self.val_data['info'],
            'licenses': self.val_data['licenses'],
            'categories': self.val_data['categories'],
            'images': [img for img in self.val_data['images'] if img['id'] in selected_val_imgs],
            'annotations': [ann for ann in self.val_data['annotations'] if ann['image_id'] in selected_val_imgs]
        }

        with open(self.output_ann_dir / "instances_val2017.json", 'w') as f:
            json.dump(new_val_data, f)

        print(f"Created annotations: {len(new_train_data['images'])} train, {len(new_val_data['images'])} val")
        print(f"Train annotations: {len(new_train_data['annotations'])}")
        print(f"Val annotations: {len(new_val_data['annotations'])}")

    def create_yaml_config(self):
        """Create YAML config file for Ultralytics"""
        yaml_content = f"""# COCO-1K dataset configuration for Knowledge Distillation validation
# Created from COCO 2017 with ~{self.target_size} balanced training images

path: {self.output_dir.absolute()}  # dataset root dir
train: train2017  # train images (relative to 'path')
val: val2017  # val images (relative to 'path')

# Classes (same as COCO)
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
"""

        with open(self.output_dir / "coco1k.yaml", 'w') as f:
            f.write(yaml_content)

        print(f"Created YAML config: {self.output_dir}/coco1k.yaml")

    def create_dataset(self):
        """Main method to create the COCO-1K dataset"""
        print(f"Creating COCO-1K dataset with ~{self.target_size} images...")
        print(f"Output directory: {self.output_dir}")

        # Check if COCO dataset exists
        if not self.train_ann_file.exists():
            print(f"COCO dataset not found at {self.coco_root}")
            print("Please download COCO dataset first:")
            print("  wget http://images.cocodataset.org/zips/train2017.zip")
            print("  wget http://images.cocodataset.org/zips/val2017.zip")
            print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
            return False

        random.seed(42)  # For reproducibility

        self.load_coco_data()
        selected_train_imgs, selected_val_imgs = self.get_balanced_sample()
        self.copy_images(selected_train_imgs, selected_val_imgs)
        self.create_annotations(selected_train_imgs, selected_val_imgs)
        self.create_yaml_config()

        print("\n✅ COCO-1K dataset created successfully!")
        print(f"📁 Dataset location: {self.output_dir}")
        print(f"📄 YAML config: {self.output_dir}/coco1k.yaml")
        print(f"🖼️  Train images: {len(selected_train_imgs)}")
        print(f"🔍 Val images: {len(selected_val_imgs)}")

        return True

def main():
    parser = argparse.ArgumentParser(description='Create COCO-1K dataset for KD validation')
    parser.add_argument('--coco-root', default='/datasets/coco',
                       help='Path to COCO dataset root')
    parser.add_argument('--target-size', type=int, default=1500,
                       help='Target number of training images')
    parser.add_argument('--output-dir', default='coco1k',
                       help='Output directory for COCO-1K dataset')

    args = parser.parse_args()

    creator = COCO1KCreator(
        coco_root=args.coco_root,
        target_size=args.target_size,
        output_dir=args.output_dir
    )

    creator.create_dataset()

if __name__ == "__main__":
    main()