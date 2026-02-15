"""
Generate annotation files for VOC-FOG training.
This script creates train.txt and val.txt files with paths to foggy images
and their corresponding bounding box annotations.
"""

import os
import random
import xml.etree.ElementTree as ET
import numpy as np

# Configuration
annotation_mode = 0  # 0 = generate both ImageSets txt and annotation txt files
trainval_percent = 0.9  # 90% for trainval, 10% for test
train_percent = 0.9  # 90% of trainval for train, 10% for val

# Path to VOC2007 dataset (relative to RDFNet folder)
VOC_PATH = "../VOC2007"

# VOC 20 classes
classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

nums = np.zeros(len(classes))

def convert_annotation(image_id, list_file, use_fog=True):
    """Parse XML annotation and write bounding boxes to list_file."""
    xml_path = os.path.join(VOC_PATH, 'Annotations', f'{image_id}.xml')
    
    if not os.path.exists(xml_path):
        print(f"Warning: {xml_path} not found")
        return False
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    has_object = False
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = int(obj.find('difficult').text)
        
        cls = obj.find('name').text
        if cls not in classes or difficult == 1:
            continue
        
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (
            int(float(xmlbox.find('xmin').text)),
            int(float(xmlbox.find('ymin').text)),
            int(float(xmlbox.find('xmax').text)),
            int(float(xmlbox.find('ymax').text))
        )
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        nums[classes.index(cls)] += 1
        has_object = True
    
    return has_object


def generate_annotations(voc_path, output_dir='./', trainval_percent=0.9, train_percent=0.9):
    """
    Generate annotation files for VOC-FOG training.
    
    Args:
        voc_path: Path to VOC dataset (e.g., '/kaggle/input/voc2007-fog/VOCdevkit/VOC2007')
        output_dir: Directory to save annotation files (default: current directory)
        trainval_percent: Ratio for trainval split (default: 0.9)
        train_percent: Ratio for train/val split within trainval (default: 0.9)
    """
    random.seed(0)
    
    # Update global VOC_PATH for convert_annotation function
    global VOC_PATH
    VOC_PATH = voc_path
    
    # Verify paths exist
    annotations_path = os.path.join(voc_path, 'Annotations')
    # Try FOG folder first, fallback to JPEGImages
    fog_path = os.path.join(voc_path, 'FOG')
    if not os.path.exists(fog_path):
        fog_path = os.path.join(voc_path, 'JPEGImages')
        print(f"FOG folder not found, using JPEGImages instead")
    
    imagesets_path = os.path.join(voc_path, 'ImageSets', 'Main')
    
    print(f"VOC Path: {os.path.abspath(voc_path)}")
    print(f"Annotations: {os.path.abspath(annotations_path)}")
    print(f"Images: {os.path.abspath(fog_path)}")
    
    if not os.path.exists(annotations_path):
        raise ValueError(f"Annotations folder not found: {annotations_path}")
    if not os.path.exists(fog_path):
        raise ValueError(f"Images folder not found: {fog_path}")
    
    os.makedirs(imagesets_path, exist_ok=True)
    
    # Get all XML files
    xml_files = [f[:-4] for f in os.listdir(annotations_path) if f.endswith('.xml')]
    total = len(xml_files)
    print(f"Found {total} annotation files")
    
    if total == 0:
        raise ValueError(f"No XML annotation files found in {annotations_path}")
    
    # Split dataset
    random.shuffle(xml_files)
    tv_count = int(total * trainval_percent)
    tr_count = int(tv_count * train_percent)
    
    trainval_ids = xml_files[:tv_count]
    test_ids = xml_files[tv_count:]
    train_ids = trainval_ids[:tr_count]
    val_ids = trainval_ids[tr_count:]
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Write ImageSets files
    with open(os.path.join(imagesets_path, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_ids))
    with open(os.path.join(imagesets_path, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_ids))
    with open(os.path.join(imagesets_path, 'trainval.txt'), 'w') as f:
        f.write('\n'.join(trainval_ids))
    with open(os.path.join(imagesets_path, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_ids))
    
    print("Generated ImageSets txt files")
    
    # Generate annotation files for training
    fog_abs_path = os.path.abspath(fog_path)
    
    # Output paths
    train_file = os.path.join(output_dir, '2007_train.txt')
    val_file = os.path.join(output_dir, '2007_val.txt')
    
    # Train annotations
    train_count = 0
    with open(train_file, 'w', encoding='utf-8') as f:
        for image_id in train_ids:
            img_path = os.path.join(fog_abs_path, f'{image_id}.jpg')
            if os.path.exists(img_path):
                f.write(img_path)
                if convert_annotation(image_id, f):
                    train_count += 1
                f.write('\n')
    
    # Val annotations
    val_count = 0
    with open(val_file, 'w', encoding='utf-8') as f:
        for image_id in val_ids:
            img_path = os.path.join(fog_abs_path, f'{image_id}.jpg')
            if os.path.exists(img_path):
                f.write(img_path)
                if convert_annotation(image_id, f):
                    val_count += 1
                f.write('\n')
    
    print(f"\n✅ Generated annotation files:")
    print(f"  {train_file}: {train_count} images")
    print(f"  {val_file}: {val_count} images")
    
    # Print class distribution
    print("\nClass distribution:")
    for i, cls in enumerate(classes):
        if nums[i] > 0:
            print(f"  {cls}: {int(nums[i])}")
    
    print("\n✅ Annotation generation complete!")
    return train_file, val_file


if __name__ == "__main__":
    random.seed(0)
    
    # Verify paths exist
    annotations_path = os.path.join(VOC_PATH, 'Annotations')
    fog_path = os.path.join(VOC_PATH, 'FOG')
    imagesets_path = os.path.join(VOC_PATH, 'ImageSets', 'Main')
    
    print(f"VOC Path: {os.path.abspath(VOC_PATH)}")
    print(f"Annotations: {os.path.abspath(annotations_path)}")
    print(f"FOG images: {os.path.abspath(fog_path)}")
    
    if not os.path.exists(annotations_path):
        raise ValueError(f"Annotations folder not found: {annotations_path}")
    if not os.path.exists(fog_path):
        raise ValueError(f"FOG folder not found: {fog_path}. Run run_fog.py first!")
    
    os.makedirs(imagesets_path, exist_ok=True)
    
    # Get all XML files
    xml_files = [f[:-4] for f in os.listdir(annotations_path) if f.endswith('.xml')]
    total = len(xml_files)
    print(f"Found {total} annotation files")
    
    # Split dataset
    random.shuffle(xml_files)
    tv_count = int(total * trainval_percent)
    tr_count = int(tv_count * train_percent)
    
    trainval_ids = xml_files[:tv_count]
    test_ids = xml_files[tv_count:]
    train_ids = trainval_ids[:tr_count]
    val_ids = trainval_ids[tr_count:]
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Write ImageSets files
    with open(os.path.join(imagesets_path, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_ids))
    with open(os.path.join(imagesets_path, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_ids))
    with open(os.path.join(imagesets_path, 'trainval.txt'), 'w') as f:
        f.write('\n'.join(trainval_ids))
    with open(os.path.join(imagesets_path, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_ids))
    
    print("Generated ImageSets txt files")
    
    # Generate annotation files for training (using FOG images)
    fog_abs_path = os.path.abspath(fog_path)
    
    # Train annotations
    with open('2007_train.txt', 'w', encoding='utf-8') as f:
        for image_id in train_ids:
            img_path = os.path.join(fog_abs_path, f'{image_id}.jpg')
            if os.path.exists(img_path):
                f.write(img_path)
                convert_annotation(image_id, f)
                f.write('\n')
    
    # Val annotations
    nums_val = np.zeros(len(classes))
    with open('2007_val.txt', 'w', encoding='utf-8') as f:
        for image_id in val_ids:
            img_path = os.path.join(fog_abs_path, f'{image_id}.jpg')
            if os.path.exists(img_path):
                f.write(img_path)
                convert_annotation(image_id, f)
                f.write('\n')
    
    print(f"\nGenerated annotation files:")
    print(f"  2007_train.txt: {len(train_ids)} images")
    print(f"  2007_val.txt: {len(val_ids)} images")
    
    # Print class distribution
    print("\nClass distribution:")
    for i, cls in enumerate(classes):
        if nums[i] > 0:
            print(f"  {cls}: {int(nums[i])}")
    
    print("\n✅ Annotation generation complete!")
    print("Next step: Run train.py to start training")
