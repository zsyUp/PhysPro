import os
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
from collections import Counter

def parse_wsss4luad_training_filename(filename):
    """
    WSSS4LUAD
    : 1109944-30032-31530-[0, 0, 1].png
    : {'Tumor': 0, 'Stroma': 0, 'Normal': 1}
    """
    match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+)\]', filename)
    if match:
        a, b, c = map(int, match.groups())
        return {
            'Tumor': a,
            'Stroma': b,
            'Normal': c
        }
    return None

def organize_wsss4luad_training_by_class(src_dir, dst_dir, mode='single_label'):
    """
    WSSS4LUAD
    """
    class_names = ['Tumor', 'Stroma', 'Normal']

    for class_name in class_names:
        os.makedirs(os.path.join(dst_dir, class_name), exist_ok=True)

    image_files = [f for f in os.listdir(src_dir)
                   if f.endswith('.png') or f.endswith('.jpg')]

    stats = {cls: 0 for cls in class_names}
    multi_label_count = 0
    parse_failed = 0

    print(f"Processing {len(image_files)} training images...")

    for img_file in tqdm(image_files):
        labels = parse_wsss4luad_training_filename(img_file)
        if labels is None:
            parse_failed += 1
            continue

        present_classes = [cls for cls, val in labels.items() if val == 1]

        if len(present_classes) == 0:
            continue

        if mode == 'single_label':
            if len(present_classes) == 1:
                class_name = present_classes[0]
                src_path = os.path.join(src_dir, img_file)
                dst_path = os.path.join(dst_dir, class_name, img_file)
                shutil.copy2(src_path, dst_path)
                stats[class_name] += 1
            else:
                multi_label_count += 1

        elif mode == 'multi_label':
            for class_name in present_classes:
                src_path = os.path.join(src_dir, img_file)
                dst_path = os.path.join(dst_dir, class_name, img_file)
                shutil.copy2(src_path, dst_path)
                stats[class_name] += 1

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    for class_name in class_names:
        print(f"{class_name}: {stats[class_name]} images")
    if mode == 'single_label':
        print(f"\nSkipped multi-label images: {multi_label_count}")
    if parse_failed > 0:
        print(f"Failed to parse: {parse_failed} files")

def organize_wsss4luad_val_test_by_mask(src_dir, dst_dir, split='val'):
    """
    maskWSSS4LUAD/
    """
    class_names = ['Tumor', 'Stroma', 'Normal']

    palette_rgb = {
        'Tumor': (0, 64, 128),
        'Stroma': (64, 128, 0),
        'Normal': (243, 152, 0),
    }

    for class_name in class_names:
        os.makedirs(os.path.join(dst_dir, class_name), exist_ok=True)

    img_dir = os.path.join(src_dir, 'img')
    mask_dir = os.path.join(src_dir, 'mask')

    if not os.path.exists(img_dir):
        print(f"Warning: {img_dir} not found!")
        return

    if not os.path.exists(mask_dir):
        print(f"Warning: {mask_dir} not found!")
        return

    image_files = [f for f in os.listdir(img_dir)
                   if f.endswith('.png') or f.endswith('.jpg')]

    stats = {cls: 0 for cls in class_names}

    print(f"\nProcessing {len(image_files)} {split} images...")

    for img_file in tqdm(image_files):
        mask_path = os.path.join(mask_dir, img_file)
        if not os.path.exists(mask_path):
            continue

        mask = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask)

        for class_name, rgb in palette_rgb.items():
            color_mask = np.all(mask_np == rgb, axis=-1)
            pixel_count = np.sum(color_mask)

            if pixel_count > 0:
                src_path = os.path.join(img_dir, img_file)
                dst_path = os.path.join(dst_dir, class_name, img_file)
                shutil.copy2(src_path, dst_path)
                stats[class_name] += 1

    print(f"\n" + "=" * 60)
    print(f"{split.upper()} Statistics")
    print("=" * 60)
    for class_name in class_names:
        print(f"{class_name}: {stats[class_name]} images")

def calculate_dataset_statistics(data_dir):
    """


    """
    class_names = ['Tumor', 'Stroma', 'Normal']

    # RGB
    all_r, all_g, all_b = [], [], []
    total_pixels = 0

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        image_files = [f for f in os.listdir(class_dir)
                      if f.endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Loading {class_name} images for statistics...")
        sample_size = min(500, len(image_files))

        for img_file in tqdm(image_files[:sample_size]):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # RGB[0,1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            all_r.extend(img[:, :, 0].flatten())
            all_g.extend(img[:, :, 1].flatten())
            all_b.extend(img[:, :, 2].flatten())

            total_pixels += img.shape[0] * img.shape[1]

    if len(all_r) == 0:
        print("Error: No images loaded for statistics calculation!")
        return None, None

    # numpy
    all_r = np.array(all_r)
    all_g = np.array(all_g)
    all_b = np.array(all_b)

    mean = [all_r.mean(), all_g.mean(), all_b.mean()]
    std = [all_r.std(), all_g.std(), all_b.std()]

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total pixels analyzed: {total_pixels:,}")
    print(f"\nMEAN = {mean}")
    print(f"STD  = {std}")

    return mean, std

def analyze_wsss4luad_labels(train_dir):
    """
    WSSS4LUAD
    """
    print("\n" + "=" * 60)
    print("WSSS4LUAD Label Distribution Analysis")
    print("=" * 60)

    image_files = [f for f in os.listdir(train_dir)
                   if f.endswith(('.png', '.jpg', '.jpeg'))]

    label_combinations = []
    label_counts = {'Tumor': 0, 'Stroma': 0, 'Normal': 0}
    single_label_count = 0
    multi_label_count = 0
    single_label_stats = {'Tumor': 0, 'Stroma': 0, 'Normal': 0}

    for filename in image_files:
        match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+)\]', filename)
        if match:
            tumor, stroma, normal = map(int, match.groups())

            label_combo = f"[{tumor}, {stroma}, {normal}]"
            label_combinations.append(label_combo)

            if tumor == 1:
                label_counts['Tumor'] += 1
            if stroma == 1:
                label_counts['Stroma'] += 1
            if normal == 1:
                label_counts['Normal'] += 1

            num_labels = tumor + stroma + normal
            if num_labels == 1:
                single_label_count += 1
                if tumor == 1:
                    single_label_stats['Tumor'] += 1
                elif stroma == 1:
                    single_label_stats['Stroma'] += 1
                elif normal == 1:
                    single_label_stats['Normal'] += 1
            elif num_labels > 1:
                multi_label_count += 1

    print(f"\n: {len(image_files)}")

    print("\n (10):")
    combo_counter = Counter(label_combinations)
    for combo, count in combo_counter.most_common(10):
        percentage = (count / len(image_files)) * 100
        print(f"  {combo}: {count:5d} ({percentage:5.2f}%)")

    print("\n ():")
    for class_name, count in label_counts.items():
        percentage = (count / len(image_files)) * 100
        print(f"  {class_name:10s}: {count:5d} ({percentage:5.2f}%)")

    print("\n vs :")
    single_pct = (single_label_count / len(image_files)) * 100
    multi_pct = (multi_label_count / len(image_files)) * 100
    print(f"  : {single_label_count:5d} ({single_pct:5.2f}%)")
    print(f"  : {multi_label_count:5d} ({multi_pct:5.2f}%)")

    print("\n 'single_label' :")
    for class_name, count in single_label_stats.items():
        print(f"  {class_name:10s}: {count:5d}")

    print("\n:")
    min_single = min(single_label_stats.values())
    if single_pct >= 30 and min_single >= 100:
        print("    mode='single_label'")
    elif single_pct < 20:
        print("     mode='multi_label'")
    else:
        print("    :")
        print("     - : single_label ()")
        print("     - : multi_label")
    print("=" * 60)

if __name__ == "__main__":
    dataset_root = "./data/WSSS4LUAD/"

    analyze_wsss4luad_labels(os.path.join(dataset_root, "train"))

    print("\n" + "=" * 60)
    print("Organizing TRAINING set...")
    print("=" * 60)
    organize_wsss4luad_training_by_class(
        src_dir=os.path.join(dataset_root, "train"),
        dst_dir=os.path.join(dataset_root, "proto_train"),
        mode='single_label'
    )

    print("\n" + "=" * 60)
    print("Organizing VALIDATION set...")
    print("=" * 60)
    organize_wsss4luad_val_test_by_mask(
        src_dir=os.path.join(dataset_root, "val"),
        dst_dir=os.path.join(dataset_root, "proto_val"),
        split='val'
    )

    print("\n" + "=" * 60)
    print("Calculating dataset statistics...")
    print("=" * 60)
    mean, std = calculate_dataset_statistics(
        os.path.join(dataset_root, "proto_train")
    )
