import os
import shutil
from tqdm import tqdm
import re

def extract_label_from_filename(filename):
    """

    : TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+101[1100].png
    [1100] :
    0 (): Tumor (TUM)
    1: Stroma (STR)
    2: Lymphocytic infiltrate (LYM)
    3 (): Necrosis (NEC)
    """
    match = re.search(r'\[([01]{4})\]\.png$', filename)
    if match:
        binary_string = match.group(1)
        return binary_string
    return None


def map_labels_to_classes(binary_string):
    """

    : '1100' -> ['TUM', 'STR']

    """
    label_map = {
        0: 'TUM',  # Tumor
        1: 'STR',  # Stroma
        2: 'LYM',  # Lymphocytic infiltrate
        3: 'NEC'   # Necrosis
    }

    classes = []
    for i, bit in enumerate(binary_string):
        if bit == '1':
            classes.append(label_map[i])

    return classes


def organize_images_by_class(source_dir, output_dir, copy_mode='primary'):
    """


    :
    - source_dir:  (: 'data/BCSS-WSSS/train')
    - output_dir:  (: 'data/BCSS-WSSS/proto')
    - copy_mode:
        'primary' - 1
        'all' - 1
        'majority' - 11
    """

    classes = ['TUM', 'STR', 'LYM', 'NEC']
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

    print(f"\nFound {len(image_files)} images in {source_dir}")
    print(f"Copy mode: {copy_mode}")
    print(f"Output directory: {output_dir}\n")

    stats = {class_name: 0 for class_name in classes}
    skipped = 0
    multi_label_skipped = 0

    for img_name in tqdm(image_files, desc="Organizing images", ncols=100):
        binary_string = extract_label_from_filename(img_name)

        if binary_string is None:
            skipped += 1
            continue

        classes_in_image = map_labels_to_classes(binary_string)

        if not classes_in_image:
            skipped += 1
            continue

        if copy_mode == 'primary':
            primary_class = classes_in_image[0]
            src_path = os.path.join(source_dir, img_name)
            dst_path = os.path.join(output_dir, primary_class, img_name)
            shutil.copy2(src_path, dst_path)
            stats[primary_class] += 1

        elif copy_mode == 'majority':
            if len(classes_in_image) == 1:
                cls = classes_in_image[0]
                src_path = os.path.join(source_dir, img_name)
                dst_path = os.path.join(output_dir, cls, img_name)
                shutil.copy2(src_path, dst_path)
                stats[cls] += 1
            else:
                multi_label_skipped += 1

        elif copy_mode == 'all':
            for cls in classes_in_image:
                src_path = os.path.join(source_dir, img_name)
                dst_path = os.path.join(output_dir, cls, img_name)
                shutil.copy2(src_path, dst_path)
                stats[cls] += 1

    print(f"\n{'='*60}")
    print(f"Organization completed!")
    print(f"{'='*60}")
    print(f"Statistics:")
    for class_name in classes:
        count = stats[class_name]
        class_dir = os.path.join(output_dir, class_name)
        actual_count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
        print(f"  {class_name}: {actual_count} images")

    if skipped > 0:
        print(f"\nSkipped {skipped} images (no valid label found or all zeros)")
    if multi_label_skipped > 0:
        print(f"Skipped {multi_label_skipped} multi-label images (mode=majority)")
    print(f"{'='*60}\n")


def analyze_label_distribution(source_dir):
    """
    copy_mode
    """
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

    label_map = {
        0: 'TUM',
        1: 'STR',
        2: 'LYM',
        3: 'NEC'
    }

    single_label_count = 0
    multi_label_count = 0
    zero_label_count = 0
    label_combinations = {}
    class_counts = {'TUM': 0, 'STR': 0, 'LYM': 0, 'NEC': 0}

    print(f"\nAnalyzing label distribution in {source_dir}...")
    print(f"Total images: {len(image_files)}\n")

    for img_name in tqdm(image_files, desc="Analyzing", ncols=100):
        binary_string = extract_label_from_filename(img_name)

        if binary_string:
            num_ones = binary_string.count('1')

            if num_ones == 0:
                zero_label_count += 1
            elif num_ones == 1:
                single_label_count += 1
            else:
                multi_label_count += 1

            label_combinations[binary_string] = label_combinations.get(binary_string, 0) + 1

            for i, bit in enumerate(binary_string):
                if bit == '1':
                    cls = label_map[i]
                    class_counts[cls] += 1

    print(f"\n{'='*60}")
    print(f"Label Distribution Analysis:")
    print(f"{'='*60}")
    print(f"Images with no label (0000): {zero_label_count}")
    print(f"Single-label images: {single_label_count}")
    print(f"Multi-label images: {multi_label_count}")
    print(f"\nClass occurrence counts:")
    for cls in ['TUM', 'STR', 'LYM', 'NEC']:
        count = class_counts[cls]
        percentage = (count / len(image_files)) * 100 if len(image_files) > 0 else 0
        print(f"  {cls}: {count} images ({percentage:.2f}%)")

    print(f"\nTop 10 label combinations:")
    sorted_combinations = sorted(label_combinations.items(), key=lambda x: x[1], reverse=True)
    for binary, count in sorted_combinations[:10]:
        classes_in_combo = [label_map[i] for i, bit in enumerate(binary) if bit == '1']
        if classes_in_combo:
            combo_str = '+'.join(classes_in_combo)
        else:
            combo_str = 'NONE'
        percentage = (count / len(image_files)) * 100 if len(image_files) > 0 else 0
        print(f"  [{binary}] {combo_str}: {count} images ({percentage:.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Organize BCSS-WSSS training images by tissue class'
    )
    parser.add_argument('--source_dir', type=str,
                       default='data/BCSS-WSSS/train',
                       help='Source directory containing training images')
    parser.add_argument('--output_dir', type=str,
                       default='data/BCSS-WSSS/proto',
                       help='Output directory for organized images')
    parser.add_argument('--mode', type=str,
                       choices=['primary', 'majority', 'all'],
                       default='primary',
                       help='Copy mode: primary (first label), majority (single-label only), all (copy to all classes)')
    parser.add_argument('--analyze', action='store_true',
                       help='Only analyze label distribution without copying files')

    args = parser.parse_args()

    if args.analyze:
        analyze_label_distribution(args.source_dir)
    else:
        organize_images_by_class(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            copy_mode=args.mode
        )
