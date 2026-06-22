# import os
# import shutil
# import cv2
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import re

# def parse_training_filename(filename):
#     """
#     : 1031280-2300-27920-[1 0 0 1].png
#     : {'TE': 1, 'NEC': 0, 'LYM': 0, 'TAS': 1}
#     """
#     #  [a b c d]
#     match = re.search(r'\[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]', filename)
#     if match:
#         a, b, c, d = map(int, match.groups())
#         return {
#             'TE': a,   # Tumor epithelial
#             'NEC': b,  # Necrosis
#             'LYM': c,  # Lymphocyte
#             'TAS': d   # Tumor-associated stroma
#         }
#     return None

# def organize_training_by_class(src_dir, dst_dir, mode='single_label'):
#     """

#     - src_dir:  (LUAD-HistoSeg/training/)
#     - dst_dir:  (: LUAD-HistoSeg/proto/)
#     - mode: 'single_label'  'multi_label'
#         - 'single_label':
#         - 'multi_label':
#     """
#     class_names = ['TE', 'NEC', 'LYM', 'TAS']

#
#     for class_name in class_names:
#         os.makedirs(os.path.join(dst_dir, class_name), exist_ok=True)

#
#     image_files = [f for f in os.listdir(src_dir)
#                    if f.endswith('.png') or f.endswith('.jpg')]

#     stats = {cls: 0 for cls in class_names}
#     multi_label_count = 0

#     print(f"Processing {len(image_files)} training images...")

#     for img_file in tqdm(image_files):
#         labels = parse_training_filename(img_file)
#         if labels is None:
#             continue

#
#         present_classes = [cls for cls, val in labels.items() if val == 1]

#         if len(present_classes) == 0:
#             continue

#         if mode == 'single_label':
#
#             if len(present_classes) == 1:
#                 class_name = present_classes[0]
#                 src_path = os.path.join(src_dir, img_file)
#                 dst_path = os.path.join(dst_dir, class_name, img_file)
#                 shutil.copy2(src_path, dst_path)
#                 stats[class_name] += 1
#             else:
#                 multi_label_count += 1

#         elif mode == 'multi_label':
#
#             for class_name in present_classes:
#                 src_path = os.path.join(src_dir, img_file)
#                 dst_path = os.path.join(dst_dir, class_name, img_file)
#                 shutil.copy2(src_path, dst_path)
#                 stats[class_name] += 1

#
#     print("\n=== Statistics ===")
#     for class_name in class_names:
#         print(f"{class_name}: {stats[class_name]} images")
#     if mode == 'single_label':
#         print(f"Skipped multi-label images: {multi_label_count}")

# def organize_val_test_by_mask(src_dir, dst_dir, split='val'):
#     """
#     mask/

#     - src_dir: / (LUAD-HistoSeg/val/  LUAD-HistoSeg/test/)
#     - dst_dir:
#     - split: 'val'  'test'
#     """
#     class_names = ['TE', 'NEC', 'LYM', 'TAS']

#
#     palette_rgb = {
#         'TE': (205, 51, 51),
#         'NEC': (0, 255, 0),
#         'LYM': (65, 105, 225),
#         'TAS': (255, 165, 0)
#     }

#
#     for class_name in class_names:
#         os.makedirs(os.path.join(dst_dir, class_name), exist_ok=True)

#     img_dir = os.path.join(src_dir, 'img')
#     mask_dir = os.path.join(src_dir, 'mask')

#     image_files = [f for f in os.listdir(img_dir)
#                    if f.endswith('.png') or f.endswith('.jpg')]

#     stats = {cls: 0 for cls in class_names}

#     print(f"\nProcessing {len(image_files)} {split} images...")

#     for img_file in tqdm(image_files):
#         mask_path = os.path.join(mask_dir, img_file)
#         if not os.path.exists(mask_path):
#             continue

#         # mask
#         mask = Image.open(mask_path).convert('RGB')
#         mask_np = np.array(mask)

#
#         for class_name, rgb in palette_rgb.items():
#             # mask
#             color_mask = np.all(mask_np == rgb, axis=-1)
#             pixel_count = np.sum(color_mask)

#
#             if pixel_count > 0:  #  > 100
#                 src_path = os.path.join(img_dir, img_file)
#                 dst_path = os.path.join(dst_dir, class_name, img_file)
#                 shutil.copy2(src_path, dst_path)
#                 stats[class_name] += 1

#
#     print(f"\n=== {split.upper()} Statistics ===")
#     for class_name in class_names:
#         print(f"{class_name}: {stats[class_name]} images")

# def calculate_dataset_statistics(data_dir):
#     """
#     """
#     class_names = ['TE', 'NEC', 'LYM', 'TAS']

#     all_images = []
#     for class_name in class_names:
#         class_dir = os.path.join(data_dir, class_name)
#         if not os.path.exists(class_dir):
#             continue

#         image_files = [f for f in os.listdir(class_dir)
#                       if f.endswith(('.png', '.jpg', '.jpeg'))]

#         print(f"Loading {class_name} images for statistics...")
#         for img_file in tqdm(image_files[:500]):
#             img_path = os.path.join(class_dir, img_file)
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             all_images.append(img / 255.0)

#     all_images = np.array(all_images)
#     mean = all_images.mean(axis=(0, 1, 2))
#     std = all_images.std(axis=(0, 1, 2))

#     print("\n=== Dataset Statistics ===")
#     print(f"MEAN = {mean.tolist()}")
#     print(f"STD = {std.tolist()}")

#     return mean, std

# if __name__ == "__main__":
#     # =====  =====
#     dataset_root = "./data/LUAD/"

#     # ===== 1:  =====
#     print("=" * 60)
#     print("Organizing TRAINING set by class...")
#     print("=" * 60)
#     organize_training_by_class(
#         src_dir=os.path.join(dataset_root, "train"),
#         dst_dir=os.path.join(dataset_root, "proto_train"),
#         mode='single_label'  # 'single_label'  'multi_label'

#     # ===== 2:  =====
#     print("\n" + "=" * 60)
#     print("Organizing VALIDATION set by class...")
#     print("=" * 60)
#     organize_val_test_by_mask(
#         src_dir=os.path.join(dataset_root, "val"),
#         dst_dir=os.path.join(dataset_root, "proto_val"),
#         split='val'

#     # ===== 3:  =====
#     # organize_val_test_by_mask(
#     #     src_dir=os.path.join(dataset_root, "test"),
#     #     dst_dir=os.path.join(dataset_root, "proto_test"),
#     #     split='test'
#     # )

#     # =====  =====
#     print("\n" + "=" * 60)
#     print("Calculating dataset statistics...")
#     print("=" * 60)
#     mean, std = calculate_dataset_statistics(
#         os.path.join(dataset_root, "proto_train")




## WSSS4LUAD
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import re

def parse_wsss4luad_training_filename(filename):
    """
    WSSS4LUAD
    : 1.2.276.0.7230010.3.1.4.8323329.5323.1517875224.440737_38208_17984_[1_1_0].png
    : {'Tumor': 1, 'Stroma': 1, 'Normal': 0}
    """
    #  [a_b_c]  (WSSS4LUAD)
    match = re.search(r'\[(\d+)_(\d+)_(\d+)\]', filename)
    if match:
        a, b, c = map(int, match.groups())
        return {
            'Tumor': a,   # Tumor epithelial
            'Stroma': b,  # Tumor-associated stroma
            'Normal': c   # Normal tissue
        }
    return None

def organize_wsss4luad_training_by_class(src_dir, dst_dir, mode='single_label'):
    """
    WSSS4LUAD

    :
    - src_dir:  (WSSS4LUAD/1.training/)
    - dst_dir:  (: WSSS4LUAD/proto_train/)
    - mode: 'single_label'  'multi_label'
        - 'single_label':
        - 'multi_label':
    """
    class_names = ['Tumor', 'Stroma', 'Normal']

    for class_name in class_names:
        os.makedirs(os.path.join(dst_dir, class_name), exist_ok=True)

    image_files = [f for f in os.listdir(src_dir)
                   if f.endswith('.png') or f.endswith('.jpg')]

    stats = {cls: 0 for cls in class_names}
    multi_label_count = 0

    print(f"Processing {len(image_files)} training images...")

    for img_file in tqdm(image_files):
        labels = parse_wsss4luad_training_filename(img_file)
        if labels is None:
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

    print("\n=== Statistics ===")
    for class_name in class_names:
        print(f"{class_name}: {stats[class_name]} images")
    if mode == 'single_label':
        print(f"Skipped multi-label images: {multi_label_count}")

def organize_wsss4luad_val_test_by_mask(src_dir, dst_dir, split='val'):
    """
    maskWSSS4LUAD/

    :
    - src_dir: / (WSSS4LUAD/2.validation/  WSSS4LUAD/3.testing/)
    - dst_dir:
    - split: 'val'  'test'
    """
    class_names = ['Tumor', 'Stroma', 'Normal']

    # WSSS4LUAD
    palette_rgb = {
        'Tumor': (0, 64, 128),      # label 0
        'Stroma': (64, 128, 0),     # label 1
        'Normal': (243, 152, 0),    # label 2
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

        # mask
        mask = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask)

        for class_name, rgb in palette_rgb.items():
            # mask
            color_mask = np.all(mask_np == rgb, axis=-1)
            pixel_count = np.sum(color_mask)

            #  > 0
            if pixel_count > 0:
                src_path = os.path.join(img_dir, img_file)
                dst_path = os.path.join(dst_dir, class_name, img_file)
                shutil.copy2(src_path, dst_path)
                stats[class_name] += 1

    print(f"\n=== {split.upper()} Statistics ===")
    for class_name in class_names:
        print(f"{class_name}: {stats[class_name]} images")

def calculate_dataset_statistics(data_dir):
    """

    """
    class_names = ['Tumor', 'Stroma', 'Normal']

    all_images = []
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            all_images.append(img / 255.0)

    if len(all_images) == 0:
        print("Error: No images loaded for statistics calculation!")
        return None, None

    all_images = np.array(all_images)
    mean = all_images.mean(axis=(0, 1, 2))
    std = all_images.std(axis=(0, 1, 2))

    print("\n=== Dataset Statistics ===")
    print(f"MEAN = {mean.tolist()}")
    print(f"STD = {std.tolist()}")

    return mean, std

if __name__ == "__main__":
    dataset_root = "./data/WSSS4LUAD/"  # WSSS4LUAD

    print("=" * 60)
    print("Organizing WSSS4LUAD TRAINING set by class...")
    print("=" * 60)
    organize_wsss4luad_training_by_class(
        src_dir=os.path.join(dataset_root, "train"),
        dst_dir=os.path.join(dataset_root, "proto_train"),
        mode='single_label'  # 'single_label'  'multi_label'
    )

    print("\n" + "=" * 60)
    print("Organizing WSSS4LUAD VALIDATION set by class...")
    print("=" * 60)
    organize_wsss4luad_val_test_by_mask(
        src_dir=os.path.join(dataset_root, "2.validation"),
        dst_dir=os.path.join(dataset_root, "proto_val"),
        split='val'
    )

    print("\n" + "=" * 60)
    print("Organizing WSSS4LUAD TEST set by class...")
    print("=" * 60)
    organize_wsss4luad_val_test_by_mask(
        src_dir=os.path.join(dataset_root, "3.testing"),
        dst_dir=os.path.join(dataset_root, "proto_test"),
        split='test'
    )

    print("\n" + "=" * 60)
    print("Calculating WSSS4LUAD dataset statistics...")
    print("=" * 60)
    mean, std = calculate_dataset_statistics(
        os.path.join(dataset_root, "proto_train")
    )
