import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import torch
from PIL import Image
import pickle as pkl
from torchvision import transforms
from tqdm import tqdm
import cv2 as cv
from utils.pyutils import set_seed
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np

def get_transform():
    MEAN = [0.66791496, 0.47791372, 0.70623304]
    STD = [0.1736589, 0.22564577, 0.19820057]

    transform = A.Compose([
        A.Normalize(MEAN, STD),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        ToTensorV2(transpose_mask=True),
    ])
    return transform


def extract_phase_features(img_tensor):
    """

    Stain-Invariant
    """
    C, H, W = img_tensor.shape
    phase_features_list = []

    for c in range(C):
        fft_result = torch.fft.fft2(img_tensor[c])
        phase = torch.angle(fft_result)
        phase_reconstructed = torch.fft.ifft2(torch.exp(1j * phase))
        phase_reconstructed = torch.abs(phase_reconstructed)

        phase_mean = phase_reconstructed.mean()
        phase_std = phase_reconstructed.std()
        phase_max = phase_reconstructed.max()
        phase_min = phase_reconstructed.min()

        phase_features_list.extend([
            phase_mean.item(),
            phase_std.item(),
            phase_max.item(),
            phase_min.item()
        ])

    return np.array(phase_features_list)


def extract_amplitude_features(img_tensor):
    """


    """
    C, H, W = img_tensor.shape
    amplitude_features_list = []

    for c in range(C):
        fft_result = torch.fft.fft2(img_tensor[c])
        amplitude = torch.abs(fft_result)
        amplitude_log = torch.log(amplitude + 1e-8)

        center_h, center_w = H // 2, W // 2
        radius_low = min(H, W) // 8
        low_freq_mask = torch.zeros_like(amplitude_log, dtype=torch.bool)
        for i in range(H):
            for j in range(W):
                if (i - center_h) ** 2 + (j - center_w) ** 2 <= radius_low ** 2:
                    low_freq_mask[i, j] = True

        low_freq_values = amplitude_log[low_freq_mask]
        low_freq_mean = low_freq_values.mean()
        low_freq_std = low_freq_values.std()

        high_freq_mask = ~low_freq_mask
        high_freq_values = amplitude_log[high_freq_mask]
        high_freq_mean = high_freq_values.mean()
        high_freq_std = high_freq_values.std()

        global_mean = amplitude_log.mean()
        global_std = amplitude_log.std()

        amplitude_features_list.extend([
            low_freq_mean.item(),
            low_freq_std.item(),
            high_freq_mean.item(),
            high_freq_std.item(),
            global_mean.item(),
            global_std.item()
        ])

    return np.array(amplitude_features_list)


def extract_frequency_features(image_dir, spatial_features_path):
    """


    :
    - image_dir:
    - spatial_features_path:
    """
    set_seed(42)

    print(f"Loading spatial features from {spatial_features_path} to get filename list...")
    with open(spatial_features_path, 'rb') as f:
        spatial_dict = pkl.load(f)

    frequency_dict = {}

    for class_name in ['NEC', 'LYM', 'STR', 'TUM']:
        class_path = os.path.join(image_dir, class_name)
        frequency_dict[class_name] = []

        print(f"\nProcessing {class_name} images for frequency features...")

        for spatial_item in tqdm(spatial_dict[class_name], desc=f"{class_name}", ncols=100):
            img_name = spatial_item['name']
            img_path = os.path.join(class_path, img_name)

            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found, skipping...")
                continue

            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

            transform = get_transform()
            img = transform(image=img)["image"]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            img_cpu = img.cpu()
            phase_features = extract_phase_features(img_cpu)
            amplitude_features = extract_amplitude_features(img_cpu)

            frequency_dict[class_name].append({
                'name': img_name,
                'phase_features': phase_features,
                'amplitude_features': amplitude_features
            })

    save_dir = "./features/image_features"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "bcss_features_frequency.pkl")

    with open(save_path, 'wb') as f:
        pkl.dump(frequency_dict, f)

    print(f"\n{'='*60}")
    print(f"Frequency features saved to {save_path}")
    print(f"Feature dimensions:")
    sample_item = frequency_dict['TUM'][0]
    print(f"  - Phase features: {len(sample_item['phase_features'])} dims")
    print(f"  - Amplitude features: {len(sample_item['amplitude_features'])} dims")
    print(f"\nSample counts per class:")
    for class_name in ['TUM', 'STR', 'LYM', 'NEC']:
        print(f"  {class_name}: {len(frequency_dict[class_name])} samples")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract frequency domain features (phase + amplitude)'
    )
    parser.add_argument('--image_dir', type=str,
                       default='./data/BCSS-WSSS/proto/',
                       help='Directory containing organized images')
    parser.add_argument('--spatial_features', type=str,
                       default='./image_features/bcss_features_pro.pkl',
                       help='Path to existing spatial features file')

    args = parser.parse_args()

    extract_frequency_features(
        image_dir=args.image_dir,
        spatial_features_path=args.spatial_features
    )
