import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import pickle as pkl
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class FrequencyKMeans:
    """


    """
    def __init__(self, n_clusters, beta=0.5, gamma=0.5,
                 max_iter=100, random_state=None, use_phase_constraint=True):
        """

        - n_clusters:
        - beta:
        - gamma:
        - use_phase_constraint:
        """
        self.n_clusters = n_clusters
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_phase_constraint = use_phase_constraint
        np.random.seed(random_state)

        self.phase_centers_ = None
        self.amplitude_centers_ = None

    def normalize_features(self, features):
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        return (features - mean) / std

    def fuse_features(self, phase_feat, amplitude_feat):
        phase_norm = self.normalize_features(phase_feat)
        amplitude_norm = self.normalize_features(amplitude_feat)

        fused = np.concatenate([
            self.beta * phase_norm,
            self.gamma * amplitude_norm
        ], axis=1)

        return fused

    def compute_phase_distance(self, feat1, feat2):
        return np.linalg.norm(feat1 - feat2)

    def fit_predict(self, phase_features, amplitude_features):
        """

        1.
        2.
        """
        n_samples = phase_features.shape[0]

        fused_features = self.fuse_features(phase_features, amplitude_features)

        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        fused_centers = fused_features[idx]

        phase_centers = phase_features[idx]
        amplitude_centers = amplitude_features[idx]

        for iteration in range(self.max_iter):
            similarities = cosine_similarity(fused_features, fused_centers)
            labels = np.argmax(similarities, axis=1)

            if self.use_phase_constraint:
                phase_threshold = np.percentile(
                    [self.compute_phase_distance(phase_features[i], phase_centers[labels[i]])
                     for i in range(n_samples)], 75
                )

                for i in range(n_samples):
                    current_label = labels[i]
                    phase_dist = self.compute_phase_distance(
                        phase_features[i], phase_centers[current_label]
                    )

                    if phase_dist > phase_threshold:
                        phase_distances = [
                            self.compute_phase_distance(phase_features[i], phase_centers[k])
                            for k in range(self.n_clusters)
                        ]
                        labels[i] = np.argmin(phase_distances)

            old_fused_centers = fused_centers.copy()

            for k in range(self.n_clusters):
                cluster_mask = labels == k
                if cluster_mask.sum() > 0:
                    fused_centers[k] = fused_features[cluster_mask].mean(axis=0)
                    phase_centers[k] = phase_features[cluster_mask].mean(axis=0)
                    amplitude_centers[k] = amplitude_features[cluster_mask].mean(axis=0)

                    fused_centers[k] /= (np.linalg.norm(fused_centers[k]) + 1e-8)

            if np.allclose(old_fused_centers, fused_centers, atol=1e-4):
                break

        self.phase_centers_ = torch.from_numpy(phase_centers)
        self.amplitude_centers_ = torch.from_numpy(amplitude_centers)

        final_similarities = cosine_similarity(fused_features, fused_centers)

        return labels, final_similarities


def cluster_frequency_features(frequency_path, k_list, beta=0.5, gamma=0.5,
                               use_phase_constraint=True):
    """


    :
    - frequency_path:
    - k_list:  [TUM, STR, LYM, NEC]
    - beta:
    - gamma:
    """
    print(f"Loading frequency features from {frequency_path}...")
    with open(frequency_path, 'rb') as f:
        frequency_dict = pkl.load(f)

    if len(k_list) != 4:
        raise ValueError("k_list must contain 4 values for TUM, STR, LYM, and NEC respectively")

    all_phase_centers = []
    all_amplitude_centers = []

    class_order = ['TUM', 'STR', 'LYM', 'NEC']

    print(f"\n{'='*60}")
    print(f"Frequency Domain Clustering Configuration:")
    print(f"  beta (phase weight): {beta}")
    print(f"  gamma (amplitude weight): {gamma}")
    print(f"  phase constraint: {use_phase_constraint}")
    print(f"{'='*60}\n")

    for class_name, k in zip(class_order, k_list):
        print(f"\n{'='*20} Class: {class_name} (k={k}) {'='*20}")

        phase_features = []
        amplitude_features = []
        class_names = []

        for item in frequency_dict[class_name]:
            phase_features.append(item['phase_features'])
            amplitude_features.append(item['amplitude_features'])
            class_names.append(item['name'])

        phase_array = np.array(phase_features)
        amplitude_array = np.array(amplitude_features)

        print(f"Sample count: {len(phase_features)}")
        print(f"Phase feature dim: {phase_array.shape[1]}")
        print(f"Amplitude feature dim: {amplitude_array.shape[1]}")

        frequency_kmeans = FrequencyKMeans(
            n_clusters=k,
            beta=beta,
            gamma=gamma,
            random_state=42,
            use_phase_constraint=use_phase_constraint
        )

        cluster_labels, similarities = frequency_kmeans.fit_predict(
            phase_array, amplitude_array
        )

        all_phase_centers.append(frequency_kmeans.phase_centers_)
        all_amplitude_centers.append(frequency_kmeans.amplitude_centers_)

        for cluster_idx in range(k):
            cluster_mask = cluster_labels == cluster_idx
            cluster_count = cluster_mask.sum()

            print(f"\nCluster {cluster_idx + 1}:")
            print(f"  Total samples: {cluster_count}")

            cluster_phase = phase_array[cluster_mask]
            cluster_amplitude = amplitude_array[cluster_mask]

            print(f"  Phase feature variance: {cluster_phase.var():.4f}")
            print(f"  Amplitude feature variance: {cluster_amplitude.var():.4f}")

            cluster_indices = np.where(cluster_mask)[0]
            fused_sim = similarities[cluster_mask][:, cluster_idx]
            top_5_indices = np.argsort(fused_sim)[-5:][::-1]

            print(f"  Top 5 samples closest to center:")
            for idx, sample_idx in enumerate(top_5_indices, 1):
                global_idx = cluster_indices[sample_idx]
                print(f"    {idx}. {class_names[global_idx]} (similarity: {fused_sim[sample_idx]:.4f})")

    all_phase_centers_tensor = torch.cat(all_phase_centers, dim=0)
    all_amplitude_centers_tensor = torch.cat(all_amplitude_centers, dim=0)

    save_info = {
        'phase_prototypes': all_phase_centers_tensor,
        'amplitude_prototypes': all_amplitude_centers_tensor,
        'k_list': k_list,
        'class_order': class_order,
        'cumsum_k': np.cumsum([0] + k_list),
        'frequency_params': {
            'beta': beta,
            'gamma': gamma
        }
    }

    save_dir = "./features/image_features"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"bcss_frequency_prototypes_{k_list[0]}{k_list[1]}{k_list[2]}{k_list[3]}.pkl")

    with open(save_path, 'wb') as f:
        pkl.dump(save_info, f)

    print(f"\n{'='*60}")
    print(f"Frequency prototypes saved to {save_path}")
    print(f"Prototype shapes:")
    print(f"  - Phase prototypes: {all_phase_centers_tensor.shape}")
    print(f"  - Amplitude prototypes: {all_amplitude_centers_tensor.shape}")
    print(f"K list: {k_list}")
    print(f"Cumulative sum of k: {save_info['cumsum_k']}")
    print("Class prototypes index ranges:")
    for i, class_name in enumerate(class_order):
        start_idx = save_info['cumsum_k'][i]
        end_idx = save_info['cumsum_k'][i+1]
        print(f"  {class_name}: {start_idx} to {end_idx}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Cluster frequency domain features to generate frequency prototypes'
    )
    parser.add_argument('--frequency_features', type=str,
                       default='./features/image_features/bcss_features_frequency.pkl',
                       help='Path to frequency features file')
    parser.add_argument('--k_list', type=int, nargs=4, default=[3, 3, 3, 3],
                       help='Number of clusters for each class [TUM, STR, LYM, NEC]')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Weight for phase features (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Weight for amplitude features (default: 0.5)')
    parser.add_argument('--use_phase_constraint', action='store_true', default=True,
                       help='Use phase constraint to force separation (default: True)')

    args = parser.parse_args()

    total_weight = args.beta + args.gamma
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: weights sum to {total_weight}, normalizing...")
        args.beta /= total_weight
        args.gamma /= total_weight

    cluster_frequency_features(
        frequency_path=args.frequency_features,
        k_list=args.k_list,
        beta=args.beta,
        gamma=args.gamma,
        use_phase_constraint=args.use_phase_constraint
    )
