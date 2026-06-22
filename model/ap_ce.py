
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyQualityEvaluator(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

        self.freq_weight_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Tanh()
        )

        self.scale = nn.Parameter(torch.tensor(0.3))
        self.bias = nn.Parameter(torch.tensor(0.6))

    def compute_frequency_features(self, cam, features):
        """CAM"""
        B, C, H, W = cam.shape

        feat_mean = features.mean(dim=1, keepdim=True)
        feat_fft = torch.fft.rfft2(feat_mean.float(), norm='ortho')
        feat_magnitude = torch.abs(feat_fft)

        quality_features = []

        for c in range(C):
            cam_c = cam[:, c:c+1]
            cam_fft = torch.fft.rfft2(cam_c.float(), norm='ortho')
            cam_magnitude = torch.abs(cam_fft)

            consistency = F.cosine_similarity(
                cam_magnitude.flatten(1),
                feat_magnitude.flatten(1),
                dim=1
            )

            energy = cam_magnitude ** 2
            total_energy = energy.sum(dim=[1, 2, 3])

            h_cutoff = H // 4
            w_cutoff = W // 4
            low_freq_energy = energy[:, :, :h_cutoff, :w_cutoff].sum(dim=[1, 2, 3])
            compactness = low_freq_energy / (total_energy + 1e-8)

            high_freq_energy = energy[:, :, h_cutoff:, w_cutoff:].sum(dim=[1, 2, 3])
            sharpness = high_freq_energy / (total_energy + 1e-8)

            feat_c = torch.stack([consistency, compactness, sharpness], dim=1)
            quality_features.append(feat_c)

        quality_features = torch.stack(quality_features, dim=1)
        return quality_features

    def forward(self, cam, features):
        """
        :
            cam: [B, num_classes, H, W]
            features: [B, C, H, W]
        :
            quality_scores: [B, num_classes]
        """
        freq_features = self.compute_frequency_features(cam, features)

        B, C, _ = freq_features.shape
        freq_features_flat = freq_features.view(B * C, 3)

        raw_scores = self.freq_weight_net(freq_features_flat)
        quality_scores_flat = self.bias + self.scale * raw_scores
        quality_scores = quality_scores_flat.view(B, C)

        quality_scores = torch.clamp(quality_scores, min=0.2, max=0.95)

        return quality_scores


class AdaptivePrototypeBank(nn.Module):
    def __init__(self, num_classes=4, feat_dim=512, num_prototypes_per_class=3, momentum=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.K = num_prototypes_per_class
        self.momentum = momentum

        self.register_buffer(
            'prototypes',
            torch.randn(num_classes, self.K, feat_dim)
        )
        self.prototypes = F.normalize(self.prototypes, dim=2)

        self.register_buffer(
            'prototype_confidence',
            torch.ones(num_classes, self.K) * 0.5
        )

        self.register_buffer(
            'update_counts',
            torch.zeros(num_classes, self.K)
        )

    def get_best_prototype(self, class_idx):
        confidences = self.prototype_confidence[class_idx]
        best_idx = confidences.argmax()
        return self.prototypes[class_idx, best_idx]

    def get_dynamic_threshold(self, quality_scores, cls_labels, current_epoch):
        if current_epoch < 5:
            return 0.20  # 0.25  0.20
        elif current_epoch < 10:
            return 0.20 + (current_epoch - 5) * 0.02  # 0.03  0.02
        else:
            valid_scores = quality_scores[cls_labels > 0]
            if len(valid_scores) > 0:
                threshold = torch.quantile(valid_scores.float(), 0.20).item()  # 0.25  0.20
                return max(threshold, 0.25)  # 0.30  0.25
            else:
                return 0.30  # 0.35  0.30

    def update_prototypes(self, features, cam, quality_scores, cls_labels,
                         current_epoch=1, quality_threshold=None):
        B, C_feat, H, W = features.shape
        _, num_classes, _, _ = cam.shape

        if quality_threshold is None:
            quality_threshold = self.get_dynamic_threshold(
                quality_scores, cls_labels, current_epoch
            )

        pseudo_labels = cam.argmax(dim=1)

        for class_idx in range(num_classes):
            if (cls_labels[:, class_idx] == 0).all():
                continue

            high_quality_mask = quality_scores[:, class_idx] > quality_threshold
            if high_quality_mask.sum() == 0:
                continue

            for b in range(B):
                if not high_quality_mask[b] or cls_labels[b, class_idx] == 0:
                    continue

                class_mask = (pseudo_labels[b] == class_idx)
                if class_mask.sum() == 0:
                    continue

                feat_b = features[b]
                class_features = feat_b[:, class_mask]

                if class_features.shape[1] == 0:
                    continue

                feat_mean = class_features.mean(dim=1)
                feat_mean = F.normalize(feat_mean, dim=0)

                similarities = F.cosine_similarity(
                    feat_mean.unsqueeze(0),
                    self.prototypes[class_idx],
                    dim=1
                )

                best_proto_idx = similarities.argmax()

                old_proto = self.prototypes[class_idx, best_proto_idx]
                new_proto = self.momentum * old_proto + (1 - self.momentum) * feat_mean
                self.prototypes[class_idx, best_proto_idx] = F.normalize(new_proto, dim=0)

                old_conf = self.prototype_confidence[class_idx, best_proto_idx]
                quality = quality_scores[b, class_idx].item()
                self.prototype_confidence[class_idx, best_proto_idx] = \
                    self.momentum * old_conf + (1 - self.momentum) * quality

                self.update_counts[class_idx, best_proto_idx] += 1


class APCE(nn.Module):
    """
     APCE v2.3
    """
    def __init__(self, num_classes=4, feat_dim=512, num_prototypes_per_class=3,
                 momentum=0.9, use_frequency_feedback=True, initial_temperature=0.07):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.quality_evaluator = FrequencyQualityEvaluator(num_classes)
        self.prototype_bank = AdaptivePrototypeBank(
            num_classes, feat_dim, num_prototypes_per_class, momentum
        )

        self.use_frequency_feedback = use_frequency_feedback
        self.low_freq_ratio = 0.3

        #  : temperature0.07
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        self.temperature_min = 0.05
        self.temperature_max = 0.15

    def generate_frequency_mask(self, H, W, device):
        """mask"""
        h_cutoff = int(H * self.low_freq_ratio)
        w_cutoff = int((W // 2 + 1) * self.low_freq_ratio)

        mask_low = torch.zeros(H, W // 2 + 1, device=device)
        mask_low[:h_cutoff, :w_cutoff] = 1.0

        mask_high = 1.0 - mask_low

        return mask_low, mask_high

    def get_adaptive_temperature(self):
        temp = torch.clamp(self.temperature, self.temperature_min, self.temperature_max)
        return temp

    def generate_cam_feedback_v2(self, cam, quality_scores, mask_low, mask_high):
        B, C, H, W = cam.shape

        avg_quality = quality_scores.mean()

        if avg_quality > 0.50:
            return cam

        cam_fft = torch.fft.rfft2(cam.float(), norm='ortho')
        feedback_fft = torch.zeros_like(cam_fft)

        for c in range(C):
            quality_c = quality_scores[:, c:c+1, None, None]
            need_correction = (quality_c < 0.65).float()
            low_freq_boost = need_correction * (1.0 - quality_c) * 0.3

            cam_fft_c = cam_fft[:, c:c+1]
            feedback_fft[:, c:c+1] = cam_fft_c * (
                mask_low * (1 + low_freq_boost) +
                mask_high * 1.0
            )

        cam_feedback = torch.fft.irfft2(feedback_fft, s=(H, W), norm='ortho')

        return cam_feedback.to(cam.dtype)

    def contrastive_prototype_loss_v3(self, features, cam, quality_scores, cls_labels):
        """
         : InfoNCE +
        """
        B, C_feat, H, W = features.shape
        pseudo_labels = cam.argmax(dim=1)

        temperature = self.get_adaptive_temperature()

        total_loss = 0.0
        valid_samples = 0

        for class_idx in range(self.num_classes):
            proto_c = self.prototype_bank.get_best_prototype(class_idx)
            proto_c_norm = F.normalize(proto_c, dim=0)

            for b in range(B):
                if cls_labels[b, class_idx] == 0:
                    continue

                if quality_scores[b, class_idx] < 0.17:
                    continue

                class_mask = (pseudo_labels[b] == class_idx)
                if class_mask.sum() == 0:
                    continue

                feat_b = features[b]
                class_features = feat_b[:, class_mask]

                if class_features.shape[1] == 0:
                    continue

                class_features = F.normalize(class_features, dim=0)
                class_feat_mean = class_features.mean(dim=1)
                class_feat_mean = F.normalize(class_feat_mean, dim=0)

                pos_sim = torch.dot(proto_c_norm, class_feat_mean)

                neg_sims = []
                for neg_class_idx in range(self.num_classes):
                    if neg_class_idx == class_idx:
                        continue

                    proto_neg = self.prototype_bank.get_best_prototype(neg_class_idx)
                    proto_neg_norm = F.normalize(proto_neg, dim=0)
                    neg_sim = torch.dot(proto_neg_norm, class_feat_mean)
                    neg_sims.append(neg_sim)

                if len(neg_sims) == 0:
                    continue

                neg_sims = torch.stack(neg_sims)

                # InfoNCE
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / temperature
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)

                #  loss clamp
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                loss = torch.clamp(loss, max=5.0)  # loss

                total_loss += loss
                valid_samples += 1

        if valid_samples == 0:
            return torch.tensor(0.0, device=features.device)

        avg_loss = total_loss / valid_samples

        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            return torch.tensor(0.0, device=features.device)

        return avg_loss

    def forward(self, cam, features, pseudo_labels, cls_labels,
                current_epoch=1, update_proto=True, **kwargs):
        """
         APCE
        """
        B, num_classes, H, W = cam.shape

        # Step 1: CAM
        quality_scores = self.quality_evaluator(cam, features)

        # Step 2:
        if update_proto and self.training:
            self.prototype_bank.update_prototypes(
                features.detach(),
                cam.detach(),
                quality_scores.detach(),
                cls_labels,
                current_epoch=current_epoch
            )

        # Step 3:
        contrastive_loss = self.contrastive_prototype_loss_v3(
            features, cam, quality_scores, cls_labels
        )

        # Step 4:
        mask_low, mask_high = self.generate_frequency_mask(H, W, cam.device)

        with torch.no_grad():
            if self.use_frequency_feedback:
                cam_feedback = self.generate_cam_feedback_v2(
                    cam, quality_scores, mask_low, mask_high
                )
            else:
                cam_feedback = cam

        feedback_strength = (cam_feedback - cam.detach()).abs().mean()

        total_loss = contrastive_loss

        aux_outputs = {
            'quality_scores': quality_scores.detach(),
            'cam_feedback': cam_feedback,
            'mask_low': mask_low,
            'mask_high': mask_high,
            'feedback_strength': feedback_strength,
            'prototype_confidence': self.prototype_bank.prototype_confidence.mean(),
            'temperature': self.get_adaptive_temperature().detach(),
            'contrastive_loss': contrastive_loss.detach()
        }

        return total_loss, aux_outputs
