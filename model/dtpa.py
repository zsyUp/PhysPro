# innovations/dtpa.py
"""
Adaptive-DTPA: Adaptive Differentiable Topological Persistence Anchoring
 -
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TopologicalDescriptorBranch(nn.Module):
    """ - """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes

        #  + MLP
        self.gap = nn.AdaptiveAvgPool2d(1)

        # MLP(0_weight, 1_weight)
        hidden_dim = max(64, in_channels // 4)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes * 2),  # 2
            nn.Sigmoid()  # [0,1]
        )

        nn.init.constant_(self.mlp[-2].bias, 0.5)

    def forward(self, deep_features):
        """
        Args:
            deep_features: (B, C, H, W)
        Returns:
            topo_weights: (B, num_classes, 2) - [0, 1]
        """
        B = deep_features.size(0)
        global_feat = self.gap(deep_features).view(B, -1)  # (B, C)

        weights = self.mlp(global_feat)  # (B, num_classes*2)
        topo_weights = weights.view(B, self.num_classes, 2)

        return topo_weights


class PersistentHomologyExtractor(nn.Module):
    """ - """
    def __init__(self, img_size=512, base_persistence_threshold=0.1):
        super().__init__()
        self.img_size = img_size
        self.base_threshold = base_persistence_threshold

    def extract_persistence_diagram(self, prob_map, adaptive_threshold=None):
        """
         -

        Args:
            prob_map: (H, W)  [0, 1]
            adaptive_threshold:
        Returns:
            pd: list of (birth, death, dim, persistence)
        """
        pers_threshold = adaptive_threshold if adaptive_threshold is not None else self.base_threshold

        thresholds = torch.linspace(0.1, 0.9, 9, device=prob_map.device)

        pd_points = []

        for t in thresholds:
            binary = (prob_map > t).float()
            beta0, beta1 = self.compute_betti_numbers(binary)

            lifetime = 1.0 - t.item()

            if beta0 > 0 and lifetime > pers_threshold:
                pd_points.append((t.item(), 1.0, 0, lifetime))  # 0
            if beta1 > 0 and lifetime > pers_threshold:
                pd_points.append((t.item(), 1.0, 1, lifetime))  # 1

        return pd_points

    def compute_betti_numbers(self, binary_mask):
        """Betti"""
        from scipy import ndimage
        labeled, beta0 = ndimage.label(binary_mask.cpu().numpy())

        foreground = binary_mask.sum().item()

        kernel = torch.tensor([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]], device=binary_mask.device).float()

        mask_pad = F.pad(binary_mask.unsqueeze(0).unsqueeze(0).float(), (1,1,1,1), mode='constant', value=0)
        neighbors = F.conv2d(mask_pad, kernel.unsqueeze(0).unsqueeze(0), padding=0).squeeze()

        boundary = ((binary_mask > 0) & (neighbors < 4)).sum().item()
        chi = foreground - boundary
        beta1 = max(0, beta0 - chi)

        return beta0, beta1


class AdaptiveTopologicalAnchoringLoss(nn.Module):
    """ - """
    def __init__(self):
        super().__init__()

    def weighted_wasserstein_distance(self, pd_pred, beta0_weight, beta1_weight):
        """
        Wasserstein -

        Args:
            pd_pred:  [(birth, death, dim, persistence), ...]
            beta0_weight: 0 [0,1]
            beta1_weight: 1 [0,1]
        Returns:
            distance:
        """
        if len(pd_pred) == 0:
            return torch.tensor(0.0, device='cuda')

        pred_beta0 = sum([1 for p in pd_pred if p[2] == 0])
        pred_beta1 = sum([1 for p in pd_pred if p[2] == 1])

        target_beta0 = 1.0 if beta0_weight > 0.5 else max(1, pred_beta0)
        target_beta1 = 1.0 if beta1_weight > 0.5 else 0.0

        cost_beta0 = beta0_weight * (pred_beta0 - target_beta0) ** 2
        cost_beta1 = beta1_weight * (pred_beta1 - target_beta1) ** 2

        persistence_penalty = 0.0
        for p in pd_pred:
            if p[3] < 0.2:
                weight = beta0_weight if p[2] == 0 else beta1_weight
                persistence_penalty += weight * (0.2 - p[3]) ** 2

        total_cost = cost_beta0 + cost_beta1 + 0.1 * persistence_penalty

        return torch.tensor(total_cost, dtype=torch.float32, device='cuda')

    def forward(self, pred_mask, topo_weights, extractor):
        """

        """
        B, C, H, W = pred_mask.shape

        losses = []
        per_class_losses = []

        for b in range(B):
            for c in range(C):
                prob_map = pred_mask[b, c]
                beta0_w = topo_weights[b, c, 0].item()
                beta1_w = topo_weights[b, c, 1].item()

                adaptive_threshold = 0.05 + 0.15 * min(beta0_w, beta1_w)
                pd_pred = extractor.extract_persistence_diagram(
                    prob_map,
                    adaptive_threshold=adaptive_threshold
                )

                loss_bc = self.weighted_wasserstein_distance(pd_pred, beta0_w, beta1_w)
                losses.append(loss_bc)
                per_class_losses.append((c, loss_bc.item()))

        if len(losses) == 0:
            return torch.tensor(0.0, device=pred_mask.device, requires_grad=True)

        # per_class_lossesaux_outputs
        return torch.stack(losses).mean()


class MonteCarloConsistencyRegularizer(nn.Module):
    """Monte Carlo - """
    def __init__(self, num_samples=3, dropout_rate=0.3):
        super().__init__()
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

    def forward(self, model, inputs, cls_labels):
        """


        Args:
            model:
            inputs: (B, 3, H, W)
            cls_labels: (B, C)
        Returns:
            consistency_loss:
        """
        model.train()  # Dropout

        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                outputs = model(inputs, cls_labels)
                cam = outputs[7]  # cam4
                pred_prob = torch.sigmoid(cam)
                predictions.append(pred_prob)

        pred_stack = torch.stack(predictions, dim=0)  # (S, B, C, H, W)
        pred_std = pred_stack.std(dim=0)  # (B, C, H, W)

        confidence_mask = (pred_stack.mean(dim=0) > 0.7).float()
        consistency_loss = (pred_std * confidence_mask).mean()

        return consistency_loss


class GatedLossModule(nn.Module):
    """ - """
    def __init__(self):
        super().__init__()

    def forward(self, topo_loss, cls_labels, class_idx=None):
        """
        1

        Args:
            topo_loss:
            cls_labels: (B, C)
            class_idx:
        Returns:
            gated_loss:
        """
        B, C = cls_labels.shape

        if class_idx is not None:
            gate = cls_labels[:, class_idx].float()  # (B,)
            gated_loss = topo_loss * gate.mean()
        else:
            gate = cls_labels.float()  # (B, C)
            gated_loss = topo_loss * gate.mean()

        return gated_loss


class GradientBackpropModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_mask, topo_loss):
        pred_soft = torch.sigmoid(pred_mask)

        kernel_size = 5
        sigma = 1.0

        channels = pred_soft.shape[1]
        kernel = self._get_gaussian_kernel(kernel_size, sigma, channels, pred_soft.device)

        pred_smooth = F.conv2d(
            pred_soft,
            kernel,
            padding=kernel_size//2,
            groups=channels
        )

        smoothness_loss = ((pred_soft - pred_smooth) ** 2).mean()

        total_loss = topo_loss + 0.1 * smoothness_loss

        return total_loss

    def _get_gaussian_kernel(self, kernel_size, sigma, channels, device):
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel


class AdaptiveDTPA(nn.Module):
    """
    Adaptive-DTPA

    1.  -
    2.  -
    3. Monte Carlo -
    """
    def __init__(self, num_classes=4, in_channels=512, warmup_epochs=3,
                 persistence_threshold=0.1, enable_mc_consistency=True):
        super().__init__()

        self.topo_descriptor = TopologicalDescriptorBranch(in_channels, num_classes)

        self.topo_loss_fn = AdaptiveTopologicalAnchoringLoss()

        self.enable_mc_consistency = enable_mc_consistency
        if enable_mc_consistency:
            self.mc_regularizer = MonteCarloConsistencyRegularizer(num_samples=3)

        self.gated_loss = GatedLossModule()

        self.gradient_module = GradientBackpropModule()

        self.extractor = PersistentHomologyExtractor(
            img_size=512,
            base_persistence_threshold=persistence_threshold
        )

        self.warmup_epochs = warmup_epochs
        self.register_buffer('current_epoch', torch.tensor(0))

        self.num_classes = num_classes

    def set_epoch(self, epoch):
        """epoch"""
        self.current_epoch.fill_(epoch)

    def forward(self, pred_logits, deep_features, cls_labels,
                class_idx=None, current_iter=None, iters_per_epoch=None,
                model=None, inputs=None):
        """
        Args:
            pred_logits: (B, C, H, W) logits
            deep_features: (B, C_deep, H, W)
            cls_labels: (B, C)
            class_idx:
            current_iter:
            iters_per_epoch: epoch
            model: MC
            inputs: MC
        """
        # epoch
        if current_iter is not None and iters_per_epoch is not None:
            computed_epoch = current_iter // iters_per_epoch
            self.current_epoch.fill_(computed_epoch)

        if self.current_epoch.item() < self.warmup_epochs:
            return torch.tensor(0.0, device=pred_logits.device, requires_grad=True), {
                'topo_loss_raw': 0.0,
                'status': f'warmup ({self.current_epoch.item()}/{self.warmup_epochs})',
                'topo_weights_mean': [0.0, 0.0]
            }

        topo_weights = self.topo_descriptor(deep_features)  # (B, num_classes, 2)

        pred_prob = torch.sigmoid(pred_logits)

        if class_idx is not None:
            pred_mask = pred_prob[:, class_idx:class_idx+1]
            topo_weights_subset = topo_weights[:, class_idx:class_idx+1, :]
            topo_loss = self.topo_loss_fn(pred_mask, topo_weights_subset, self.extractor)

            topo_loss = self.gated_loss(topo_loss, cls_labels, class_idx)
        else:
            topo_loss = self.topo_loss_fn(pred_prob, topo_weights, self.extractor)

            topo_loss = self.gated_loss(topo_loss, cls_labels)

        consistency_loss = torch.tensor(0.0, device=pred_logits.device)
        if self.enable_mc_consistency and model is not None and inputs is not None:
            consistency_loss = self.mc_regularizer(model, inputs, cls_labels)

        total_loss = self.gradient_module(pred_logits, topo_loss)
        total_loss = total_loss + 0.05 * consistency_loss

        aux_outputs = {
            'topo_loss_raw': topo_loss.item() if isinstance(topo_loss, torch.Tensor) else topo_loss,
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0.0,
            'status': f'active (epoch {self.current_epoch.item()})',
            'topo_weights_mean': topo_weights.mean(dim=0).mean(dim=0).tolist()
        }

        return total_loss, aux_outputs
