import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedSpatialExpert(nn.Module):
    """Spatial expert that mixes local and dilated depthwise responses."""
    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.selection_fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)

        b, c, _, _ = x.shape
        pooling = self.gap(feat3 + feat5).view(b, c)
        weights = self.selection_fc(pooling).view(b, 2, 1, 1, 1)

        return feat3 * weights[:, 0] + feat5 * weights[:, 1]


class WaveletPhaseExpert(nn.Module):
    """Frequency expert combining wavelet detail, phase, and amplitude priors."""
    def __init__(self, channels):
        super().__init__()
        self.hf_refine = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        )
        self.phase_atten = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

        self.amp_prototype_align = nn.Sequential(
            nn.Conv2d(1, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def get_wav_decom(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) / 4.0
        lh = (x00 - x01 + x10 - x11) / 4.0
        hl = (x00 + x01 - x10 - x11) / 4.0
        hh = (x00 - x01 - x10 + x11) / 4.0
        return ll, torch.cat([lh, hl, hh], dim=1)

    def forward(self, x, amplitude_prior=None):
        """Refine features with spectral cues and optional amplitude guidance."""
        B, C, H, W = x.shape
        x_float = x.float()

        ll, hfs = self.get_wav_decom(x_float)
        hfs_feat = self.hf_refine(hfs)
        hfs_up = F.interpolate(hfs_feat, size=(H, W), mode='bilinear', align_corners=False)
        ll_up = F.interpolate(ll, size=(H, W), mode='bilinear', align_corners=False)

        x_fft = torch.fft.rfft2(x_float, norm='ortho')
        mag = torch.abs(x_fft)
        pha = torch.angle(x_fft)

        edge_guidance_float = self.phase_atten(x_float).float()
        mag_modulation = 1 + torch.fft.rfft2(edge_guidance_float, norm='ortho').abs()

        if amplitude_prior is not None:
            amp_prior_resized = F.interpolate(
                amplitude_prior,
                size=(mag.shape[-2], mag.shape[-1]),
                mode='bilinear',
                align_corners=False
            )
            amp_align_weight = self.amp_prototype_align(amp_prior_resized)
            mag = mag * (mag_modulation * (1 + 0.3 * amp_align_weight))
        else:
            mag = mag * mag_modulation

        x_complex = torch.polar(mag, pha)
        x_res = torch.fft.irfft2(x_complex, s=(H, W), norm='ortho')

        result = (ll_up + hfs_up + x_res) / 3.0

        if torch.isnan(result).any() or torch.isinf(result).any():
            result = x_float

        return result.to(x.dtype)


class SEEGating(nn.Module):
    """Semantic-entropy gating module for balancing spatial and spectral experts."""
    def __init__(self, low_ch, high_ch, num_classes=4, temperature=1.0):
        super().__init__()
        self.low_ch = low_ch
        self.high_ch = high_ch
        self.temperature = temperature
        self.num_classes = num_classes

        self.low_feat_downsample = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(low_ch),
            nn.GELU(),
            nn.Conv2d(low_ch, low_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(low_ch),
            nn.GELU()
        )

        self.semantic_aggregator = nn.Sequential(
            nn.Conv2d(3, 16, 1),  # 3 -> 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),  # ->
            nn.Sigmoid()
        )

        #  low_ch + high_ch + 1
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(low_ch + high_ch + 1, (low_ch + high_ch + 1) // 4, 1),
            nn.GELU(),
            nn.Conv2d((low_ch + high_ch + 1) // 4, 1, 1)
        )

        self.entropy_map = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def compute_spectral_entropy(self, x):
        x_float = x.float()

        x_freq = torch.fft.rfft2(x_float, norm='ortho')
        power = torch.abs(x_freq) ** 2
        p_uv = power / (power.sum(dim=[1, 2, 3], keepdim=True) + 1e-8)
        entropy = -(p_uv * torch.log(p_uv + 1e-8)).sum(dim=[1, 2, 3])

        max_entropy = torch.log(torch.tensor(x.shape[-2] * x.shape[-1], dtype=torch.float32, device=x.device))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy.mean()

    def aggregate_semantic_evidence(self, guidance_maps, target_size):
        """


        Args:
            guidance_maps:  'spatial', 'phase', 'amplitude'
                           (B, Classes, H, W)
            target_size:  (H_target, W_target)

        Returns:
            semantic_confidence: (B, 1, H_target, W_target)
        """
        confidence_maps = []

        for domain in ['spatial', 'phase', 'amplitude']:
            if domain in guidance_maps and guidance_maps[domain] is not None:
                #  Classes
                conf, _ = torch.max(guidance_maps[domain], dim=1, keepdim=True)  # (B, 1, H, W)

                conf = F.interpolate(conf, size=target_size, mode='bilinear', align_corners=False)
                confidence_maps.append(conf)

        if len(confidence_maps) == 0:
            B = 1
            return torch.ones(B, 1, *target_size, device=next(self.parameters()).device)

        #  (B, 3, H, W)
        stacked_conf = torch.cat(confidence_maps, dim=1)

        semantic_confidence = self.semantic_aggregator(stacked_conf)  # (B, 1, H, W)

        return semantic_confidence

    def forward(self, x, x_low=None, guidance_maps=None, pseudo_labels=None, cls_labels=None):
        """
        Args:
            x:  Stage 4 (B, high_ch, H, W)
            x_low:  Stage 1 (B, low_ch, H_low, W_low)
            guidance_maps:  'spatial', 'phase', 'amplitude'
        """
        B, _, H, W = x.shape

        #  x_low
        if x_low is not None:
            H_spec = self.compute_spectral_entropy(x_low)
        else:
            H_spec = self.compute_spectral_entropy(x)

        entropy_gain = self.entropy_map(H_spec.view(1, 1)).view(1, 1, 1, 1)

        if x_low is not None:
            x_low_down = self.low_feat_downsample(x_low)  # (B, low_ch, H, W)
            x_multiscale = torch.cat([x, x_low_down], dim=1)  # (B, low_ch+high_ch, H, W)
        else:
            x_multiscale = torch.cat([x, x], dim=1)  # (B, high_ch*2, H, W)

        if guidance_maps is not None:
            semantic_confidence = self.aggregate_semantic_evidence(guidance_maps, (H, W))  # (B, 1, H, W)
        else:
            semantic_confidence = torch.ones(B, 1, H, W, device=x.device)

        x_guided = torch.cat([x_multiscale, semantic_confidence], dim=1)  # (B, low_ch+high_ch+1, H, W)

        gamma_base = torch.sigmoid(self.gate_conv(x_guided) / self.temperature)

        #  gamma:   ( + )
        gamma = gamma_base.expand(B, 1, H, W) * (0.5 + 0.5 * entropy_gain) * (0.7 + 0.3 * semantic_confidence)
        gamma = torch.clamp(gamma, min=0.08)

        return gamma, H_spec


class SEEMoE(nn.Module):
    """- SEE-MoE v4.0"""
    def __init__(self, low_ch, high_ch, num_classes=4, enable_apce_coordination=True):
        super().__init__()
        self.low_ch = low_ch
        self.high_ch = high_ch

        self.spatial_expert = AdvancedSpatialExpert(high_ch)
        self.spectral_expert = WaveletPhaseExpert(high_ch)
        self.gating = SEEGating(low_ch=low_ch, high_ch=high_ch, num_classes=num_classes)

        self.fusion = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, 3, padding=1),
            nn.BatchNorm2d(high_ch),
            nn.GELU(),
            nn.Conv2d(high_ch, high_ch, 1)
        )

        self.alpha = nn.Parameter(torch.tensor(0.6))
        self.enable_apce_coordination = enable_apce_coordination

        self.diversity_config = {
            'warmup_epochs': 8,
            'target_similarity': 0.5,
            'max_weight': 0.01,
            'ortho_weight': 0.0001,
            'penalty_max_clamp': 0.40
        }

        print(f"\n SEE-MoE v4.0 Configuration:")
        print(f"   Low channels (Stage 1): {low_ch}")
        print(f"   High channels (Stage 4): {high_ch}")
        print(f"   Gate input channels: {low_ch + high_ch + 1} (multi-scale + semantic)")
        print(f"   Routing: Semantic-Physical Dual-Drive")
        print(f"   Entropy source: Stage 1 (high-res)\n")

    def _compute_diversity_loss_internal(self, F_spat, F_spec, current_epoch=0):
        """diversity loss ()"""
        return torch.tensor(0.0, device=F_spat.device, requires_grad=True)

    def forward(self, x, current_epoch=0, apce_quality_scores=None, x_low=None, guidance_maps=None):
        """
        Args:
            x:  Stage 4 (B, high_ch, H, W)
            x_low:  Stage 1 (B, low_ch, H_low, W_low)
            guidance_maps: :
                - 'spatial': (B, Classes, H, W)
                - 'phase': (B, Classes, H, W)
                - 'amplitude': (B, Classes, H, W)
        """
        F_spat = self.spatial_expert(x)

        amplitude_prior = None
        if guidance_maps is not None and 'amplitude' in guidance_maps:
            amplitude_prior, _ = torch.max(guidance_maps['amplitude'], dim=1, keepdim=True)

        F_spec = self.spectral_expert(x, amplitude_prior=amplitude_prior)

        gamma, H_spec = self.gating(x, x_low=x_low, guidance_maps=guidance_maps)

        # APCE
        if self.enable_apce_coordination and apce_quality_scores is not None:
            B = x.shape[0]

            if apce_quality_scores.shape[0] == B:
                avg_quality = apce_quality_scores.mean(dim=1, keepdim=True)
            elif apce_quality_scores.shape[0] > B:
                avg_quality = apce_quality_scores[:B].mean(dim=1, keepdim=True)
            else:
                avg_quality = None

            if avg_quality is not None:
                quality_map = avg_quality[:, :, None, None]
                adjustment = torch.where(
                    quality_map < 0.5,
                    quality_map,
                    torch.ones_like(quality_map)
                )
                gamma = gamma * adjustment

        alpha = torch.sigmoid(self.alpha)
        F_combined = alpha * F_spat + (1 - alpha) * (gamma * F_spec) + 0.1 * x

        output = self.fusion(F_combined)

        diversity_loss = self._compute_diversity_loss_internal(
            F_spat.detach(),
            F_spec.detach(),
            current_epoch=current_epoch
        )

        aux_outputs = {
            'entropy': H_spec,
            'gamma_mean': gamma.mean(),
            'gamma_full': gamma,
            'F_spat': F_spat.detach(),
            'F_spec': F_spec.detach(),
            'spectral_contribution': (gamma * F_spec).abs().mean(),
            'spatial_contribution': (alpha * F_spat).abs().mean(),
            'diversity_loss': diversity_loss,
            'diversity_weight': self.diversity_config['max_weight'] * min(current_epoch / self.diversity_config['warmup_epochs'], 1.0)
        }

        return output, aux_outputs

