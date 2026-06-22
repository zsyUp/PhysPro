
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from model.segform import mix_transformer
from see_moe import SEEMoE
from ap_ce import APCE
from utils.hierarchical_utils import merge_to_parent_predictions, expand_parent_to_subclass_labels
import numpy as np


class AdaptiveLayer(nn.Module):
    """

    : backbone

    :
        in_dim:
        n_ratio:
        out_dim: backbone
    """
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class FrequencyFeatureExtractor(nn.Module):
    """

    :

    :
        - :
        - :
    """
    def __init__(self):
        super().__init__()

    def extract_phase_features(self, img_tensor):
        """


        : [B, C, H, W]
        : [B, C*4, H, W]

        4: mean, std, max, min
        """
        B, C, H, W = img_tensor.shape
        phase_features_list = []

        for c in range(C):
            fft_result = torch.fft.fft2(img_tensor[:, c])

            phase = torch.angle(fft_result)

            phase_reconstructed = torch.fft.ifft2(torch.exp(1j * phase))
            phase_reconstructed = torch.abs(phase_reconstructed)

            phase_mean = phase_reconstructed.mean(dim=[1, 2], keepdim=True)
            phase_std = phase_reconstructed.std(dim=[1, 2], keepdim=True)
            phase_max = phase_reconstructed.amax(dim=[1, 2], keepdim=True)
            phase_min = phase_reconstructed.amin(dim=[1, 2], keepdim=True)

            #  [B, 1, 1] -> [B, 1, H, W]
            phase_mean = phase_mean.unsqueeze(-1).expand(B, 1, H, W)
            phase_std = phase_std.unsqueeze(-1).expand(B, 1, H, W)
            phase_max = phase_max.unsqueeze(-1).expand(B, 1, H, W)
            phase_min = phase_min.unsqueeze(-1).expand(B, 1, H, W)

            phase_features_list.append(torch.cat([phase_mean, phase_std, phase_max, phase_min], dim=1))

        # [B, C*4, H, W]
        return torch.cat(phase_features_list, dim=1)

    def extract_amplitude_features(self, img_tensor):
        """


        : [B, C, H, W]
        : [B, C*6, H, W]

        6:
            (mean, std) + (mean, std) + (mean, std)
        """
        B, C, H, W = img_tensor.shape
        amplitude_features_list = []

        for c in range(C):
            fft_result = torch.fft.fft2(img_tensor[:, c])

            # log
            amplitude = torch.abs(fft_result)
            amplitude_log = torch.log(amplitude + 1e-8)

            center_h, center_w = H // 2, W // 2
            radius_low = min(H, W) // 8

            # mask
            y_coords = torch.arange(H, device=img_tensor.device).view(-1, 1).expand(H, W)
            x_coords = torch.arange(W, device=img_tensor.device).view(1, -1).expand(H, W)
            low_freq_mask = ((y_coords - center_h) ** 2 + (x_coords - center_w) ** 2) <= radius_low ** 2
            low_freq_mask = low_freq_mask.unsqueeze(0).expand(B, H, W)

            low_freq_values = amplitude_log.masked_select(low_freq_mask).view(B, -1)
            # low_freq_mean = low_freq_values.mean(dim=1, keepdim=True).unsqueeze(-1).expand(B, 1, H, W)
            low_freq_mean = low_freq_values.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).expand(B, 1, H, W)
            low_freq_std = low_freq_values.std(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).expand(B, 1, H, W)

            high_freq_mask = ~low_freq_mask
            high_freq_values = amplitude_log.masked_select(high_freq_mask).view(B, -1)
            high_freq_mean = high_freq_values.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).expand(B, 1, H, W)
            high_freq_std = high_freq_values.std(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).expand(B, 1, H, W)

            global_mean = amplitude_log.mean(dim=[1, 2], keepdim=True).unsqueeze(-1).expand(B, 1, H, W)
            global_std = amplitude_log.std(dim=[1, 2], keepdim=True).unsqueeze(-1).expand(B, 1, H, W)
            amplitude_features_list.append(torch.cat([
                low_freq_mean, low_freq_std,
                high_freq_mean, high_freq_std,
                global_mean, global_std
            ], dim=1))

        # [B, C*6, H, W]
        return torch.cat(amplitude_features_list, dim=1)

    def forward(self, x):
        """
        : [B, C, H, W]
        :
            phase_features: [B, C*4, H, W]
            amplitude_features: [B, C*6, H, W]
        """
        phase_features = self.extract_phase_features(x)
        amplitude_features = self.extract_amplitude_features(x)
        return phase_features, amplitude_features


class HierarchicalQualityPropagator(nn.Module):
    """

    : CAM

    :
        CAM
    """
    def __init__(self, k_list):
        """
        :
            k_list:  [K_TUM, K_STR, K_LYM, K_NEC]
        """
        super().__init__()
        self.k_list = k_list
        self.num_parent = len(k_list)

    def propagate_quality(self, quality_parent, cam_parent, cam_child):
        """


        :
            quality_parent: [B, num_parent] - APCE
            cam_parent: [B, num_parent, H, W] - CAM
            cam_child: [B, total_child, H, W] - CAM

        :
            quality_child: [B, total_child] -
            propagation_weight: [B, total_child, H, W] -
        """
        B, num_parent, H, W = cam_parent.shape
        total_child = sum(self.k_list)

        quality_child = []
        propagation_weight = []

        start_idx = 0
        for parent_idx, k in enumerate(self.k_list):
            q_parent = quality_parent[:, parent_idx:parent_idx+1]  # [B, 1]

            cam_p = cam_parent[:, parent_idx:parent_idx+1]  # [B, 1, H, W]
            cam_p_norm = F.softmax(cam_p.view(B, 1, -1), dim=2).view(B, 1, H, W)

            # CAM
            cam_c = cam_child[:, start_idx:start_idx+k]  # [B, k, H, W]

            cam_c_pooled = F.adaptive_avg_pool2d(cam_c, 1).view(B, k)  # [B, k]
            cam_p_pooled = F.adaptive_avg_pool2d(cam_p, 1).view(B, 1)  # [B, 1]

            #  = sigmoid( * )
            consistency = torch.sigmoid(cam_c_pooled * cam_p_pooled)  # [B, k]

            q_child = q_parent * consistency  # [B, k]
            quality_child.append(q_child)

            weight = cam_p_norm.expand(B, k, H, W) * q_parent.view(B, 1, 1, 1)
            propagation_weight.append(weight)

            start_idx += k

        quality_child = torch.cat(quality_child, dim=1)  # [B, total_child]
        propagation_weight = torch.cat(propagation_weight, dim=1)  # [B, total_child, H, W]

        return quality_child, propagation_weight


class ImprovedClsNetwork(nn.Module):
    """
     - -

    :
        1. Backbone: SegFormer4stage
        2. : MedCLIP  CAM
        3. :
           -   CAM
           -   CAM
        4. SEE-MoE: -
        5. APCE:
        6. :
    """
    def __init__(self,
                 backbone='mit_b1',
                 cls_num_classes=4,
                 stride=[4, 2, 2, 1],
                 pretrained=True,
                 n_ratio=0.5,
                 l_fea_path=None,
                 freq_proto_path=None,
                 enable_frequency_branch=True,
                 enable_see_moe=True,
                 enable_ap_ce=True,
                 enable_dtpa=False,
                 enable_adaptive_dtpa=False):
        """
        :
            backbone:
            cls_num_classes: 4: TUM, STR, LYM, NEC
            stride: stage
            pretrained:
            n_ratio:
            l_fea_path: .pkl
            freq_proto_path: .pkl
            enable_frequency_branch:
            enable_see_moe: SEE-MoE
            enable_ap_ce: APCE
            enable_dtpa: DTPA
        """
        super().__init__()

        self.cls_num_classes = cls_num_classes
        self.stride = stride
        self.enable_frequency_branch = enable_frequency_branch

        # 1. Backbone
        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims  # [64, 128, 320, 512] for mit_b1
        C4 = self.in_channels[3]  # Stage 4
        feat_dim = C4

        if pretrained:
            state_dict = torch.load(f'./pretrained/{backbone}.pth', map_location="cpu")
            state_dict.pop('head.weight', None)
            state_dict.pop('head.bias', None)
            state_dict = {k: v for k, v in state_dict.items()
                         if k in self.encoder.state_dict().keys()}
            self.encoder.load_state_dict(state_dict, strict=False)

        self.pooling = F.adaptive_avg_pool2d
        self.current_epoch = 1

        print("\n" + "="*80)
        print("...")

        # 512stage
        self.l_fc1 = AdaptiveLayer(512, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(512, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(512, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(512, n_ratio, self.in_channels[3])

        with open(f"./features/image_features/{l_fea_path}.pkl", "rb") as lf:
            info = pkl.load(lf)
            self.l_fea = info['features'].cpu()  #  [total_classes, 512]
            self.k_list = info['k_list']  #  [K_TUM, K_STR, K_LYM, K_NEC]
            self.cumsum_k = info['cumsum_k']  #  [0, K_TUM, K_TUM+K_STR, ...]

        self.total_classes = sum(self.k_list)
        print(f"  : {self.l_fea.shape}")
        print(f"  : {self.k_list} ({self.total_classes})")
        print(f"  TUM: {self.cumsum_k[0]}~{self.cumsum_k[1]}")
        print(f"  STR: {self.cumsum_k[1]}~{self.cumsum_k[2]}")
        print(f"  LYM: {self.cumsum_k[2]}~{self.cumsum_k[3]}")
        print(f"  NEC: {self.cumsum_k[3]}~{self.cumsum_k[4]}")

        if self.enable_frequency_branch:
            print("\n...")

            self.freq_extractor = FrequencyFeatureExtractor()

            if freq_proto_path is None:
                freq_proto_path = f"bcss_frequency_prototypes_{''.join(map(str, self.k_list))}"

            with open(f"./features/features/image_features/{freq_proto_path}.pkl", "rb") as ff:
                freq_info = pkl.load(ff)
                self.phase_prototypes = freq_info['phase_prototypes'].cpu()  # [total_classes, 12]
                self.amplitude_prototypes = freq_info['amplitude_prototypes'].cpu()  # [total_classes, 18]

            print(f"  : {self.phase_prototypes.shape}")
            print(f"  : {self.amplitude_prototypes.shape}")

            # Stage 4
            # : 34=12  C4
            # : 36=18  C4
            phase_feat_dim = 12  # 3 channels * 4 stats
            amplitude_feat_dim = 18  # 3 channels * 6 stats

            self.phase_fc4 = AdaptiveLayer(phase_feat_dim, n_ratio, C4)
            self.amplitude_fc4 = AdaptiveLayer(amplitude_feat_dim, n_ratio, C4)

            # 12/18C4
            self.phase_proto_fc4 = AdaptiveLayer(12, n_ratio, C4)
            self.amplitude_proto_fc4 = AdaptiveLayer(18, n_ratio, C4)

            self.alpha_spatial = nn.Parameter(torch.tensor(0.6))
            self.beta_phase = nn.Parameter(torch.tensor(0.2))
            self.gamma_amplitude = nn.Parameter(torch.tensor(0.2))

            print(f"  : spatial={0.6}, phase={0.2}, amplitude={0.2}")

        print("="*80 + "\n")

        self.logit_scale1 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale2 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)

        # 5. SEE-MoE
        self.enable_moe = False
        self.current_iter = 0
        self.enable_see_moe = enable_see_moe

        if enable_see_moe:
            print("SEE-MoE...")
            self.see_moe = SEEMoE(
                low_ch=self.in_channels[0],
                high_ch=C4,
                num_classes=cls_num_classes,
                enable_apce_coordination=True  # APCE-SEE
            )
            self.last_quality_scores = None  # APCE
            print("  SEE-MoE: -")
            print("    -  + ")
            print("    - :  +  + ")

        # 6. APCE
        self.enable_ap_ce = enable_ap_ce
        if enable_ap_ce:
            print("APCE...")
            self.ap_ce = APCE(
                num_classes=cls_num_classes,
                feat_dim=feat_dim,
                num_prototypes_per_class=3,
                momentum=0.9,
                use_frequency_feedback=True,
                initial_temperature=0.07
            )
            print("  APCE:  +  + ")

        self.quality_propagator = HierarchicalQualityPropagator(self.k_list)
        print("  ")

        # 8. DTPA
        self.enable_dtpa = enable_dtpa or enable_adaptive_dtpa
        if self.enable_dtpa:
            from .dtpa import AdaptiveDTPA
            self.dtpa = AdaptiveDTPA(
                num_classes=cls_num_classes,
                in_channels=C4,
                warmup_epochs=8,
                persistence_threshold=0.08,
                enable_mc_consistency=True
            )
            print("  DTPA")

        print("="*80 + "\n")

    def get_param_groups(self):
        """

        bias1Dweight decay
        """
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized},
                {'params': not_regularized, 'weight_decay': 0.}]

    def expand_parent_to_subclass_cam(self, cam_parent_refined, cam_child_original):
        """
        CAM

        :

        :
            cam_parent_refined: [B, num_parent, H, W] - APCECAM
            cam_child_original: [B, total_child, H, W] - CAM

        :
            cam_child_refined: [B, total_child, H, W] - CAM
        """
        cam_child_refined = []
        start_idx = 0

        for parent_idx, k in enumerate(self.k_list):
            # CAM
            parent_cam = cam_parent_refined[:, parent_idx:parent_idx+1]  # [B, 1, H, W]

            # CAM
            child_cams = cam_child_original[:, start_idx:start_idx+k]  # [B, k, H, W]

            # CAM
            parent_weight = torch.sigmoid(parent_cam)

            child_cams_refined = child_cams * parent_weight

            cam_child_refined.append(child_cams_refined)
            start_idx += k

        return torch.cat(cam_child_refined, dim=1)

    def frequency_domain_collaborative_refinement(self, cam_child, apce_feedback, mask_low, mask_high):
        """


        : APCE + SEE-MoE

        :
            cam_child: [B, total_child, H, W] - CAM
            apce_feedback: [B, total_child, H, W] - APCE
            mask_low: [H, W//2+1] - mask
            mask_high: [H, W//2+1] - mask

        :
            cam_final: [B, total_child, H, W] - CAM
        """
        B, C, H, W = cam_child.shape

        cam_child_detached = cam_child.detach()
        apce_feedback_detached = apce_feedback.detach()

        cam_fft = torch.fft.rfft2(cam_child_detached.float(), norm='ortho')
        feedback_fft = torch.fft.rfft2(apce_feedback_detached.float(), norm='ortho')

        # - : APCE
        # - : CAM
        mask_low_expanded = mask_low.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        mask_high_expanded = mask_high.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

        cam_fft_refined = (
            feedback_fft * mask_low_expanded +  # APCE
            cam_fft * mask_high_expanded        # SEE
        )

        cam_refined = torch.fft.irfft2(cam_fft_refined, s=(H, W), norm='ortho')

        refinement_delta = cam_refined - cam_child_detached
        cam_final = cam_child + refinement_delta.detach()

        return cam_final.to(cam_child.dtype)

    def forward(self, x, cls_labels, update_apce_proto=True):
        """
         - -

        :
            x: [B, 3, H, W] -
            cls_labels: [B, num_classes] -
            update_apce_proto: bool - APCE

        :
            :
                - cls1~4: stagelogits
                - cam1~4: stageCAM
                - cam4_child_final: CAM
                - cam4_parent_final: CAM
                - see_moe_aux: SEE-MoE
                - apce_aux: APCE
        """

        # Stage 1-4: Backbone
        _x, _attns = self.encoder(x)
        # _x[0]: [B, C1, H/4, W/4]
        # _x[1]: [B, C2, H/8, W/8]
        # _x[2]: [B, C3, H/16, W/16]
        # _x[3]: [B, C4, H/32, W/32]

        scales = [self.logit_scale1, self.logit_scale2, self.logit_scale3, self.logit_scale4]
        imshape = [_.shape for _ in _x]

        l_fea = self.l_fea.to(x.device)  # [total_classes, 512]
        l_fcs = [self.l_fc1, self.l_fc2, self.l_fc3, self.l_fc4]
        projected_l_fea = [fc(l_fea) for fc in l_fcs]
        # projected_l_fea[i]: [total_classes, Ci]

        # Stage 1-3: CAM
        outputs = []
        for i in range(3):
            feat = _x[i].permute(0, 2, 3, 1).reshape(-1, _x[i].shape[1])  # [B*H*W, Ci]
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)

            logits = scales[i] * feat @ projected_l_fea[i].t().float()  # [B*H*W, total_classes]

            #  logits [B*H*W, classes] [B, classes, H, W]
            #  $(x, y)$
            out = logits.view(imshape[i][0], imshape[i][2], imshape[i][3], -1).permute(0, 3, 1, 2)
            # out: [B, total_classes, H, W]

            # logits
            cls_res = self.pooling(out, (1, 1)).view(-1, self.total_classes)

            outputs.append((cls_res, out))

        cls1, cam1 = outputs[0]
        cls2, cam2 = outputs[1]
        cls3, cam3 = outputs[2]

        # Stage 4: -
        F4_raw = _x[3]  # [B, C4, H/32, W/32]
        see_moe_aux = {}
        apce_aux = {}

        # 4.1 CAM
        _x4_init = F4_raw.permute(0, 2, 3, 1).reshape(-1, F4_raw.shape[1])
        _x4_init = _x4_init / (_x4_init.norm(dim=-1, keepdim=True) + 1e-8)
        logits4_spatial_init = scales[3] * _x4_init @ projected_l_fea[3].t().float()
        cam4_spatial_init = logits4_spatial_init.view(imshape[3][0], imshape[3][2], imshape[3][3], -1).permute(0, 3, 1, 2)
        # cam4_spatial_init: [B, total_classes, H/32, W/32]

        # 4.2 CAM
        cam4_phase_init = None
        cam4_amplitude_init = None

        if self.enable_frequency_branch:
            phase_features, amplitude_features = self.freq_extractor(x)
            # phase_features: [B, 12, H, W]
            # amplitude_features: [B, 18, H, W]

            # F4
            phase_feat = F.interpolate(phase_features, size=(imshape[3][2], imshape[3][3]), mode='bilinear')
            amplitude_feat = F.interpolate(amplitude_features, size=(imshape[3][2], imshape[3][3]), mode='bilinear')

            # C4
            phase_proto_projected = self.phase_proto_fc4(self.phase_prototypes.to(x.device).to(x.dtype))
            amplitude_proto_projected = self.amplitude_proto_fc4(self.amplitude_prototypes.to(x.device).to(x.dtype))
            # shape: [total_classes, C4]

            # CAM
            phase_feat_flat = phase_feat.permute(0, 2, 3, 1).reshape(-1, 12)
            phase_feat_flat = phase_feat_flat / (phase_feat_flat.norm(dim=-1, keepdim=True) + 1e-8)
            phase_feat_projected = self.phase_fc4(phase_feat_flat)  # [B*H*W, C4]
            phase_feat_projected = phase_feat_projected / (phase_feat_projected.norm(dim=-1, keepdim=True) + 1e-8)

            logits4_phase_init = scales[3] * phase_feat_projected @ phase_proto_projected.t().float()
            cam4_phase_init = logits4_phase_init.view(imshape[3][0], imshape[3][2], imshape[3][3], -1).permute(0, 3, 1, 2)

            # CAM
            amplitude_feat_flat = amplitude_feat.permute(0, 2, 3, 1).reshape(-1, 18)
            amplitude_feat_flat = amplitude_feat_flat / (amplitude_feat_flat.norm(dim=-1, keepdim=True) + 1e-8)
            amplitude_feat_projected = self.amplitude_fc4(amplitude_feat_flat)  # [B*H*W, C4]
            amplitude_feat_projected = amplitude_feat_projected / (amplitude_feat_projected.norm(dim=-1, keepdim=True) + 1e-8)

            logits4_amplitude_init = scales[3] * amplitude_feat_projected @ amplitude_proto_projected.t().float()
            cam4_amplitude_init = logits4_amplitude_init.view(imshape[3][0], imshape[3][2], imshape[3][3], -1).permute(0, 3, 1, 2)

        # 4.3 SEE-MoE-
        if self.enable_see_moe and self.enable_moe:
            if self.enable_frequency_branch and cam4_phase_init is not None and cam4_amplitude_init is not None:
                guidance_maps = {
                    'spatial': cam4_spatial_init,      # [B, total_classes, H/32, W/32]
                    'phase': cam4_phase_init,          # [B, total_classes, H/32, W/32]
                    'amplitude': cam4_amplitude_init   # [B, total_classes, H/32, W/32]
                }
            else:
                # CAM
                guidance_maps = {
                    'spatial': cam4_spatial_init,
                    'phase': cam4_spatial_init,
                    'amplitude': cam4_spatial_init
                }

            # SEE-MoE
            # 1. APCE
            F4_enhanced, see_moe_aux = self.see_moe(
                F4_raw,
                current_epoch=self.current_epoch,
                apce_quality_scores=self.last_quality_scores,  # APCESEE
                x_low=_x[0],
                guidance_maps=guidance_maps
            )

            # if self.training:
            #     print(f"[SEE-MoE] Entropy: {see_moe_aux['entropy']:.4f}, "
            #           f"Gamma: {see_moe_aux['gamma_mean']:.4f}")
        else:
            F4_enhanced = F4_raw

        # 4.4 CAM
        _x4 = F4_enhanced.permute(0, 2, 3, 1).reshape(-1, F4_enhanced.shape[1])
        _x4 = _x4 / (_x4.norm(dim=-1, keepdim=True) + 1e-8)
        logits4_spatial = scales[3] * _x4 @ projected_l_fea[3].t().float()
        cam4_spatial = logits4_spatial.view(imshape[3][0], imshape[3][2], imshape[3][3], -1).permute(0, 3, 1, 2)
        # cam4_spatial: [B, total_classes, H/32, W/32]

        # 4.5 CAMCAM
        if self.enable_frequency_branch:
            cam4_phase = cam4_phase_init
            cam4_amplitude = cam4_amplitude_init

            alpha = torch.sigmoid(self.alpha_spatial)
            beta = torch.sigmoid(self.beta_phase)
            gamma = torch.sigmoid(self.gamma_amplitude)

            total_weight = alpha + beta + gamma
            alpha = alpha / total_weight
            beta = beta / total_weight
            gamma = gamma / total_weight

            # CAM:  +  +
            cam4_child_init = alpha * cam4_spatial + beta * cam4_phase + gamma * cam4_amplitude

            # if self.training:
            #     print(f"[Frequency Branch] ={alpha.item():.3f}, ={beta.item():.3f}, ={gamma.item():.3f}")
        else:
            # CAM
            cam4_child_init = cam4_spatial
            cam4_phase = None
            cam4_amplitude = None

        # CAM
        cam4_parent_init = merge_to_parent_predictions(cam4_child_init, self.k_list, method='max')
        # cam4_parent_init: [B, num_parent, H/32, W/32]

        # 4.6 APCE
        # 4.6 APCE
        if self.enable_ap_ce:
            parent_pseudo = cam4_parent_init.argmax(dim=1)

            # APCECAM
            ap_ce_loss, apce_aux = self.ap_ce(
                cam=cam4_parent_init,
                features=F4_enhanced,
                pseudo_labels=parent_pseudo,
                cls_labels=cls_labels,
                current_epoch=self.current_epoch,
                update_proto=update_apce_proto and self.training
            )

            # batchSEE-MoE
            self.last_quality_scores = apce_aux["quality_scores"]

            # if self.training:
            #     print(f"[APCE] Quality: {apce_aux['quality_scores'].mean():.4f}, "
            #           f"Temp: {apce_aux['temperature']:.4f}")
        else:
            ap_ce_loss = torch.tensor(0.0, device=x.device)
            apce_aux = {
                'quality_scores': torch.zeros(x.shape[0], self.cls_num_classes).to(x.device),
                'cam_feedback': cam4_parent_init,
                'mask_low': torch.ones(imshape[3][2], imshape[3][3] // 2 + 1).to(x.device),
                'mask_high': torch.zeros(imshape[3][2], imshape[3][3] // 2 + 1).to(x.device)
            }

        quality_child, propagation_weight = self.quality_propagator.propagate_quality(
            quality_parent=apce_aux['quality_scores'],
            cam_parent=cam4_parent_init,
            cam_child=cam4_child_init
        )
        # quality_child: [B, total_classes] -

        # 4.6 CAM
        if self.enable_ap_ce:
            # APCECAM
            cam4_parent_refined = apce_aux['cam_feedback']

            # CAM
            cam4_child_propagated = self.expand_parent_to_subclass_cam(
                cam4_parent_refined,
                cam4_child_init
            )
        else:
            cam4_child_propagated = cam4_child_init

        # 4.7 APCE + SEE
        if self.enable_ap_ce and self.enable_see_moe and self.enable_moe:
            # APCE
            cam_feedback_child = self.expand_parent_to_subclass_cam(
                apce_aux['cam_feedback'],
                cam4_child_init
            )

            cam4_child_final = self.frequency_domain_collaborative_refinement(
                cam_child=cam4_child_propagated,
                apce_feedback=cam_feedback_child,
                mask_low=apce_aux['mask_low'],
                mask_high=apce_aux['mask_high']
            )

            if self.training:
                print(f"[Collaborative Refinement] APCE(Low-freq) + SEE(High-freq)")
        else:
            cam4_child_final = cam4_child_propagated

        # CAM
        cam4_parent_final = merge_to_parent_predictions(cam4_child_final, self.k_list, method='max')

        # logits
        cls4 = self.pooling(cam4_child_final, (1, 1)).view(-1, self.total_classes)

        return {
            # logits
            'cls1': cls1,
            'cls2': cls2,
            'cls3': cls3,
            'cls4': cls4,

            # CAMs
            'cam1': cam1,
            'cam2': cam2,
            'cam3': cam3,
            'cam4': cam4_child_final,
            'cam4_child_final': cam4_child_final,
            'cam4_parent_final': cam4_parent_final,

            # CAM
            'cam4_spatial': cam4_spatial,
            'cam4_phase': cam4_phase if self.enable_frequency_branch else None,
            'cam4_amplitude': cam4_amplitude if self.enable_frequency_branch else None,

            'quality_child': quality_child,
            'quality_parent': apce_aux['quality_scores'],

            'see_moe_aux': see_moe_aux,
            'apce_aux': apce_aux,
            'ap_ce_loss': ap_ce_loss,

            'features': F4_enhanced,
            'l_fea': l_fea           #  device

        }
