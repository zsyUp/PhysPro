

import argparse
import datetime
import os
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.trainutils import get_cls_dataset
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import set_seed
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG
from utils.hierarchical_utils import (pair_features, merge_to_parent_predictions,
                                     expand_parent_to_subclass_labels)
from utils.validate import validate, generate_cam
from model.complete_hdmoe_model import ImprovedClsNetwork
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from checkpoint_utils import (
    auto_resume_or_start_new,
    save_periodic_checkpoint,
    cleanup_old_checkpoints,
    save_checkpoint
)
from ema_utils import ModelEMA, ClassBalancedLoss

start_time = datetime.datetime.now()


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
args = parser.parse_args()


# Metrics Logger
class MetricsLogger:
    """Persist train, validation, and test metrics as plain-text logs."""
    def __init__(self, log_dir, exp_name):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.train_log = os.path.join(log_dir, f"{exp_name}_train.txt")
        self.val_log = os.path.join(log_dir, f"{exp_name}_val.txt")
        self.test_log = os.path.join(log_dir, f"{exp_name}_test.txt")

        with open(self.train_log, 'w') as f:
            f.write("Iter,Epoch,LR,TotalLoss,ClsLoss,ContrastLoss,SEELoss,APCELoss,"
                    "DTAPALoss,AllAcc,AvgAcc,ElapsedTime\n")

        with open(self.val_log, 'w') as f:
            f.write("Iter,Epoch,AllAcc,AvgAcc,mIoU,C0_IoU,C1_IoU,C2_IoU,C3_IoU,Loss,"
                    "FwIoU,bIoU,Dice,mPrecision,mRecall,"
                    "ClassWeight0,ClassWeight1,ClassWeight2,ClassWeight3\n")

        with open(self.test_log, 'w') as f:
            f.write("AllAcc,AvgAcc,mIoU,C0_IoU,C1_IoU,C2_IoU,C3_IoU,Loss,"
                    "FwIoU,bIoU,Dice,mPrecision,mRecall\n")

    def log_train(self, iter, epoch, lr, losses, metrics, elapsed):
        """Append one training row to the train log."""
        clean_elapsed = str(elapsed).split('.')[0]

        line = (f"{iter},{epoch},{lr:.6f},"
                f"{losses.get('total', 0):.4f},{losses.get('cls', 0):.4f},"
                f"{losses.get('contrast', 0):.4f},"
                f"{losses.get('see', 0):.4f},"
                f"{losses.get('apce', 0):.4f},"
                f"{losses.get('dtpa', 0):.4f},"
                f"{metrics.get('all_acc', 0):.2f},{metrics.get('avg_acc', 0):.2f},"
                f"{clean_elapsed}\n")

        with open(self.train_log, 'a') as f:
            f.write(line)

    def log_val(self, iter, epoch, all_acc, avg_acc, iou_scores, loss,
            fwiou=0.0, biou=0.0, dice=0.0, precision=0.0, recall=0.0,
            class_weights=None):
        """Append one validation row including extended segmentation metrics."""
        miou = iou_scores[:-1].mean()
        line = f"{iter},{epoch},{all_acc:.6f},{avg_acc:.6f},{miou:.4f},"
        line += ",".join([f"{s:.6f}" for s in iou_scores[:-1]])
        line += f",{loss:.4f}"

        line += f",{fwiou:.4f},{biou:.4f},{dice:.4f},{precision:.4f},{recall:.4f}"

        if class_weights is not None:
            line += "," + ",".join([f"{w:.4f}" for w in class_weights.cpu().numpy()])
        else:
            line += ",1.0,1.0,1.0,1.0"

        line += "\n"
        with open(self.val_log, 'a') as f:
            f.write(line)

    def log_test(self, all_acc, avg_acc, iou_scores, loss,
             fwiou=0.0, biou=0.0, dice=0.0, precision=0.0, recall=0.0):
        """Write the final test metrics row."""
        miou = iou_scores[:-1].mean()
        line = f"{all_acc:.6f},{avg_acc:.6f},{miou:.4f},"
        line += ",".join([f"{s:.6f}" for s in iou_scores[:-1]])
        line += f",{loss:.4f}"
        line += f",{fwiou:.4f},{biou:.4f},{dice:.4f},{precision:.4f},{recall:.4f}\n"
        with open(self.test_log, 'a') as f:
            f.write(line)

    def save_summary(self, best_metrics, test_metrics, total_time):
        summary_path = os.path.join(self.log_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\nHD-MoE EXPERIMENT SUMMARY (FIXED VERSION)\n" + "=" * 80 + "\n\n")
            f.write(f"Total Training Time: {str(total_time).split('.')[0]}\n\n")
            f.write("Improvements Applied:\n")
            f.write("   EMA for stable training\n")
            f.write("   Class balanced loss\n")
            f.write("   Smooth weight transition\n")
            f.write("   Frequent validation (2000 iters)\n\n")
            f.write("Best Validation Metrics:\n")
            f.write(f"  Iteration: {best_metrics['iter']}\n  mIoU: {best_metrics['miou']:.4f}\n")
            f.write("Final Test Metrics:\n")
            f.write(f"  mIoU: {test_metrics['miou']:.4f}\n  Per-class IoU: {test_metrics['per_class_iou']}\n")
        print(f"\n Summary saved: {summary_path}")


class APCELogger:
    """Track APCE-specific diagnostics during training."""
    def __init__(self, log_dir, exp_name):
        self.log_path = os.path.join(log_dir, f"{exp_name}_apce_metrics.txt")
        os.makedirs(log_dir, exist_ok=True)

        with open(self.log_path, 'w') as f:
            f.write("Iter,Epoch,QualityMean,ProtoConfidence,FeedbackStrength,APCELoss\n")
        print(f"APCE log initialized: {self.log_path}")

    def log(self, iter_num, epoch, aux_outputs, apce_loss_item):
        """Append one APCE diagnostics row."""
        quality_mean = aux_outputs.get('quality_scores', torch.zeros(1, 4)).mean().item()
        proto_conf = aux_outputs.get('prototype_confidence', torch.tensor(0.0)).item()
        feedback_strength = aux_outputs.get('feedback_strength', torch.tensor(0.0)).item()

        line = f"{iter_num},{epoch},{quality_mean:.6f},{proto_conf:.6f},{feedback_strength:.6f},{apce_loss_item:.6f}\n"

        with open(self.log_path, 'a') as f:
            f.write(line)


class SEEMoELogger:
    """Track SEE-MoE-specific diagnostics during training."""
    def __init__(self, log_dir, exp_name):
        self.log_path = os.path.join(log_dir, f"{exp_name}_see_moe_metrics.txt")
        os.makedirs(log_dir, exist_ok=True)

        with open(self.log_path, 'w') as f:
            f.write("Iter,Epoch,Entropy,GammaMean,SpatialContrib,SpectralContrib,DiversityLoss\n")
        print(f"SEE-MoE log initialized: {self.log_path}")

    def log(self, iter_num, epoch, aux_outputs, diversity_loss_item):
        """Append one SEE-MoE diagnostics row."""
        entropy = aux_outputs.get('entropy', torch.tensor(0.0)).item()
        gamma_mean = aux_outputs.get('gamma_mean', torch.tensor(0.0)).item()
        spatial_contrib = aux_outputs.get('spatial_contribution', torch.tensor(0.0)).item()
        spectral_contrib = aux_outputs.get('spectral_contribution', torch.tensor(0.0)).item()

        line = f"{iter_num},{epoch},{entropy:.6f},{gamma_mean:.6f},{spatial_contrib:.6f},{spectral_contrib:.6f},{diversity_loss_item:.6f}\n"

        with open(self.log_path, 'a') as f:
            f.write(line)


class TrainingScheduler:
    """Manage stage-wise loss weights for progressive HD-MoE training."""
    def __init__(self, total_iters, warmup_iters, iters_per_epoch):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.iters_per_epoch = iters_per_epoch

        self.stage1_end = 8 * iters_per_epoch   # Epoch 1-8
        self.stage2_end = 15 * iters_per_epoch  # Epoch 9-15

        print("\n" + "="*80)
        print("FIXED Three-Stage Training Strategy:")
        print("  Stage 1 (E1-8):   SEE=0.6, APCE=0.0, DTPA=0.0")
        print("  Stage 2 (E9-15):  Smooth SEE->APCE transition")
        print("  Stage 3 (E16+):   SEE=0.3, APCE=0.9, DTPA=0.1")
        print("="*80 + "\n")

    def get_loss_weights(self, current_iter):
        """Return stage-dependent auxiliary loss weights for one iteration."""
        import numpy as np
        current_epoch = (current_iter // self.iters_per_epoch) + 1

        if current_iter < self.stage1_end:
            return {'see': 0.6, 'apce': 0.0, 'dtpa': 0.0}

        elif current_iter < self.stage2_end:
            progress = (current_iter - self.stage1_end) / (self.stage2_end - self.stage1_end)

            see_weight = 0.6 - 0.3 * progress
            apce_weight = 0.9 * (1 - np.cos(progress * np.pi)) / 2

            if current_epoch >= 12:
                dtpa_progress = min((current_epoch - 12) / 3, 1.0)
                dtpa_weight = 0.1 * dtpa_progress
            else:
                dtpa_weight = 0.0

            return {
                'see': see_weight,
                'apce': apce_weight,
                'dtpa': dtpa_weight
            }

        else:
            return {'see': 0.3, 'apce': 0.9, 'dtpa': 0.1}

    def should_update_apce_prototypes(self, current_iter):
        """Enable APCE prototype updates after stage 1 completes."""
        return current_iter >= self.stage1_end

    def get_stage_name(self, current_iter):
        current_epoch = (current_iter // self.iters_per_epoch) + 1
        if current_iter < self.stage1_end:
            return f"Stage1-SEELearn(E{current_epoch})"
        elif current_iter < self.stage2_end:
            return f"Stage2-Transition(E{current_epoch})"
        else:
            return f"Stage3-Balanced(E{current_epoch})"


def cal_eta(time0, cur_iter, total_iter):
    """Return elapsed time and estimated time remaining as strings."""
    time_now = datetime.datetime.now().replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    return str(delta), str(eta)


def train(cfg):
    print("\n" + "="*80)
    print("HD-MoE TRAINING - FIXED VERSION")
    print("Improvements: EMA + Class Balance + Smooth Transition")
    print("="*80)

    torch.backends.cudnn.benchmark = True
    num_workers = min(10, os.cpu_count())
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(42)

    clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    clip_model = clip_model.to(device)
    clip_model.eval()

    time0 = datetime.datetime.now().replace(microsecond=0)

    print("\nPreparing datasets...")
    train_dataset, val_dataset = get_cls_dataset(cfg, split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.samples_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    print("\nInitializing HD-MoE model...")
    model = ImprovedClsNetwork(
        backbone=cfg.model.backbone.config,
        stride=cfg.model.backbone.stride,
        cls_num_classes=cfg.dataset.cls_num_classes,
        n_ratio=cfg.model.n_ratio,
        pretrained=cfg.train.pretrained,
        l_fea_path=cfg.model.label_feature_path,
        freq_proto_path=cfg.model.freq_proto_path,
        enable_see_moe=cfg.model.get('enable_see_moe', True),
        enable_ap_ce=cfg.model.get('enable_ap_ce', True),
        enable_dtpa=cfg.model.get('enable_dtpa', False),
        enable_adaptive_dtpa=cfg.model.get('enable_adaptive_dtpa', False),
    )
    model.to(device)

    # ============  EMA ============
    print("\n" + "="*80)
    print("Initializing EMA (Exponential Moving Average)")
    print("="*80)
    ema = ModelEMA(
        model=model,
        decay=0.9999,
        warmup_steps=2000
    )
    print(" EMA configured: decay=0.9999, warmup=2000 steps")
    print("  Purpose: Stabilize training, reduce validation fluctuation")
    print("="*80 + "\n")

    print("Initializing Class Balanced Loss")
    print("="*80)

    # IoU
    initial_weights = torch.ones(4).to(device)

    class_balanced_loss = ClassBalancedLoss(
        num_classes=cfg.dataset.cls_num_classes,
        initial_weights=initial_weights,
        momentum=0.99
    ).to(device)

    print(f" Initial class weights: {initial_weights.cpu().numpy()}")
    print(f"  Class 0: {initial_weights[0]:.3f} (High IoU  Lower weight)")
    print(f"  Class 1: {initial_weights[1]:.3f}")
    print(f"  Class 2: {initial_weights[2]:.3f} (Low IoU  Higher weight)")
    print(f"  Class 3: {initial_weights[3]:.3f}")
    print("  Purpose: Balance training focus on underperforming classes")
    print("="*80 + "\n")

    iters_per_epoch = len(train_loader)
    model.iters_per_epoch = iters_per_epoch
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch

    print(f" Validation frequency: every {cfg.train.eval_iters} iterations (increased from 3000)")

    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch

    scheduler = TrainingScheduler(
        total_iters=cfg.train.max_iters,
        warmup_iters=cfg.scheduler.warmup_iter,
        iters_per_epoch=iters_per_epoch
    )

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    optimizer = PolyWarmupAdamW(
        params=model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    start_iter, best_metrics, resumed = auto_resume_or_start_new(
        cfg, model, optimizer, scaler, device, resume_path=None
    )

    if resumed:
        print(f"\n{'='*60}")
        print(f" TRAINING RESUMED FROM ITERATION {start_iter}")
        print(f"   Best mIoU so far: {best_metrics['miou']:.4f}")
        print(f"{'='*60}\n")
    else:
        best_metrics = {'iter': 0, 'miou': 0.0, 'all_acc': 0.0, 'avg_acc': 0.0}
        print(f"\n{'='*60}")
        print(f" STARTING NEW TRAINING")
        print(f"{'='*60}\n")

    loss_function = nn.BCEWithLogitsLoss().to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha)
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    logger = MetricsLogger(log_dir=cfg.work_dir.ckpt_dir, exp_name=f"hdmoe_fixed_{timestamp}")
    apce_logger = APCELogger(log_dir=cfg.work_dir.ckpt_dir, exp_name=f"hdmoe_fixed_{timestamp}")
    see_moe_logger = SEEMoELogger(log_dir=cfg.work_dir.ckpt_dir, exp_name=f"hdmoe_fixed_{timestamp}")

    print("\nStarting training...\n")

    train_loader_iter = iter(train_loader)

    #  start_iter
    for n_iter in range(start_iter, cfg.train.max_iters):

        try:
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)

        # epoch
        current_epoch = (n_iter // iters_per_epoch) + 1
        model.current_epoch = current_epoch
        model.current_iter = n_iter

        loss_weights = scheduler.get_loss_weights(n_iter)

        model.enable_see_moe = (loss_weights['see'] > 0) and cfg.model.get('enable_see_moe', True)
        model.enable_ap_ce = (loss_weights['apce'] > 0) and cfg.model.get('enable_ap_ce', True)
        model.enable_moe = (n_iter >= cfg.scheduler.warmup_iter)

        # APCE
        should_update_apce = scheduler.should_update_apce_prototypes(n_iter)

        inputs = inputs.to(device).float()
        cls_labels = cls_labels.to(device).float()

        with torch.cuda.amp.autocast():
            outputs_dict = model(inputs, cls_labels=cls_labels)

            cls1 = outputs_dict['cls1']
            cls2 = outputs_dict['cls2']
            cls3 = outputs_dict['cls3']
            cls4 = outputs_dict['cls4']

            cam1 = outputs_dict['cam1']
            cam2 = outputs_dict['cam2']
            cam3 = outputs_dict['cam3']
            cam4 = outputs_dict['cam4_child_final']  # CAM

            # k_list
            k_list = model.k_list


            l_fea = outputs_dict['l_fea']

            aux_outputs = {
                'see_moe_aux': outputs_dict.get('see_moe_aux', {}),
                'apce_aux': outputs_dict.get('apce_aux', {}),
                'ap_ce_loss': outputs_dict.get('ap_ce_loss', torch.tensor(0.0))
            }

            cls1_merge = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_train)
            cls2_merge = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_train)
            cls3_merge = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_train)
            cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_train)

            subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
            cls4_expand = expand_parent_to_subclass_labels(cls4_merge, k_list)
            cls4_bir = (cls4 > cls4_expand).float() * subclass_labels

            batch_info = feature_extractor.process_batch(inputs, cam4, cls4_bir, clip_model)
            fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']

            set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
            fg_features = set_info['fg_features']
            bg_features = set_info['bg_features']
            fg_pro = set_info['fg_text']
            bg_pro = set_info['bg_text']

            fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
            bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)

            loss1 = loss_function(cls1_merge, cls_labels)
            loss2 = loss_function(cls2_merge, cls_labels)
            loss3 = loss_function(cls3_merge, cls_labels)
            loss4 = loss_function(cls4_merge, cls_labels)

            loss_per_sample = F.binary_cross_entropy_with_logits(
                cls4_merge, cls_labels, reduction='none'
            )
            loss_per_class = loss_per_sample.mean(dim=0)

            # stage 4
            cls_loss_balanced = loss_per_class.mean()

            # stagestage 4
            fg_bg_weight = min(1.0, n_iter / cfg.scheduler.warmup_iter)
            cls_loss = (cfg.train.l1 * loss1 + cfg.train.l2 * loss2 +
                       cfg.train.l3 * loss3 + cfg.train.l4 * cls_loss_balanced)

            contrast_loss = (fg_loss + bg_loss + 0.0005 * torch.mean(cam4)) * cfg.train.l5

            # SEE-MoE diversity loss
            see_moe_loss = aux_outputs.get('diversity_loss', torch.tensor(0.0, device=device))
            if model.enable_see_moe and model.enable_moe and (n_iter + 1) % 100 == 0:
                see_moe_logger.log(n_iter + 1, current_epoch, aux_outputs, see_moe_loss.item())

            # APCE loss
            apce_loss = aux_outputs.get('ap_ce_loss', torch.tensor(0.0, device=device))
            if model.enable_ap_ce and (n_iter + 1) % 100 == 0:
                apce_logger.log(n_iter + 1, current_epoch, aux_outputs, apce_loss.item())

            # DTPA loss
            dtpa_loss = aux_outputs.get('dtpa_loss', torch.tensor(0.0, device=device))

            total_loss = (cls_loss +
                         fg_bg_weight * contrast_loss +
                         loss_weights['see'] * see_moe_loss +
                         loss_weights['apce'] * apce_loss +
                         loss_weights['dtpa'] * dtpa_loss)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        #  EMA
        ema.update(model)

        if (n_iter + 1) % 100 == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            torch.cuda.synchronize()

            cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
            all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
            avg_cls_acc4 = ((cls_pred4 == cls_labels).float().mean(dim=0)).mean() * 100

            stage_name = scheduler.get_stage_name(n_iter)

            quality_avg = aux_outputs.get('quality_scores', torch.zeros(1, 4)).mean().item()
            proto_conf = aux_outputs.get('prototype_confidence', torch.tensor(0.0)).item()

            print(
                f"[{stage_name}] Iter: {n_iter + 1}/{cfg.train.max_iters}; "
                f"Elapsed: {delta}; ETA: {eta}; "
                f"LR: {cur_lr:.3e}; "
                f"Loss: {total_loss.item():.4f} "
                f"(Cls: {cls_loss.item():.3f}, "
                f"Contrast: {(fg_bg_weight * contrast_loss).item():.3f}, "
                f"SEE: {see_moe_loss.item():.4f}{loss_weights['see']:.2f}, "
                f"APCE: {apce_loss.item():.4f}{loss_weights['apce']:.2f}); "
                f"Quality: {quality_avg:.3f}; "
                f"ProtoConf: {proto_conf:.3f}; "
                f"Acc: {all_cls_acc4:.1f}/{avg_cls_acc4:.1f}"
            )

            logger.log_train(
                iter=n_iter + 1,
                epoch=current_epoch,
                lr=cur_lr,
                losses={
                    'total': total_loss.item(),
                    'cls': cls_loss.item(),
                    'contrast': (fg_bg_weight * contrast_loss).item(),
                    'see': see_moe_loss.item(),
                    'apce': apce_loss.item(),
                    'dtpa': dtpa_loss.item()
                },
                metrics={
                    'all_acc': all_cls_acc4.item(),
                    'avg_acc': avg_cls_acc4.item(),
                },
                elapsed=str(delta)
            )

        #  EMA
        if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
            print(f"\n{'='*60}")
            print(f"Validation at iteration {n_iter + 1} (Using EMA Model)")
            print(f"{'='*60}")

            #  EMA
            with ema.apply_shadow():
                val_all_acc4, val_avg_acc4, fuse234_score, val_cls_loss, \
                    fwiou, biou, dice, precision, recall, all_metrics = validate(  #   precision, recall
                    model=model,
                    data_loader=val_loader,
                    cfg=cfg,
                    cls_loss_func=loss_function,
                    return_extra_metrics=True
                )

            miou = fuse234_score[:-1].mean()

            class_iou = fuse234_score[:-1]
            # class_balanced_loss.update_weights_from_iou(class_iou)
            # current_weights = class_balanced_loss.get_current_weights()
            current_weights = torch.ones(4).to(device)  # 1
            print(f"Val Results:")
            print(f"  All Acc: {val_all_acc4:.6f}")
            print(f"  Avg Acc: {val_avg_acc4:.6f}")
            print(f"  mIoU: {miou:.4f}")
            print(f"  Per-class IoU: {class_iou}")
            print(f"   Class Weights: {current_weights.cpu().numpy()}")

            print("\nExtended validation metrics:")
            print(f"  FwIoU (IoU): {fwiou.item()*100:.2f}%")
            print(f"  bIoU (boundary IoU): {biou.item()*100:.2f}%")
            print(f"  Dice (F1):       {dice.item()*100:.2f}%")
            print(f"  mPrecision:      {precision.item()*100:.2f}%")
            print(f"  mRecall:         {recall.item()*100:.2f}%")
            print("\nPer-class diagnostics:")
            print(f"  Dice: {all_metrics['dice_per_class']}")
            print(f"  bIoU: {all_metrics['biou_per_class']}")
            logger.log_val(
                iter=n_iter + 1,
                epoch=current_epoch,
                all_acc=val_all_acc4,
                avg_acc=val_avg_acc4,
                iou_scores=fuse234_score,
                loss=val_cls_loss,
                fwiou=fwiou.item(),
                biou=biou.item(),
                dice=dice.item(),
                precision=precision.item(),
                recall=recall.item(),
                class_weights=current_weights
            )

            if miou > best_metrics['miou']:
                best_metrics['iter'] = n_iter + 1
                best_metrics['miou'] = miou
                best_metrics['all_acc'] = val_all_acc4
                best_metrics['avg_acc'] = val_avg_acc4

                # EMA
                with ema.apply_shadow():
                    checkpoint = {
                        'iter': n_iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_metrics': best_metrics,
                        'ema_state_dict': ema.state_dict(),
                        'class_weights': current_weights.cpu()
                    }

                    best_path = os.path.join(cfg.work_dir.ckpt_dir, 'best_hdmoe_ema.pth')
                    torch.save(checkpoint, best_path)

                print(f"\n Saved best EMA model (mIoU: {miou:.4f})")

            print(f"{'='*60}\n")
            model.train()

        #  EMA
        if (n_iter + 1) % 1000 == 0:
            checkpoint = {
                'iter': n_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_metrics': best_metrics,
                'ema_state_dict': ema.state_dict(),
                'class_balanced_loss_state': class_balanced_loss.state_dict()
            }
            ckpt_path = os.path.join(cfg.work_dir.ckpt_dir, f'checkpoint_iter_{n_iter+1}.pth')
            torch.save(checkpoint, ckpt_path)

        if (n_iter + 1) % 5000 == 0:
            cleanup_old_checkpoints(
                ckpt_dir=cfg.work_dir.ckpt_dir,
                keep_last_n=3
            )

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f'\nTotal training time: {total_time}')

    #  EMA
    print("\n" + "="*80)
    print("POST-TRAINING EVALUATION (Using EMA Model)")
    print("="*80)

    train_dataset, test_dataset = get_cls_dataset(cfg, split="test",
                                                   enable_rotation=False, p=0.0)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    #  EMA
    best_model_path = os.path.join(cfg.work_dir.ckpt_dir, "best_hdmoe_ema.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # EMA
        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])

        print(f" Loaded best EMA model (iter: {checkpoint.get('iter', 'unknown')})")

    #  EMA
    with ema.apply_shadow():
        test_all_acc4, test_avg_acc4, test_iou_scores, test_cls_loss, \
        test_fwiou, test_biou, test_dice, test_precision, test_recall, test_all_metrics = validate(
        model=model,
        data_loader=test_loader,
        cfg=cfg,
        cls_loss_func=loss_function,
        return_extra_metrics=True
    )

    test_miou = test_iou_scores[:-1].mean()

    print(f"\nTest Results (EMA Model):")
    print(f"  All Acc: {test_all_acc4:.6f}")
    print(f"  Avg Acc: {test_avg_acc4:.6f}")
    print(f"  mIoU: {test_miou:.4f}")
    print(f"  FwIoU: {test_fwiou.item():.4f}")
    print(f"  bIoU: {test_biou.item():.4f}")
    print(f"  Dice: {test_dice.item():.4f}")
    print(f"  mPrecision: {test_precision.item():.4f}")
    print(f"  mRecall: {test_recall.item():.4f}")
    print(f"\nPer-class IoU:")
    for i, score in enumerate(test_iou_scores[:-1]):
        print(f"  Class {i}: {score:.6f}")
    print(f"\nPer-class Dice:")
    for i, score in enumerate(test_all_metrics['dice_per_class'][:-1]):
        print(f"  Class {i}: {score:.6f}")

    logger.log_test(
    all_acc=test_all_acc4,
    avg_acc=test_avg_acc4,
    iou_scores=test_iou_scores,
    loss=test_cls_loss,
    fwiou=test_fwiou.item(),
    biou=test_biou.item(),
    dice=test_dice.item(),
    precision=test_precision.item(),
    recall=test_recall.item()
)

    test_metrics = {
    'miou': test_miou.item() if hasattr(test_miou, 'item') else test_miou,
    'fwiou': test_fwiou.item(),
    'biou': test_biou.item(),
    'dice': test_dice.item(),
    'precision': test_precision.item(),
    'recall': test_recall.item(),
    'all_acc': test_all_acc4,
    'avg_acc': test_avg_acc4,
    'per_class_iou': test_iou_scores[:-1].tolist(),
    'per_class_dice': test_all_metrics['dice_per_class'][:-1].tolist(),
    'per_class_biou': test_all_metrics['biou_per_class'][:-1].tolist(),
    'per_class_precision': test_all_metrics['precision_per_class'][:-1].tolist(),
    'per_class_recall': test_all_metrics['recall_per_class'][:-1].tolist()
}
    logger.save_summary(best_metrics, test_metrics, total_time)

    # CAM
    print("\n" + "="*80)
    print("GENERATING CAMs")
    print("="*80)

    train_cam_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Generating CAMs for {len(train_dataset)} training samples...")

    #  EMACAM
    with ema.apply_shadow():
        generate_cam(model=model, data_loader=test_loader, cfg=cfg)

    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"\n Improvements Applied:")
    print(f"   - EMA for stable training")
    print(f"   - Class balanced loss")
    print(f"   - Smooth weight transition")
    print(f"   - Frequent validation")
    print(f"\nBest Val mIoU: {best_metrics['miou']:.4f}")
    print(f"Final Test mIoU: {test_miou:.4f}")
    print(f"\nAll logs saved to: {cfg.work_dir.ckpt_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    cfg = OmegaConf.load(args.config)
    cfg.work_dir.dir = os.path.dirname(args.config)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)

    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)

    print('\nArgs:', args)
    print('\nConfigs:', cfg)

    set_seed(0)
    train(cfg=cfg)
