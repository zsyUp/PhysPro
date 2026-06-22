"""Validation and CAM generation utilities for HD-MoE experiments."""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import ttach as tta
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from .evaluate import ConfusionMatrixAllClass
from .hierarchical_utils import merge_to_parent_predictions, merge_subclass_cams_to_parent
from .pyutils import AverageMeter


def get_seg_label(cams, inputs, label):
    """Generate segmentation labels from CAMs"""
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)

        # Normalize CAMs to [0,1]
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams).float()

        # Resize to input dimensions
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

    return cams


def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None, return_extra_metrics=False):
    """Run validation with test-time augmentation and extended metrics.

    Args:
        return_extra_metrics: If True, return the full metrics dictionary.
    """
    model.eval()
    avg_meter = AverageMeter()

    fuse234_matrix = ConfusionMatrixAllClass(
        num_classes=cfg.dataset.cls_num_classes + 1,
        boundary_width=3,
        compute_boundary=True
    )

    # Test-time augmentation setup
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            outputs_dict = model(inputs, cls_labels=cls_label)

            cls1 = outputs_dict['cls1']
            cls2 = outputs_dict['cls2']
            cls3 = outputs_dict['cls3']
            cls4 = outputs_dict['cls4']

            cam1 = outputs_dict['cam1']
            cam2 = outputs_dict['cam2']
            cam3 = outputs_dict['cam3']
            cam4 = outputs_dict['cam4_child_final']

            k_list = model.k_list

            aux_outputs = {
                'see_moe_aux': outputs_dict.get('see_moe_aux', {}),
                'apce_aux': outputs_dict.get('apce_aux', {}),
                'ap_ce_loss': outputs_dict.get('ap_ce_loss', torch.tensor(0.0))
            }

            cls1 = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test)
            cls2 = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_test)
            cls3 = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_test)
            cls4 = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)

            cls_loss1 = cls_loss_func(cls1, cls_label)
            cls_loss2 = cls_loss_func(cls2, cls_label)
            cls_loss3 = cls_loss_func(cls3, cls_label)
            cls_loss4 = cls_loss_func(cls4, cls_label)
            cls_loss = cfg.train.l1 * cls_loss1 + cfg.train.l2 * cls_loss2 + cfg.train.l3 * cls_loss3 + cfg.train.l4 * cls_loss4

            cls4 = (torch.sigmoid(cls4) > 0.5).float()
            all_cls_acc4 = (cls4 == cls_label).all(dim=1).float().sum() / cls4.shape[0] * 100
            avg_cls_acc4 = ((cls4 == cls_label).sum(dim=0) / cls4.shape[0]).mean() * 100
            avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})

            # Generate evaluation CAMs with TTA
            cams1 = []
            cams2 = []
            cams3 = []
            cams4 = []

            for tta_trans in tta_transform:
                augmented_tensor = tta_trans.augment_image(inputs)

                #  v2TTA
                outputs_dict = model(augmented_tensor, cls_labels=cls_label)

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

                aux_outputs = {
                    'see_moe_aux': outputs_dict.get('see_moe_aux', {}),
                    'apce_aux': outputs_dict.get('apce_aux', {}),
                    'ap_ce_loss': outputs_dict.get('ap_ce_loss', torch.tensor(0.0))
                }

                cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
                cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
                cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
                cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

                cls1 = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test)
                cls2 = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_test)
                cls3 = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_test)
                cls4 = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)

                cam1 = get_seg_label(cam1, augmented_tensor, cls_label).cuda()
                cam1 = tta_trans.deaugment_mask(cam1).unsqueeze(dim=0)
                cams1.append(cam1)
                cam2 = get_seg_label(cam2, augmented_tensor, cls_label).cuda()
                cam2 = tta_trans.deaugment_mask(cam2).unsqueeze(dim=0)
                cams2.append(cam2)
                cam3 = get_seg_label(cam3, augmented_tensor, cls_label).cuda()
                cam3 = tta_trans.deaugment_mask(cam3).unsqueeze(dim=0)
                cams3.append(cam3)
                cam4 = get_seg_label(cam4, augmented_tensor, cls_label).cuda()
                cam4 = tta_trans.deaugment_mask(cam4).unsqueeze(dim=0)
                cams4.append(cam4)

            cams1 = torch.cat(cams1, dim=0).mean(dim=0)
            cams2 = torch.cat(cams2, dim=0).mean(dim=0)
            cams3 = torch.cat(cams3, dim=0).mean(dim=0)
            cams4 = torch.cat(cams4, dim=0).mean(dim=0)

            # Fuse multi-scale predictions
            fuse234 = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4
            fuse_label234 = torch.argmax(fuse234, dim=1).cuda()

            fuse234_matrix.update(labels.detach().clone(), fuse_label234.clone())

    all_cls_acc4, avg_cls_acc4, cls_loss = avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4"), avg_meter.pop("cls_loss")

    all_metrics = fuse234_matrix.compute_all_metrics()

    #  fuse234_score IoUtensormIoU
    #  fuse234_score[:-1].mean() mIoU
    fuse234_score = all_metrics['iou_per_class']  # IoUtensor

    miou = all_metrics['miou']          # mIoU
    fwiou = all_metrics['fwiou']        # IoU
    biou = all_metrics['biou']          # IoU
    dice = all_metrics['dice']          # Dice
    precision = all_metrics['precision']  # mPrecision
    recall = all_metrics['recall']        # mRecall

    model.train()

    if return_extra_metrics:
        # 10fuse234_scoreIoU tensor
        return all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss, fwiou, biou, dice, precision, recall, all_metrics
    else:
        print("\n" + "="*60)
        print(" ")
        print("="*60)
        print(f"  mIoU ():  {miou.item()*100:.2f}%")
        print(f"  FwIoU (): {fwiou.item()*100:.2f}%")
        print(f"  bIoU (IoU):   {biou.item()*100:.2f}%")
        print(f"  Dice (F1):    {dice.item()*100:.2f}%")
        print(f"  mPrecision:       {precision.item()*100:.2f}%")
        print(f"  mRecall:          {recall.item()*100:.2f}%")
        print(f"  IoU: {[f'{v*100:.1f}%' for v in fuse234_score.tolist()]}")
        print("="*60)
        return all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss


def validate_simple(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """

    : all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss
    """
    results = validate(model, data_loader, cfg, cls_loss_func)
    return results[0], results[1], results[2], results[3]


def generate_cam(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """
     v2
    """
    model.eval()

    with torch.no_grad():
        for data in tqdm(data_loader,
                 total=len(data_loader),
                 ncols=120,
                 ascii=True,
                 dynamic_ncols=False,
                 leave=False):
            name, inputs, cls_label, labels = data

            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()


            outputs_dict = model(inputs, cls_labels=cls_label)

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

            aux_outputs = {
                'see_moe_aux': outputs_dict.get('see_moe_aux', {}),
                'apce_aux': outputs_dict.get('apce_aux', {}),
                'ap_ce_loss': outputs_dict.get('ap_ce_loss', torch.tensor(0.0))
            }

            cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
            cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

            cam1 = get_seg_label(cam1, inputs, cls_label).cuda()
            cam2 = get_seg_label(cam2, inputs, cls_label).cuda()
            cam3 = get_seg_label(cam3, inputs, cls_label).cuda()
            cam4 = get_seg_label(cam4, inputs, cls_label).cuda()

            fuse234 = 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
            output_fuse234 = torch.argmax(fuse234, dim=1).long()

            PALETTE = [
             [255, 0, 0],
             [0, 255, 0],
             [0, 0, 255],
             [153, 0, 255],
             [255, 255, 255],
             [0, 0, 0],]

            for i in range(len(output_fuse234)):
                pred_mask = Image.fromarray(output_fuse234[i].cpu().clone().squeeze().numpy().astype(np.uint8)).convert('P')
                flat_palette = [val for sublist in PALETTE for val in sublist]
                pred_mask.putpalette(flat_palette)
                pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i] + ".png"))

    model.train()
    return


def print_metrics_summary(all_metrics, logger=None):
    """


    Args:
        all_metrics: compute_all_metrics()
        logger: logger
    """
    def format_percent(val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        return f"{val * 100:.2f}%"

    def format_list(tensor):
        return [f'{i:.1f}' for i in (tensor * 100).tolist()]

    summary = f"""




   mIoU ():     {format_percent(all_metrics['miou']):>10}
   FwIoU ():    {format_percent(all_metrics['fwiou']):>10}
   bIoU (IoU):      {format_percent(all_metrics['biou']):>10}
   Dice (F1):       {format_percent(all_metrics['dice']):>10}
   mPrecision:          {format_percent(all_metrics['precision']):>10}
   mRecall:             {format_percent(all_metrics['recall']):>10}


   :          {format_percent(all_metrics['acc_global']):>10}
   IoU: {str(format_list(all_metrics['iou_per_class'])):>45}
   Dice: {str(format_list(all_metrics['dice_per_class'])):>44}
   bIoU: {str(format_list(all_metrics['biou_per_class'])):>44}
   Prec: {str(format_list(all_metrics['precision_per_class'])):>44}
   Rec:  {str(format_list(all_metrics['recall_per_class'])):>44}

"""

    if logger is not None:
        logger.info(summary)
    else:
        print(summary)

    return summary
