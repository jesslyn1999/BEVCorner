# Copyright (c) Megvii Inc. All rights reserved.
import os
from functools import partial
from collections import defaultdict
import logging

import mmcv
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from bevdepth.evaluators.det_evaluators import DetNuscEvaluator
from bevdepth.models.base_bev_depth import BaseBEVDepth
from bevdepth.utils.torch_dist import all_gather_object, get_rank, synchronize
from bevdepth.utils.visualization import visualize_from_image_arr, show_image_with_boxes_in_mono

from .base_exp_ext import *

class BEVDepthLightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                    gpus: int = 1,
                    data_root=data_root,
                    eval_interval=1,
                    batch_size_per_device=8,
                    class_names=CLASSES,
                    backbone_conf=backbone_conf,
                    monoflex_backbone_conf=monoflex_backbone_conf,
                    head_conf=head_conf,
                    detector_predictor_conf=detector_predictor_conf,
                    detector_loss_conf=detector_loss_conf,
                    ida_aug_conf=ida_aug_conf,
                    bda_aug_conf=bda_aug_conf,
                    filter_params=filter_annos,
                    down_ratio=down_ratio,
                    orientation_method=orientation,
                    multibin_size=orientation_bin_size,
                    default_root_dir='./outputs/',
                    heatmap_ratio=heatmap_ratio,
                    run_mode='bevcorner',
                    **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.basic_lr_per_img = 2e-4 / 64
        self.class_names = class_names
        
        self.backbone_conf = backbone_conf
        self.monoflex_backbone_conf = monoflex_backbone_conf
        self.head_conf = head_conf

        self.detector_predictor_conf = detector_predictor_conf
        self.detector_loss_conf = detector_loss_conf
        self.detector_post_processor_conf = detector_post_processor_conf

        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetNuscEvaluator(class_names=self.class_names,
                                          data_root=data_root,
                                          version=data_root_version,
                                          output_dir=self.default_root_dir)
        
        self.is_bev = False
        self.is_monocular = False
        self.run_mode = run_mode
        if self.run_mode in ['bevdepth', 'bevcorner']:
            self.is_bev = True
        if self.run_mode in ['monoflex', 'bevcorner']:
            self.is_monocular = True

        self.model = BaseBEVDepth(self.backbone_conf,
                                  monoflex_backbone_conf,
                                  self.head_conf,
                                  self.detector_predictor_conf,
                                  self.detector_loss_conf,
                                  self.detector_post_processor_conf,
                                  is_train_depth=True,
                                    is_bev=self.is_bev,
                                    is_monocular=self.is_monocular,
                                  )

        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.data_return_depth = True
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.dbound = self.backbone_conf['d_bound']
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.use_fusion = False
        self.train_info_paths = os.path.join(self.data_root,
                                             'nuscenes_infos_train.bevcorner2.pkl')
        self.val_info_paths = os.path.join(self.data_root,
                                           'nuscenes_infos_val.bevcorner2.pkl')
        self.predict_info_paths = os.path.join(self.data_root,
                                               'nuscenes_infos_test.bevcorner2.pkl')
        
        self.filter_params = filter_params
        self.down_ratio = down_ratio
        self.orientation_method = orientation_method
        self.multibin_size = multibin_size
        self.heatmap_ratio = heatmap_ratio

        self.width_train = final_dim[1]  # monoflex
        self.height_train = final_dim[0]
        

    def forward(self, sweep_imgs, mats, targets_monoflex):
        return self.model(sweep_imgs, mats, targets_monoflex=targets_monoflex)

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels, obj_lvl_depth_labels, targets_monocular) = batch
        if torch.cuda.is_available():
            sweep_imgs = sweep_imgs.cuda()
            if self.is_bev:
                for key, value in mats.items():
                    mats[key] = value.cuda()
                gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
                gt_labels = [gt_label.cuda() for gt_label in gt_labels]
            
        preds, depth_preds, loss_monocular = self(
            sweep_imgs, mats, targets_monocular)
        
        if self.is_monocular:
            (loss_dict, log_loss_dict) = loss_monocular
        
        targets=None
        detection_loss=0
        detection_loss_monocular=0
        depth_loss=0
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            if self.is_bev:
                targets = self.model.module.get_targets(sweep_imgs, gt_boxes, gt_labels)
                detection_loss = self.model.module.loss(targets, preds)
            
            if self.is_monocular:
                detection_loss_monocular = self.model.module.loss_monoflex(loss_dict, log_loss_dict)
        else:
            if self.is_bev:
                targets = self.model.get_targets(sweep_imgs, gt_boxes, gt_labels)
                detection_loss = self.model.loss(targets, preds)
            
            if self.is_monocular:
                detection_loss_monocular = self.model.loss_monoflex(loss_dict, log_loss_dict)

       
        if self.is_bev:
            if len(depth_labels.shape) == 5:
                # only key-frame will calculate depth loss
                depth_labels = depth_labels[:, 0, ...]
            depth_loss = self.get_depth_loss(
                depth_labels.cuda(), 
                # obj_lvl_depth_labels.cuda() if obj_lvl_depth_labels is not None else None, 
                None,
                depth_preds, sweep_imgs
            )
            self.log('detection_loss', detection_loss)
            self.log('depth_loss', depth_loss)
        
        
        if self.is_monocular:
            self.log('detection_loss_monoflex', detection_loss_monocular)
        return detection_loss + depth_loss + detection_loss_monocular


    def get_depth_loss(self, depth_labels_in, obj_lvl_depth_labels_in, depth_preds_in, sweep_imgs):
        depth_labels = self.get_downsampled_gt_depth(depth_labels_in)
        
        if obj_lvl_depth_labels_in is not None:
            obj_lvl_depth_labels = self.get_downsampled_gt_3d_depth(obj_lvl_depth_labels_in).cuda()
        
        combined_depth_labels = depth_labels.clone()
        
        if obj_lvl_depth_labels_in is not None:
            mask = combined_depth_labels < obj_lvl_depth_labels
            combined_depth_labels[mask] = obj_lvl_depth_labels[mask]
        
            cumsum_indicator = torch.cumsum(combined_depth_labels, dim=-1)
            mask = (combined_depth_labels==1) & (cumsum_indicator == 1)
            combined_depth_labels *= mask

        depth_preds = depth_preds_in.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(combined_depth_labels, dim=1).values > 0.0

        
        # for visualization debug purpose
        if obj_lvl_depth_labels_in is not None:
            cam_idx = 2
            b_idx = 0
            B, N, D, H, W = obj_lvl_depth_labels_in.shape
            if obj_lvl_depth_labels_in is not None:
                obj_lvl_depth_labels_vis = obj_lvl_depth_labels.reshape(B * N,
                    H // self.downsample_factor,
                    W // self.downsample_factor,
                    D
                )[b_idx * N + cam_idx].permute(2, 0, 1)
            depth_labels_vis = depth_labels.reshape(B * N,
                H // self.downsample_factor,
                W // self.downsample_factor,
                D
            )[b_idx * N + cam_idx].permute(2, 0, 1)
            combined_depth_labels_vis = combined_depth_labels.reshape(B * N,
                H // self.downsample_factor,
                W // self.downsample_factor,
                D
            )[b_idx * N + cam_idx].permute(2, 0, 1)
        
        # TODO: debug steps
        # run visualize_3d_depth(obj_lvl_depth_labels_vis, need_ratio_down=False)
        # in_obj_lvl_depth_labels_vis = obj_lvl_depth_labels_in[b_idx][cam_idx]
        # run visualize_3d_depth(in_obj_lvl_depth_labels_vis)
        # 
        # obj_lvl_depth_labels_vis[obj_lvl_depth_labels_vis > 0].shape
        # run visualize_3d_depth(obj_lvl_depth_labels_vis)
        # sweep_img_vis = visualize_from_image_arr(sweep_imgs[b_idx][0][cam_idx]) # batch=0, id_sweep=0, idx cam
        # see the ori imgs and compared, sweep_img_vis.save('depth_{}_cam.png.png'.format(cam_idx))
        

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                combined_depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        # gt_depths background is zero value
        return gt_depths.float()
    

    def get_downsampled_gt_3d_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, D, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, D, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            D,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 2, 4, 6, 3, 5).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)

        gt_depths = (gt_depths == 1).any(dim=-1).long()
        gt_depths = gt_depths.view(
            B * N,
            D,
            H // self.downsample_factor,
            W // self.downsample_factor,
        )  # value is either 0 or 1

        # B*N, H, W, D
        gt_depths = gt_depths.permute(0, 2, 3, 1).contiguous()

        bn_indices, h_indices, w_indices, d_indices = torch.nonzero(gt_depths, as_tuple=True)

        gt_depths_new = torch.zeros((
            B * N,
            H // self.downsample_factor,
            W // self.downsample_factor,
            D
        ))

        d_indices = (d_indices - (self.dbound[0] - self.dbound[2])) / self.dbound[2]

        d_indices = torch.where(
            (d_indices < self.depth_channels + 1) & (d_indices >= 1.0),
            d_indices - 1, torch.zeros_like(d_indices))
        
        mask = d_indices > 0
        gt_depths_new[bn_indices[mask], h_indices[mask], w_indices[mask], d_indices[mask].long()] = 1
        gt_depths_new = gt_depths_new.view(-1, self.depth_channels)
        return gt_depths_new.float()


    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _, _, _, targets_monoflex) = batch
        if self.is_monocular and not self.is_bev:
            cpu_device = torch.device("cpu")
            dis_ious = defaultdict(list)
            depth_errors = defaultdict(list)
            sweep_imgs = sweep_imgs.cuda()
            _, _, (result, eval_utils, visualize_preds, pred_props) = self.model(sweep_imgs, None, targets_monoflex=targets_monoflex)
            output = result
            output = output.to(cpu_device)

            dis_iou = eval_utils['dis_ious']

            if dis_iou is not None:
                for key in dis_iou: dis_ious[key] += dis_iou[key].tolist()

            if False: show_image_with_boxes_in_mono(img_metas[0]['ori_img'], output, targets_monoflex[0], 
                                    visualize_preds, vis_scores=eval_utils['vis_scores'])
            return (result, eval_utils, visualize_preds, pred_props, dis_ious)
        else:
            
            if torch.cuda.is_available():
                for key, value in mats.items():
                    mats[key] = value.cuda()
                sweep_imgs = sweep_imgs.cuda()
            preds = self.model(sweep_imgs, mats, targets_monoflex=targets_monoflex)
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                results = self.model.module.get_bboxes(preds, img_metas)
            else:
                results = self.model.get_bboxes(preds, img_metas)
            for i in range(len(results)):
                results[i][0] = results[i][0].detach().cpu().numpy()
                results[i][1] = results[i][1].detach().cpu().numpy()
                results[i][2] = results[i][2].detach().cpu().numpy()
                results[i].append(img_metas[i])
            return results

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        if self.is_monocular and not self.is_bev:
            logger = logging.getLogger("monoflex.inference")
            dis_ious = validation_step_outputs
            # disentangling IoU
            for key, value in dis_ious.items():
                mean_iou = sum(value) / len(value)
                dis_ious[key] = mean_iou

            for key, value in dis_ious.items():
                logger.info("{}, MEAN IOU = {:.4f}".format(key, value))

            logger.info('Finishing generating predictions, start evaluating ...')
            logger.info('Unsupported')
            ret_dicts = []

            """
            for metric in metrics:
                result, ret_dict = evaluate_python(label_path=dataset.label_dir, 
                                                result_path=predict_folder,
                                                label_split_file=dataset.imageset_txt,
                                                current_class=dataset.classes,
                                                metric=metric)

                logger.info('metric = {}'.format(metric))
                logger.info('\n' + result)

                ret_dicts.append(ret_dict)
            """

            return
        
        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])
        synchronize()
        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def test_epoch_end(self, test_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()
        # TODO: Change another way.
        dataset_length = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:dataset_length]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]

    def train_dataloader(self):
        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                        bda_aug_conf=self.bda_aug_conf,
                                        classes=self.class_names,
                                        data_root=self.data_root,
                                        info_paths=self.train_info_paths,
                                        is_train=True,
                                        use_cbgs=self.data_use_cbgs,
                                        img_conf=self.img_conf,
                                        num_sweeps=self.num_sweeps,
                                        sweep_idxes=self.sweep_idxes,
                                        key_idxes=self.key_idxes,
                                        return_depth=self.data_return_depth,
                                        use_fusion=self.use_fusion,
                                        filter_params=self.filter_params,
                                        down_ratio=self.down_ratio,
                                        orientation_method=self.orientation_method,
                                        multibin_size=self.multibin_size,
                                        edge_heatmap_ratio=self.heatmap_ratio,
                                        width_train=self.width_train,
                                        height_train=self.height_train,
                                        depth_channel=self.depth_channels
                                        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn, is_bev=self.is_bev, 
                               is_monocular=self.is_monocular, 
                               is_return_depth=self.data_return_depth
                               or self.use_fusion),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                        bda_aug_conf=self.bda_aug_conf,
                                        classes=self.class_names,
                                        data_root=self.data_root,
                                        info_paths=self.val_info_paths,
                                        is_train=False,
                                        img_conf=self.img_conf,
                                        num_sweeps=self.num_sweeps,
                                        sweep_idxes=self.sweep_idxes,
                                        key_idxes=self.key_idxes,
                                        return_depth=self.use_fusion,
                                        use_fusion=self.use_fusion,
                                        filter_params=self.filter_params,
                                        down_ratio=self.down_ratio,
                                        orientation_method=self.orientation_method,
                                        multibin_size=self.multibin_size,
                                        edge_heatmap_ratio=self.heatmap_ratio,
                                        width_train=self.width_train,
                                        height_train=self.height_train,
                                        depth_channel=self.depth_channels,
                                        is_bev = self.is_bev,
                                        is_monocular = self.is_monocular,
                                        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_bev=self.is_bev, 
                               is_monocular=self.is_monocular, 
                               is_return_depth=self.use_fusion),
            num_workers=4,
            sampler=None,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        predict_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                            bda_aug_conf=self.bda_aug_conf,
                                            classes=self.class_names,
                                            data_root=self.data_root,
                                            info_paths=self.predict_info_paths,
                                            is_train=False,
                                            img_conf=self.img_conf,
                                            num_sweeps=self.num_sweeps,
                                            sweep_idxes=self.sweep_idxes,
                                            key_idxes=self.key_idxes,
                                            return_depth=self.use_fusion,
                                            use_fusion=self.use_fusion,
                                            filter_params=self.filter_params,
                                            down_ratio=self.down_ratio,
                                            orientation_method=self.orientation_method,
                                            multibin_size=self.multibin_size,
                                            edge_heatmap_ratio=self.heatmap_ratio,
                                            width_train=self.width_train,
                                            height_train=self.height_train,
                                            depth_channel=self.depth_channels
                                        )
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_bev=self.is_bev, 
                               is_monocular=self.is_monocular, 
                               is_return_depth=self.use_fusion),
            num_workers=4,
            sampler=None,
        )
        return predict_loader

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'predict')

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser

    def load_model(self, ckpt_path):
        # ckpt_path_good = "pretrained/bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
        # state_dict = torch.load(ckpt_path, map_location=torch.device("cuda"))['state_dict']

        if isinstance(ckpt_path, list):
            ckpt_raw={}
            ckpt_raw_bevdepth = torch.load(ckpt_path[0], map_location=torch.device("cuda"))
            ckpt_raw_monoflex = torch.load(ckpt_path[1], map_location=torch.device("cuda"))
            state_dict_bevdepth = ckpt_raw_bevdepth['state_dict']
            # state_dict_monoflex = ckpt_raw_monoflex['model']
            state_dict_monoflex = ckpt_raw_monoflex['state_dict']

            for key in list(state_dict_bevdepth.keys()):
                new_key = key.replace("model.", "")
                if new_key != key:
                    state_dict_bevdepth[new_key] = state_dict_bevdepth[key]
                    del state_dict_bevdepth[key]
            
            for key in list(state_dict_monoflex.keys()):
                new_key = key.replace("model.", "")
                if new_key != key:
                    state_dict_monoflex[new_key] = state_dict_monoflex[key]
                    del state_dict_monoflex[key]

            state_dict = {}
            state_dict.update(state_dict_monoflex)
            state_dict.update(state_dict_bevdepth)

        else:
            ckpt_raw = torch.load(ckpt_path, map_location=torch.device("cuda"))

            if 'state_dict' in ckpt_raw:
                state_dict = ckpt_raw['state_dict']
                for key in list(state_dict.keys()):
                    new_key = key.replace("model.", "")
                    if new_key != key:
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]
                    
            elif 'model' in ckpt_raw:  # monoflex pretrained
                state_dict = ckpt_raw['model']
                for key in list(state_dict.keys()):
                    new_key = key.replace("backbone.", "backbone_monoflex.")
                    new_key = new_key.replace("heads.predictor.", "detector_predictor.")
                    
                    if new_key != key:
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]
                        
                    if state_dict[new_key].shape == self.model.state_dict()[new_key].shape:
                        pass # do nothing
                    else:
                        print(f"Ignoring {new_key} due to shape mismatch: "
                            f"checkpoint shape {state_dict[new_key].shape}, "
                            f"model shape {self.model.state_dict()[new_key].shape}")
                        del state_dict[new_key]
            else:
                raise NotImplementedError('Supposed load state dict error')
            
        
        # state_dict_good = torch.load(ckpt_path_good, map_location=torch.device("cuda"))['state_dict']

        # state_dict = {k.partition('model.')[2]: state_dict[k] for k in state_dict.keys()}
        state_dict = {k: state_dict[k] for k in state_dict.keys()}
        # state_dict_good = {k.partition('model.')[2]: state_dict_good[k] for k in state_dict_good.keys()}
        
        # state_dict.update(state_dict_good)
        if 'model' in ckpt_raw:  # monoflex pretrained
            self.model.load_state_dict(state_dict, strict=False)
            
        else:
            if not self.is_monocular and self.is_bev:
                self.model.load_state_dict(state_dict, strict=False)
            else:  # bevdepth pretrained
                self.model.load_state_dict(state_dict, strict=False)  # change to False for very first time

            
        # if self.is_monocular and not self.is_bev:
        #     for name, param in self.model.named_parameters():
        #         # Example: Freeze all parameters that contain "backbone_monoflex" in their name
        #         if "backbone_monoflex" in name:
        #             param.requires_grad = False
                    
        # if not self.is_monocular and self.is_bev:
        #     for name, param in self.model.named_parameters():
        #         # Example: Freeze all parameters that contain "backbone_monoflex" in their name
        #         if "backbone" in name:
        #             param.requires_grad = False
                    
            # Verify which parameters are frozen
            # for name, param in self.model.named_parameters():
            #     print(f"{name}: requires_grad = {param.requires_grad}")
            
            
    def load_model_bevcorner(self):
        # ckpt_path_good = "pretrained/bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
        # state_dict = torch.load(ckpt_path, map_location=torch.device("cuda"))['state_dict']
        ckpt_monoflex = "pretrained/monoflex/model_moderate_best_soft.pth"
        ckpt_path_bevdepth = "pretrained/bevdepth/bev_depth_lss_r50_256x704_128x128_24e_2key.pth"

        ckpt_raw_monoflex = torch.load(ckpt_monoflex, map_location=torch.device("cuda"))
        ckpt_raw_bevdepth = torch.load(ckpt_path_bevdepth, map_location=torch.device("cuda"))

        state_dict_monoflex = ckpt_raw_monoflex['state_dict']
        state_dict_bevdepth = ckpt_raw_bevdepth['state_dict']
        state_dict = {}
        state_dict.update(state_dict_monoflex)
        state_dict.update(state_dict_bevdepth)
    
        for key in list(state_dict.keys()):
            new_key = key.replace("model.", "")
            if new_key != key:
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        state_dict = {k: state_dict[k] for k in state_dict.keys()}
        # state_dict_good = {k.partition('model.')[2]: state_dict_good[k] for k in state_dict_good.keys()}
        
        # state_dict.update(state_dict_good)
        if not self.is_monocular and self.is_bev:
            self.model.load_state_dict(state_dict, strict=False)
        else:  # bevdepth pretrained
            self.model.load_state_dict(state_dict, strict=True)
            
        # if self.is_monocular and not self.is_bev:
        #     for name, param in self.model.named_parameters():
        #         # Example: Freeze all parameters that contain "backbone_monoflex" in their name
        #         if "backbone_monoflex" in name:
        #             param.requires_grad = False
                    
        # if not self.is_monocular and self.is_bev:
        #     for name, param in self.model.named_parameters():
        #         # Example: Freeze all parameters that contain "backbone_monoflex" in their name
        #         if "backbone" in name:
        #             param.requires_grad = False
                    
            # Verify which parameters are frozen
            # for name, param in self.model.named_parameters():
            #     print(f"{name}: requires_grad = {param.requires_grad}")

