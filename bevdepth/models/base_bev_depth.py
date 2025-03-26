from torch import nn

from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN

from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN

from bevdepth.layers.heads.bev_depth_head import BEVDepthHead
from bevdepth.layers.heads.bev_corner_head import BEVCornerHead

from bevdepth.layers.backbones.monoflex_backbone import build_backbone_monoflex

from bevdepth.layers.heads.head import make_predictor, make_loss_evaluator, make_post_processor

from bevdepth.utils import visualization as VisTool

__all__ = ['BaseBEVDepth']


class BaseBEVDepth(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(
        self, backbone_conf, monoflex_backbone_conf, head_conf, detector_predictor_conf, detector_loss_conf, 
        detector_post_processor_conf, is_train_depth=False, 
        is_bev=True, 
        is_monocular=True,
    ):
        super(BaseBEVDepth, self).__init__()
        
        
        self.backbone = None
        self.backbone_monoflex = None
        self.is_train_depth = is_train_depth
        
        self.is_bev = is_bev
        self.is_monocular = is_monocular
        
        if self.is_bev:
            self.backbone = BaseLSSFPN(**backbone_conf, is_monocular=is_monocular)
            self.head = BEVCornerHead(
                **head_conf
            )
        
        if self.is_monocular:
            self.backbone_monoflex = build_backbone_monoflex(monoflex_backbone_conf)
        # self.head = BEVDepthHead(**head_conf)        
            self.detector_predictor = make_predictor(detector_predictor_conf, in_channels=self.backbone_monoflex.out_channels)
            self.detector_loss = make_loss_evaluator(detector_loss_conf)
            self.detector_post_processor = make_post_processor(detector_post_processor_conf)
        


    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
        targets_monoflex=None
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = x.shape
        ret_values_corner = None

        pred_props_monocular = None
        loss_monocular = None
        
        if self.is_monocular:
            
            img_feats =  self.backbone_monoflex(x)

            # img_feats shape = (batch_size, num_sweeps, num_cams, c, h, w)
            img_feats = img_feats[:, 0, :num_cams]
            # only filter when idx sweep = 0 per batch
            img_feats = img_feats.reshape(-1, img_feats.shape[2], img_feats.shape[3], img_feats.shape[4])

            ret_values_corner = self.detector_predictor(img_feats, targets=targets_monoflex)  

            if self.training:
                loss_dict, log_loss_dict, pred_props, pred_target_props = self.detector_loss(ret_values_corner, targets=targets_monoflex, img_vis_batch=x)
                loss_monocular = (loss_dict, log_loss_dict) 
                pred_props_monocular = pred_target_props

                if not self.is_bev: 
                    return None, None, loss_monocular

            else:
                # loss_dict, log_loss_dict, pred_props, pred_target_props = self.detector_loss(ret_values_corner, targets=targets_monoflex)
    
                result, eval_utils, visualize_preds, pred_props = self.detector_post_processor(
                    ret_values_corner, targets=targets_monoflex, test=True, features=img_feats
                )
                pred_props_monocular = pred_props

                if not self.is_bev: 
                    return None, None, (result, eval_utils, visualize_preds, pred_props)


        # the code below is applied if self.bev==True
        if self.is_train_depth and self.training:
            # The experiment is failed, so i put groundtruth directly instead
            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          monoflex_props=pred_props_monocular,
                                          sweep_intrins = mats_dict['intrin_mats'],
                                          is_return_depth=True)

            preds = self.head(x)
            return preds, depth_pred, loss_monocular
        else:
            x = self.backbone(x, mats_dict, timestamps, monoflex_props=pred_props_monocular)
            preds = self.head(x)
            return preds
    

    def get_targets(self, imgs, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        targets = self.head.get_targets(gt_boxes, gt_labels)
        heatmaps = targets[0]
        anno_boxes = targets[1]
        inds = targets[2]
        masks = targets[3]
        
        # VisTool.visualize(imgs[0], heatmaps[0], anno_boxes[0], inds[0], masks[0])
        return targets

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def loss_monoflex(self, loss_dict, log_loss_dict):
        """Loss function for Monoflex."""
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        log_losses_reduced = sum(loss for key, loss in log_loss_dict.items() if key.find('loss') >= 0)
        return losses


    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
