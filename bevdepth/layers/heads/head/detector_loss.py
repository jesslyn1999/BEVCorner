import torch
import math
import torch.distributed as dist
import pdb
import random
from torch.nn import functional as F

from .ops.anno_encoder import Anno_Encoder
from .layers.utils import select_point_of_interest
from .ops.utils import Uncertainty_Reg_Loss, Laplace_Loss

from .layers.focal_loss import *
from .layers.iou_loss import *
from .ops.depth_losses import *
from .layers.utils import Converter_key2channel


from bevdepth.utils.visualization import visualize_from_image_arr, visualize_image_with_bboxes_and_points, visualize_heatmap_per_cam_obj

def make_loss_evaluator(cfg):
	loss_evaluator = Loss_Computation(**cfg)
	return loss_evaluator

class Loss_Computation():
	def __init__(
		self, 
		anno_encoder_conf,
		regression_heads,
		regression_channels,
		max_objs,
		center_sample,
		regress_area,
		heatmap_type,
		supervise_corner_depth,
		loss_names,
		dimension_weight,
		uncertainty_range,
		loss_type,
		loss_penalty_alpha,
		loss_beta,
		orientation,
		orientation_bin_size,
		truncation_offset_loss,
		init_loss_weight,
		uncertainty_weight,
		keypoint_xy_weights,
		keypoint_norm_factor,
		modify_invalid_keypoint_depths,
  		corner_loss_depth
	):
		self.anno_encoder = Anno_Encoder(**anno_encoder_conf)
		self.key2channel = Converter_key2channel(keys=regression_heads, channels=regression_channels)

		self.max_objs = max_objs
		self.center_sample = center_sample
		self.regress_area = regress_area
		self.heatmap_type = heatmap_type
		self.corner_depth_sp = supervise_corner_depth
		self.loss_keys = loss_names

		self.dim_weight = torch.as_tensor(dimension_weight).view(1, 3)
		self.uncertainty_range = uncertainty_range

		# loss functions
		loss_types = loss_type
		self.cls_loss_fnc = FocalLoss(loss_penalty_alpha, loss_beta) # penalty-reduced focal loss
		self.iou_loss = IOULoss(loss_type=loss_types[2]) # iou loss for 2D detection

		# depth loss
		if loss_types[3] == 'berhu': self.depth_loss = Berhu_Loss()
		elif loss_types[3] == 'inv_sig': self.depth_loss = Inverse_Sigmoid_Loss()
		elif loss_types[3] == 'log': self.depth_loss = Log_L1_Loss()
		elif loss_types[3] == 'L1': self.depth_loss = F.l1_loss
		else: raise ValueError

		# regular regression loss
		self.reg_loss = loss_types[1]
		self.reg_loss_fnc = F.l1_loss if loss_types[1] == 'L1' else F.smooth_l1_loss
		self.keypoint_loss_fnc = F.l1_loss

		# multi-bin loss setting for orientation estimation
		self.multibin = (orientation == 'multi-bin')
		self.orien_bin_size = orientation_bin_size
		self.trunc_offset_loss_type = truncation_offset_loss

		self.loss_weights = {}
		for key, weight in zip(self.loss_keys, init_loss_weight): self.loss_weights[key] = weight

		# whether to compute corner loss
		self.compute_direct_depth_loss = 'depth_loss' in self.loss_keys
		self.compute_keypoint_depth_loss = 'keypoint_depth_loss' in self.loss_keys
		self.compute_weighted_depth_loss = 'weighted_avg_depth_loss' in self.loss_keys
		self.compute_corner_loss = 'corner_loss' in self.loss_keys
		self.separate_trunc_offset = 'trunc_offset_loss' in self.loss_keys

		self.pred_direct_depth = 'depth' in self.key2channel.keys
		self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
		self.compute_keypoint_corner = 'corner_offset' in self.key2channel.keys
		self.corner_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys

		self.uncertainty_weight = uncertainty_weight # 1.0
		self.keypoint_xy_weights = keypoint_xy_weights # [1, 1]
		self.keypoint_norm_factor = keypoint_norm_factor # 1.0
		self.modify_invalid_keypoint_depths = modify_invalid_keypoint_depths

		# depth used to compute 8 corners
		self.corner_loss_depth = corner_loss_depth
		self.eps = 1e-5

	def prepare_targets(self, targets):
		# clses
		heatmaps = torch.stack([t.get_field("hm") for t in targets])

		batch = heatmaps.shape[0]
		num_cams = heatmaps.shape[1]

		cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
		offset_3D = torch.stack([t.get_field("offset_3D") for t in targets])

		size = torch.stack([torch.tensor(t.size) for t in targets])
		resize_dims = torch.stack([torch.tensor(t.resize_dims) for t in targets])
		resize_dims_mini = torch.stack([torch.tensor(t.resize_dims_mini) for t in targets])

		# 2d detection
		target_centers = torch.stack([t.get_field("target_centers") for t in targets])
		bboxes = torch.stack([t.get_field("2d_bboxes") for t in targets])
		# 3d detection
		keypoints = torch.stack([t.get_field("keypoints") for t in targets])
		# keypoints_scale = torch.stack([t.get_field("keypoints_scale") for t in targets])
		keypoints_depth_mask = torch.stack([t.get_field("keypoints_depth_mask") for t in targets])
		dimensions = torch.stack([t.get_field("dimensions") for t in targets])
		locations = torch.stack([t.get_field("locations") for t in targets])
		rotys = torch.stack([t.get_field("rotys") for t in targets])
		alphas = torch.stack([t.get_field("alphas") for t in targets])
		orientations = torch.stack([t.get_field("orientations") for t in targets])
		# utils
		pad_size = torch.stack([t.get_field("pad_size") for t in targets])
		calibs = [t.get_field("calib") for t in targets]
		reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
		reg_weight = torch.stack([t.get_field("reg_weight") for t in targets])
		# ori_imgs = torch.stack([t.get_field("ori_img") for t in targets])
		trunc_mask = torch.stack([t.get_field("trunc_mask") for t in targets])


		## reshape The first two dimensions (batch, num_cams) shall be merged into one
		heatmaps = heatmaps.reshape(((-1,) + heatmaps.shape[2:]))
		cls_ids = cls_ids.reshape(((-1,) + cls_ids.shape[2:]))
		offset_3D = offset_3D.reshape(((-1,) + offset_3D.shape[2:]))
		target_centers = target_centers.reshape(((-1,) + target_centers.shape[2:]))
		bboxes = bboxes.reshape(((-1,) + bboxes.shape[2:]))
		keypoints = keypoints.reshape((-1,) + keypoints.shape[2:])
		# keypoints_scale = keypoints_scale.reshape((-1,) + keypoints_scale.shape[2:])
		keypoints_depth_mask = keypoints_depth_mask.reshape((-1,) + keypoints_depth_mask.shape[2:])
		dimensions = dimensions.reshape((-1,) + dimensions.shape[2:])
		locations = locations.reshape((-1,) + locations.shape[2:])
		rotys = rotys.reshape((-1,) + rotys.shape[2:])
		alphas = alphas.reshape((-1,) + alphas.shape[2:])
		orientations = orientations.reshape((-1,) + orientations.shape[2:])
		pad_size = pad_size.reshape((-1,) + pad_size.shape[2:])
		calibs = [item for sublist in calibs for item in sublist]
		reg_mask = reg_mask.reshape((-1,) + reg_mask.shape[2:])
		reg_weight = reg_weight.reshape((-1,) + reg_weight.shape[2:])
		# ori_imgs = ori_imgs.reshape((-1,) + ori_imgs.shape[2:])
		trunc_mask = trunc_mask.reshape((-1,) + trunc_mask.shape[2:])


		return_dict = dict(cls_ids=cls_ids, target_centers=target_centers, bboxes=bboxes, keypoints=keypoints, dimensions=dimensions,
			locations=locations, rotys=rotys, alphas=alphas, calib=calibs, pad_size=pad_size, reg_mask=reg_mask, reg_weight=reg_weight,
			offset_3D=offset_3D, trunc_mask=trunc_mask, orientations=orientations, keypoints_depth_mask=keypoints_depth_mask,
			size=size, resize_dims=resize_dims, resize_dims_mini=resize_dims_mini
		)

		return heatmaps, return_dict, batch, num_cams

	def prepare_predictions(self, targets_variables, predictions):
		pred_regression = predictions['reg']
		batch, channel, feat_h, feat_w = pred_regression.shape

		# 1. get the representative points
		targets_bbox_points = targets_variables["target_centers"] # representative points

		reg_mask_gt = targets_variables["reg_mask"]
		flatten_reg_mask_gt = reg_mask_gt.view(-1).bool()

		# the corresponding image_index for each object, used for finding pad_size, calib and so on
		batch_idxs = torch.arange(batch).view(-1, 1).expand_as(reg_mask_gt).reshape(-1)
		batch_idxs = batch_idxs[flatten_reg_mask_gt].to(reg_mask_gt.device) 

		valid_targets_bbox_points = targets_bbox_points.view(-1, 2)[flatten_reg_mask_gt]

		# fcos-style targets for 2D
		target_bboxes_2D = targets_variables['bboxes'].view(-1, 4)[flatten_reg_mask_gt]
		target_bboxes_height = target_bboxes_2D[:, 3] - target_bboxes_2D[:, 1]
		target_bboxes_width = target_bboxes_2D[:, 2] - target_bboxes_2D[:, 0]

		target_regression_2D = torch.cat((valid_targets_bbox_points - target_bboxes_2D[:, :2], target_bboxes_2D[:, 2:] - valid_targets_bbox_points), dim=1)
		mask_regression_2D = (target_bboxes_height > 0) & (target_bboxes_width > 0)
		target_regression_2D = target_regression_2D[mask_regression_2D]
		valid_targets_bbox_points_2D = valid_targets_bbox_points[mask_regression_2D]

		# targets for 3D
		target_clses = targets_variables["cls_ids"].view(-1)[flatten_reg_mask_gt]
		target_depths_3D = targets_variables['locations'][..., -1].view(-1)[flatten_reg_mask_gt]
		target_rotys_3D = targets_variables['rotys'].view(-1)[flatten_reg_mask_gt]
		target_alphas_3D = targets_variables['alphas'].view(-1)[flatten_reg_mask_gt]
		target_offset_3D = targets_variables["offset_3D"].view(-1, 2)[flatten_reg_mask_gt]
		target_dimensions_3D = targets_variables['dimensions'].view(-1, 3)[flatten_reg_mask_gt]
		
		target_orientation_3D = targets_variables['orientations'].view(-1, targets_variables['orientations'].shape[-1])[flatten_reg_mask_gt]
		target_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, target_offset_3D, target_depths_3D, 
										targets_variables['calib'], targets_variables['pad_size'], batch_idxs)

		if flatten_reg_mask_gt[flatten_reg_mask_gt==1].numel() == 0:
			target_corners_3D = torch.zeros_like(target_rotys_3D)
		else:
			target_corners_3D = self.anno_encoder.encode_box3d(target_rotys_3D, target_dimensions_3D, target_locations_3D)
		target_bboxes_3D = torch.cat((target_locations_3D, target_dimensions_3D, target_rotys_3D[:, None]), dim=1)

		target_trunc_mask = targets_variables['trunc_mask'].view(-1)[flatten_reg_mask_gt]
		obj_weights = targets_variables["reg_weight"].view(-1)[flatten_reg_mask_gt]

		# 2. extract corresponding predictions
		pred_regression_pois_3D = select_point_of_interest(batch, targets_bbox_points, pred_regression).view(-1, channel)[flatten_reg_mask_gt]
		
		pred_regression_2D = F.relu(pred_regression_pois_3D[mask_regression_2D, self.key2channel('2d_dim')])
		pred_offset_3D = pred_regression_pois_3D[:, self.key2channel('3d_offset')]
		pred_dimensions_offsets_3D = pred_regression_pois_3D[:, self.key2channel('3d_dim')]
		pred_orientation_3D = torch.cat((pred_regression_pois_3D[:, self.key2channel('ori_cls')], 
									pred_regression_pois_3D[:, self.key2channel('ori_offset')]), dim=1)
		
		# decode the pred residual dimensions to real dimensions
		pred_dimensions_3D = self.anno_encoder.decode_dimension(target_clses, pred_dimensions_offsets_3D)

		# preparing outputs
		targets = { 'reg_2D': target_regression_2D, 'offset_3D': target_offset_3D, 'depth_3D': target_depths_3D, 'orien_3D': target_orientation_3D,
					'dims_3D': target_dimensions_3D, 'corners_3D': target_corners_3D, 'width_2D': target_bboxes_width, 'rotys_3D': target_rotys_3D,
					'cat_3D': target_bboxes_3D, 'trunc_mask_3D': target_trunc_mask, 'height_2D': target_bboxes_height,
				}

		preds = {'reg_2D': pred_regression_2D, 'offset_3D': pred_offset_3D, 'orien_3D': pred_orientation_3D, 'dims_3D': pred_dimensions_3D}
		
		reg_nums = {'reg_2D': mask_regression_2D.sum(), 'reg_3D': flatten_reg_mask_gt.sum(), 'reg_obj': flatten_reg_mask_gt.sum()}
		weights = {'object_weights': obj_weights}

		# predict the depth with direct regression
		if self.pred_direct_depth:
			pred_depths_offset_3D = pred_regression_pois_3D[:, self.key2channel('depth')].squeeze(-1)
			pred_direct_depths_3D = self.anno_encoder.decode_depth(pred_depths_offset_3D)
			preds['depth_3D'] = pred_direct_depths_3D

		# predict the uncertainty of depth regression
		if self.depth_with_uncertainty:
			preds['depth_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('depth_uncertainty')].squeeze(-1)
			
			if self.uncertainty_range is not None:
				preds['depth_uncertainty'] = torch.clamp(preds['depth_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

			# else:
			# 	print('depth_uncertainty: {:.2f} +/- {:.2f}'.format(
			# 		preds['depth_uncertainty'].mean().item(), preds['depth_uncertainty'].std().item()))

		# predict the keypoints
		if self.compute_keypoint_corner:
			# targets for keypoints
			target_corner_keypoints = targets_variables["keypoints"].view(flatten_reg_mask_gt.shape[0], -1, 3)[flatten_reg_mask_gt]

			targets['keypoints'] = target_corner_keypoints[..., :2]
			targets['keypoints_mask'] = target_corner_keypoints[..., -1]
		
			reg_nums['keypoints'] = targets['keypoints_mask'].sum()

			# mask for whether depth should be computed from certain group of keypoints
			target_corner_depth_mask = targets_variables["keypoints_depth_mask"].view(-1, 3)[flatten_reg_mask_gt]
			targets['keypoints_depth_mask'] = target_corner_depth_mask

			# predictions for keypoints
			pred_keypoints_3D = pred_regression_pois_3D[:, self.key2channel('corner_offset')]

			if flatten_reg_mask_gt.sum() == 0:
				pred_keypoints_3D = torch.zeros((0, int(pred_keypoints_3D.shape[-1]/2), 2))
			else:
				pred_keypoints_3D = pred_keypoints_3D.view(flatten_reg_mask_gt.sum(), -1, 2)

			if 'keypoints_scale' in targets:
				target_keypoints_scale = targets_variables["keypoints_scale"].view(flatten_reg_mask_gt.shape[0], -1)[flatten_reg_mask_gt]
				pred_keypoints_3D = pred_keypoints_3D * target_keypoints_scale.view(-1, 1, 1)

				# 4. Validate predictions
				pred_keypoints_3D = torch.nan_to_num(pred_keypoints_3D, 0.0)
				pred_keypoints_3D = torch.clamp(pred_keypoints_3D, -100, 100)  # prevent extreme values
            
		if flatten_reg_mask_gt.sum() == 0:
			pred_keypoints_depths_3D = torch.zeros((pred_keypoints_3D.shape[0], 3)).cuda()
		else:
			pred_keypoints_depths_3D = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoints_3D, pred_dimensions_3D,
													targets_variables['calib'], batch_idxs)

		preds['keypoints'] = pred_keypoints_3D			
		preds['keypoints_depths'] = pred_keypoints_depths_3D

		# predict the uncertainties of the solved depths from groups of keypoints
		if self.corner_with_uncertainty:
			preds['corner_offset_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('corner_uncertainty')]

			if self.uncertainty_range is not None:
				preds['corner_offset_uncertainty'] = torch.clamp(preds['corner_offset_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

			# else:
			# 	print('keypoint depth uncertainty: {:.2f} +/- {:.2f}'.format(
			# 		preds['corner_offset_uncertainty'].mean().item(), preds['corner_offset_uncertainty'].std().item()))

		# compute the corners of the predicted 3D bounding boxes for the corner loss
		if self.corner_loss_depth == 'direct':
			pred_corner_depth_3D = pred_direct_depths_3D

		elif self.corner_loss_depth == 'keypoint_mean':
			pred_corner_depth_3D = preds['keypoints_depths'].mean(dim=1)
		
		else:
			assert self.corner_loss_depth in ['soft_combine', 'hard_combine']
			# make sure all depths and their uncertainties are predicted
			pred_combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(-1), preds['corner_offset_uncertainty']), dim=1).exp()
			pred_combined_depths = torch.cat((pred_direct_depths_3D.unsqueeze(-1), preds['keypoints_depths']), dim=1)
			
			if self.corner_loss_depth == 'soft_combine':
				pred_uncertainty_weights = 1 / pred_combined_uncertainty
				pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
				pred_corner_depth_3D = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)
				preds['weighted_depths'] = pred_corner_depth_3D
			
			elif self.corner_loss_depth == 'hard_combine':
				pred_corner_depth_3D = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(dim=1)]

		# compute the corners
		pred_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, pred_offset_3D, pred_corner_depth_3D, 
										targets_variables['calib'], targets_variables['pad_size'], batch_idxs)
		# decode rotys and alphas
		pred_rotys_3D, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D, pred_locations_3D)
		# encode corners
		if flatten_reg_mask_gt[flatten_reg_mask_gt==1].numel() == 0:
			pred_corners_3D = torch.zeros_like(pred_rotys_3D)
		else:
			pred_corners_3D = self.anno_encoder.encode_box3d(pred_rotys_3D, pred_dimensions_3D, pred_locations_3D)
		# concatenate all predictions
		pred_bboxes_3D = torch.cat((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[:, None]), dim=1)

		pred_bboxes_2D = self.anno_encoder.decode_box2d_fcos(
			valid_targets_bbox_points_2D, pred_regression_2D, targets_variables['pad_size'], targets_variables['resize_dims'], batch_idxs[mask_regression_2D])
		
		pred_bboxes_2D_small_ratio = self.anno_encoder.decode_box2d_fcos_ratio_same(
			valid_targets_bbox_points_2D, pred_regression_2D, targets_variables['pad_size'], batch_idxs=batch_idxs[mask_regression_2D])
  
		pred_target_bboxes_2D_small_ratio = self.anno_encoder.decode_box2d_fcos_ratio_same(
			valid_targets_bbox_points_2D, target_regression_2D, targets_variables['pad_size'], batch_idxs=batch_idxs[mask_regression_2D])

		preds.update({'corners_3D': pred_corners_3D, 'rotys_3D': pred_rotys_3D, 'cat_3D': pred_bboxes_3D[mask_regression_2D], 
				'bboxes_2d_mini': pred_bboxes_2D_small_ratio})
  
		targets.update({'bboxes_2d_mini': pred_target_bboxes_2D_small_ratio, 
                  'cat_3d_min': target_bboxes_3D[mask_regression_2D], 
                  'batch_idxs_min': batch_idxs[mask_regression_2D], 'mask_regression_2D': mask_regression_2D,
                  'valid_targets_bbox_points': valid_targets_bbox_points,
                  }
        )
		return targets, preds, reg_nums, weights, batch_idxs

	def __call__(self, predictions, targets, img_vis_batch=None):
		targets_heatmap, targets_variables, batch, num_cams = self.prepare_targets(targets)

		pred_heatmap = predictions['cls']
		pred_targets, preds, reg_nums, weights, batch_idxs = self.prepare_predictions(targets_variables, predictions)
  
		sweep_img_vis = None


		pred_props = {}
		pred_target_props = {}
		pred_props.update(reg_nums)

		pred_props.update({
			'heatmap_shape': pred_heatmap.shape[-2:],
			'target_centers': targets_variables['target_centers'],
			'cat_3D': preds['cat_3D'],
			'bboxes_2d_mini': preds['bboxes_2d_mini'],  # version down-ratio one
			'batch_idxs': pred_targets['batch_idxs_min'],
			'confs_score': None  # update below at iou_loss
		})

  
		pred_target_props.update({
			'heatmap_shape': targets_heatmap.shape[-2:],
			'cat_3D': pred_targets['cat_3d_min'],
			'bboxes_2d_mini': pred_targets['bboxes_2d_mini'],  # version down-ratio one
			'batch_idxs': pred_targets['batch_idxs_min'],
			'confs_score': torch.tensor([1]).repeat(pred_targets['cat_3d_min'].shape[0])
		})
  
		if False:  # change to True for visualization
			batch_vis, _, c_vis, channel_vis, h_vis, w_vis = img_vis_batch.shape 
			rand_batch = random.randint(0, batch_vis - 1)
			img_vis_zero_sweep = img_vis_batch[:, 0, ...].view(-1, c_vis, channel_vis, h_vis, w_vis)[rand_batch]
   
			up_ratio = 4
   
			for id_cam_vis in range(c_vis): # total camera of 6
				img_vis = visualize_from_image_arr(img_vis_zero_sweep[id_cam_vis])
				id_referred = rand_batch * c_vis + id_cam_vis
				# delta_keypoints = targets_variables['keypoints'][id_referred][targets_variables['reg_mask'][id_referred]==1][..., :2] \
				# 	* targets_variables['keypoints_scale'][id_referred][targets_variables['reg_mask'][id_referred]==1].view(-1, 1, 1)
				delta_keypoints = targets_variables['keypoints'][id_referred][targets_variables['reg_mask'][id_referred]==1][..., :2]
				t_points = ( delta_keypoints \
						+ (targets_variables['target_centers'][id_referred][targets_variables['reg_mask'][id_referred]==1]).view(
          					-1, 1, 2).repeat(1, delta_keypoints.shape[1], 1)
              		) * up_ratio
				visualize_image_with_bboxes_and_points(
    				img_vis, 
					t_bboxes=targets_variables['bboxes'][id_referred][targets_variables['reg_mask'][id_referred]==1] * up_ratio,
					t_points=t_points,
					id="b{}-cam{}-gt_loss".format(rand_batch, id_cam_vis))

				batch_idxs_min_vis = pred_targets['batch_idxs_min']
				batch_idxs_vis = batch_idxs

				preds_bboxes = preds['bboxes_2d_mini'][batch_idxs_min_vis==id_referred] * up_ratio
				preds_target_bboxes = pred_targets['bboxes_2d_mini'][batch_idxs_min_vis==id_referred] * up_ratio

				preds_target_delta_keypoints = pred_targets['keypoints'][batch_idxs==id_referred]
				preds_target_keypoints = preds_target_delta_keypoints + \
					pred_targets['valid_targets_bbox_points'][batch_idxs==id_referred].view(-1, 1, 2).repeat(1, preds_target_delta_keypoints.shape[1], 1)
				preds_keypoints = preds['keypoints'][batch_idxs==id_referred] \
        			+ pred_targets['valid_targets_bbox_points'][batch_idxs==id_referred].view(-1, 1, 2).repeat(1, preds_target_delta_keypoints.shape[1], 1)

				visualize_image_with_bboxes_and_points(
					img_vis, 
					t_bboxes=preds_bboxes, 
					t_points=preds_keypoints * up_ratio,
					id="b{}-cam{}-pred_loss".format(rand_batch, id_cam_vis))

				heatmap_cls_ids = targets_variables["cls_ids"][id_referred][targets_variables['reg_mask'][id_referred]==1]
				unique_elements, counts = torch.unique(heatmap_cls_ids, return_counts=True)
				id_cls_vis = unique_elements[torch.argmax(counts)].item() if counts.numel() else None
				if id_cls_vis is not None:
					visualize_heatmap_per_cam_obj(
						targets_heatmap[id_referred][id_cls_vis].cpu().detach().numpy(),
						id="b{}-cam{}-cls{}-gt_heatmap_loss".format(rand_batch, id_cam_vis, id_cls_vis)
					) 

					visualize_heatmap_per_cam_obj(
						pred_heatmap[id_referred][id_cls_vis].cpu().detach().numpy(),
						id="b{}-cam{}-cls{}-pred_heatmap_loss".format(rand_batch, id_cam_vis, id_cls_vis)
					)

		# heatmap loss
		if self.heatmap_type == 'centernet':
			hm_loss, num_hm_pos = self.cls_loss_fnc(pred_heatmap, targets_heatmap)
			hm_loss = self.loss_weights['hm_loss'] * hm_loss / torch.clamp(num_hm_pos, 1)

		else: raise ValueError

		# synthesize normal factors
		num_reg_2D = reg_nums['reg_2D']
		num_reg_3D = reg_nums['reg_3D']
		num_reg_obj = reg_nums['reg_obj']
		
		trunc_mask = pred_targets['trunc_mask_3D'].bool()
		num_trunc = trunc_mask.sum()
		num_nontrunc = num_reg_obj - num_trunc

		# IoU loss for 2D detection
		if num_reg_2D > 0:
			reg_2D_loss, iou_2D = self.iou_loss(preds['reg_2D'], pred_targets['reg_2D'])
			pred_props.update({
				'confs_score': 1- (self.loss_weights['bbox_loss'] * reg_2D_loss)
			})
			reg_2D_loss = self.loss_weights['bbox_loss'] * reg_2D_loss.mean()
			iou_2D = iou_2D.mean()
			depth_MAE = (preds['depth_3D'] - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']
			pred_props.update({
				'reg_2D': preds['reg_2D'],
				'depth_3D': preds['depth_3D']
			})

		if num_reg_3D > 0:
			# direct depth loss
			if self.compute_direct_depth_loss:
				depth_3D_loss = self.loss_weights['depth_loss'] * self.depth_loss(preds['depth_3D'], pred_targets['depth_3D'], reduction='none')
				real_depth_3D_loss = depth_3D_loss.detach().mean()
				
				if self.depth_with_uncertainty:
					depth_3D_loss = depth_3D_loss * torch.exp(- preds['depth_uncertainty']) + \
							preds['depth_uncertainty'] * self.loss_weights['depth_loss']

				depth_3D_loss = depth_3D_loss.mean()
				
			# offset_3D loss
			offset_3D_loss = self.reg_loss_fnc(preds['offset_3D'], pred_targets['offset_3D'], reduction='none').sum(dim=1)

			# use different loss functions for inside and outside objects
			if self.separate_trunc_offset:
				if self.trunc_offset_loss_type == 'L1':
					trunc_offset_loss = offset_3D_loss[trunc_mask]
				
				elif self.trunc_offset_loss_type == 'log':
					trunc_offset_loss = torch.log(1 + offset_3D_loss[trunc_mask])

				trunc_offset_loss = self.loss_weights['trunc_offset_loss'] * trunc_offset_loss.sum() / torch.clamp(trunc_mask.sum(), min=1)
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss[~trunc_mask].mean()

			else:
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss.mean()

			# orientation loss
			if self.multibin:
				orien_3D_loss = self.loss_weights['orien_loss'] * \
								Real_MultiBin_loss(preds['orien_3D'], pred_targets['orien_3D'], num_bin=self.orien_bin_size)

			# dimension loss
			dims_3D_loss = self.reg_loss_fnc(preds['dims_3D'], pred_targets['dims_3D'], reduction='none') * self.dim_weight.type_as(preds['dims_3D'])
			dims_3D_loss = self.loss_weights['dims_loss'] * dims_3D_loss.sum(dim=1).mean()

			with torch.no_grad(): pred_IoU_3D = get_iou_3d(preds['corners_3D'], pred_targets['corners_3D']).mean()

			# corner loss
			if self.compute_corner_loss:
				# N x 8 x 3
				corner_3D_loss = self.loss_weights['corner_loss'] * \
							self.reg_loss_fnc(preds['corners_3D'], pred_targets['corners_3D'], reduction='none').sum(dim=2).mean()

			if self.compute_keypoint_corner:
				# valid_mask = pred_targets['keypoints_mask'] > 0

				# N x K x 3
				keypoint_loss = self.loss_weights['keypoint_loss'] * self.keypoint_loss_fnc(preds['keypoints'],
								pred_targets['keypoints'], reduction='none').sum(dim=2) * pred_targets['keypoints_mask']
				
				keypoint_loss = keypoint_loss.sum() / torch.clamp(pred_targets['keypoints_mask'].sum(), min=1)

				if self.compute_keypoint_depth_loss:
					pred_keypoints_depth, keypoints_depth_mask = preds['keypoints_depths'], pred_targets['keypoints_depth_mask'].bool()
					target_keypoints_depth = pred_targets['depth_3D'].unsqueeze(-1).repeat(1, 3)
					
					valid_pred_keypoints_depth = pred_keypoints_depth[keypoints_depth_mask]
					invalid_pred_keypoints_depth = pred_keypoints_depth[~keypoints_depth_mask].detach()
					
					# valid and non-valid
					valid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(valid_pred_keypoints_depth, 
															target_keypoints_depth[keypoints_depth_mask], reduction='none')
					
					invalid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(invalid_pred_keypoints_depth, 
															target_keypoints_depth[~keypoints_depth_mask], reduction='none')
					
					# for logging
					log_valid_keypoint_depth_loss = valid_keypoint_depth_loss.detach().mean()

					if self.corner_with_uncertainty:
						# center depth, corner 0246 depth, corner 1357 depth
						pred_keypoint_depth_uncertainty = preds['corner_offset_uncertainty']

						valid_uncertainty = pred_keypoint_depth_uncertainty[keypoints_depth_mask]
						invalid_uncertainty = pred_keypoint_depth_uncertainty[~keypoints_depth_mask]

						valid_keypoint_depth_loss = valid_keypoint_depth_loss * torch.exp(- valid_uncertainty) + \
												self.loss_weights['keypoint_depth_loss'] * valid_uncertainty

						invalid_keypoint_depth_loss = invalid_keypoint_depth_loss * torch.exp(- invalid_uncertainty)

					# average
					valid_keypoint_depth_loss = valid_keypoint_depth_loss.sum() / torch.clamp(keypoints_depth_mask.sum(), 1)
					invalid_keypoint_depth_loss = invalid_keypoint_depth_loss.sum() / torch.clamp((~keypoints_depth_mask).sum(), 1)

					# the gradients of invalid depths are not back-propagated
					if self.modify_invalid_keypoint_depths:
						keypoint_depth_loss = valid_keypoint_depth_loss + invalid_keypoint_depth_loss
					else:
						keypoint_depth_loss = valid_keypoint_depth_loss
				
				# compute the average error for each method of depth estimation
				keypoint_MAE = (preds['keypoints_depths'] - pred_targets['depth_3D'].unsqueeze(-1)).abs() \
									/ pred_targets['depth_3D'].unsqueeze(-1)
				
				center_MAE = keypoint_MAE[:, 0].mean()
				keypoint_02_MAE = keypoint_MAE[:, 1].mean()
				keypoint_13_MAE = keypoint_MAE[:, 2].mean()

				if self.corner_with_uncertainty:
					if self.pred_direct_depth and self.depth_with_uncertainty:
						combined_depth = torch.cat((preds['depth_3D'].unsqueeze(1), preds['keypoints_depths']), dim=1)
						combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(1), preds['corner_offset_uncertainty']), dim=1).exp()
						combined_MAE = torch.cat((depth_MAE.unsqueeze(1), keypoint_MAE), dim=1)
					else:
						combined_depth = preds['keypoints_depths']
						combined_uncertainty = preds['corner_offset_uncertainty'].exp()
						combined_MAE = keypoint_MAE

					# the oracle MAE
					lower_MAE = torch.min(combined_MAE, dim=1)[0]
					# the hard ensemble
					hard_MAE = combined_MAE[torch.arange(combined_MAE.shape[0]), combined_uncertainty.argmin(dim=1)]
					# the soft ensemble
					combined_weights = 1 / combined_uncertainty
					combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
					soft_depths = torch.sum(combined_depth * combined_weights, dim=1)
					soft_MAE = (soft_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']
					# the average ensemble
					mean_depths = combined_depth.mean(dim=1)
					mean_MAE = (mean_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']

					# average
					lower_MAE, hard_MAE, soft_MAE, mean_MAE = lower_MAE.mean(), hard_MAE.mean(), soft_MAE.mean(), mean_MAE.mean()
				
					if self.compute_weighted_depth_loss:
						soft_depth_loss = self.loss_weights['weighted_avg_depth_loss'] * \
										self.reg_loss_fnc(soft_depths, pred_targets['depth_3D'], reduction='mean')
					

			depth_MAE = depth_MAE.mean()

		loss_dict = {
			'hm_loss':  hm_loss,
			'bbox_loss': reg_2D_loss if num_reg_2D > 0 else torch.tensor(0).cuda(),
			'dims_loss': dims_3D_loss if num_reg_3D > 0 else torch.tensor(0).cuda(),
			'orien_loss': orien_3D_loss if num_reg_3D > 0 else torch.tensor(0).cuda(),
		}

		log_loss_dict = {
			'2D_IoU': iou_2D.item() if num_reg_2D > 0 else torch.tensor(0).cuda(),
			'3D_IoU': pred_IoU_3D.item() if num_reg_3D > 0 else torch.tensor(0).cuda(),
		}

		MAE_dict = {}

		if self.separate_trunc_offset:
			loss_dict['offset_loss'] = offset_3D_loss if num_reg_3D > 0 else torch.tensor(0).cuda()
			loss_dict['trunc_offset_loss'] = trunc_offset_loss if num_reg_3D > 0 else torch.tensor(0).cuda()
		else:
			loss_dict['offset_loss'] = offset_3D_loss if num_reg_3D > 0 else torch.tensor(0).cuda()

		if self.compute_corner_loss:
			loss_dict['corner_loss'] = corner_3D_loss if num_reg_3D > 0 else torch.tensor(0).cuda()

		if self.pred_direct_depth:
			loss_dict['depth_loss'] = depth_3D_loss if num_reg_3D > 0 else torch.tensor(0).cuda()
			log_loss_dict['depth_loss'] = real_depth_3D_loss.item() if num_reg_3D > 0 else torch.tensor(0).cuda()
			MAE_dict['depth_MAE'] = depth_MAE.item() if num_reg_3D > 0 else torch.tensor(0).cuda()

		if self.compute_keypoint_corner:
			loss_dict['keypoint_loss'] = keypoint_loss if num_reg_3D > 0 else torch.tensor(0).cuda()

			MAE_dict.update({
				'center_MAE': center_MAE.item() if num_reg_3D > 0 else torch.tensor(0).cuda(),
				'02_MAE': keypoint_02_MAE.item() if num_reg_3D > 0 else torch.tensor(0).cuda(),
				'13_MAE': keypoint_13_MAE.item() if num_reg_3D > 0 else torch.tensor(0).cuda(),
			})

			if self.corner_with_uncertainty:
				MAE_dict.update({
					'lower_MAE': lower_MAE.item() if num_reg_3D > 0 else torch.tensor(0).cuda(),
					'hard_MAE': hard_MAE.item() if num_reg_3D > 0 else torch.tensor(0).cuda(),
					'soft_MAE': soft_MAE.item() if num_reg_3D > 0 else torch.tensor(0).cuda(),
					'mean_MAE': mean_MAE.item() if num_reg_3D > 0 else torch.tensor(0).cuda(),
				})

		if self.compute_keypoint_depth_loss:
			loss_dict['keypoint_depth_loss'] = keypoint_depth_loss if num_reg_3D > 0 else torch.tensor(0).cuda()
			log_loss_dict['keypoint_depth_loss'] = log_valid_keypoint_depth_loss.item() if num_reg_3D > 0 else torch.tensor(0).cuda()

		if self.compute_weighted_depth_loss:
			loss_dict['weighted_avg_depth_loss'] = soft_depth_loss if num_reg_3D > 0 else torch.tensor(0).cuda()

		# loss_dict ===> log_loss_dict
		for key, value in loss_dict.items():
			if value!=0 and key not in log_loss_dict:
				log_loss_dict[key] = value.item()

		# stop when the loss has NaN or Inf
		for v in loss_dict.values():
			if torch.isnan(v).sum() > 0:
				pdb.set_trace()
			if torch.isinf(v).sum() > 0:
				pdb.set_trace()

		log_loss_dict.update(MAE_dict)

		return loss_dict, log_loss_dict, pred_props, pred_target_props

def Real_MultiBin_loss(vector_ori, gt_ori, num_bin=4):
	gt_ori = gt_ori.view(-1, gt_ori.shape[-1]) # bin1 cls, bin1 offset, bin2 cls, bin2 offst

	cls_losses = 0
	reg_losses = 0
	reg_cnt = 0
	for i in range(num_bin):
		# bin cls loss
		cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2) : (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
		# regression loss
		valid_mask_i = (gt_ori[:, i] == 1)
		cls_losses += cls_ce_loss.mean()
		if valid_mask_i.sum() > 0:
			s = num_bin * 2 + i * 2
			e = s + 2
			pred_offset = F.normalize(vector_ori[valid_mask_i, s : e])
			reg_loss = F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
						F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

			reg_losses += reg_loss.sum()
			reg_cnt += valid_mask_i.sum()

	return cls_losses / num_bin + reg_losses / reg_cnt
 
