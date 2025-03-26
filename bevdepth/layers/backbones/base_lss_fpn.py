# Copyright (c) Megvii Inc. All rights reserved.
import torch
import numpy as np
import random
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

try:
    from bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference
    from bevdepth.ops.voxel_pooling_train import voxel_pooling_train
except ImportError:
    print('Import VoxelPooling fail.')

from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box, LidarPointCloud
from scipy.interpolate import griddata
from bevdepth.utils.visualization import show_3d_depth_map, visualize_from_image_arr


from pyquaternion import Quaternion

from bevdepth.datasets.nusc_det_dataset import post_process_coords


__all__ = ['BaseLSSFPN']


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConfMapNet2D(nn.Module):
    def __init__(self):
        super(ConfMapNet2D, self).__init__()
        # Define a simple sequence of 3D convolutional layers
        self.network = nn.Sequential(
            nn.Conv2d(112, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 112, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def forward(self, x):
        # x has shape (batch_size, D=112, H=16, W=44)
        x = x.float()
        out = self.network(x)  # Output shape: (batch_size, 112, 16, 44)
        return out



class DepthNet(nn.Module):

    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        4).repeat(1, 1, num_cams, 1, 1)
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)

        return torch.cat([depth, context], dim=1)


class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class BaseLSSFPN(nn.Module):

    def __init__(self,
                 x_bound,
                 y_bound,
                 z_bound,
                 d_bound,
                 final_dim,
                 downsample_factor,
                 output_channels,
                 img_backbone_conf,
                 img_neck_conf,
                 depth_net_conf,
                 use_da=False,
                 depth_refinement_cfgs={},
                 is_monocular=True
                 ):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(BaseLSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.depth_net = self._configure_depth_net(depth_net_conf)

        self.conf_map_net = None
        if is_monocular and depth_refinement_cfgs['depth_blend_weight_mode'] == 'learned':
            self.conf_map_net = self._configure_confidence_map_net()
            pass

        self.img_neck.init_weights()
        self.img_backbone.init_weights()
        self.use_da = use_da
        if self.use_da:
            self.depth_aggregation_net = self._configure_depth_aggregation_net(
            )
        
        self.depth_fused_mode = depth_refinement_cfgs['depth_fused_mode']
        self.depth_blend_weight_mode = depth_refinement_cfgs['depth_blend_weight_mode']
        self.monoflex_depth_mode = depth_refinement_cfgs['monoflex_depth_mode']


    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )
        
    def _configure_confidence_map_net(self):
        return ConfMapNet2D()

    def _configure_depth_aggregation_net(self):
        """build pixel cloud feature extractor"""
        return DepthAggregation(self.output_channels, self.output_channels,
                                self.output_channels)

    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(
                0, 3, 1, 4,
                2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (
                self.depth_aggregation_net(img_feat_with_depth).view(
                    n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous())
        return img_feat_with_depth

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim  # 256, 704

        # downsample_factor = 16
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor  # 16 , 44
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)

        # ida mat shape: batch, 6 * num_sweeps, 4, 4
        points = torch.inverse(ida_mat.cpu()).cuda().matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat.cpu()).cuda())
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _forward_single_sweep(self,
                              img_feats,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              monoflex_props=None,
                              sweep_cam2ego_per_cam_rotations=None,
                              sweep_intrins=None,
                              is_return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
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
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        # img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            mats_dict
        )

        depth = depth_feature[:, :self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype)  # B, C (depth_channel + output_channel), H, W

        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        
        depth_monoflex = None
        depth_fused = depth.clone()
        B, D, H, W = depth_fused.shape

        if monoflex_props:
            depth_monoflex= self.get_depth_probs_monoflex(
                monoflex_props, depth
            )
            # depth_monoflex = depth_monoflex.softmax(dim=1, dtype=depth_feature.dtype)
            
            if False and is_return_depth: # for visualization
                batch_vis, _, c_vis, channel_vis, h_vis, w_vis = sweep_imgs.shape
                vis_sweep_img = sweep_imgs[:, 0, ...].reshape(-1, *list(sweep_imgs.shape[-3:]))
                vis_source_features = source_features.reshape(-1, *list(source_features.shape[-3:]))

                rand_batch = random.randint(0, batch_vis-1)
                for id_cam_vis in range(c_vis): # total camera of 6
                    id_referred = rand_batch * c_vis + id_cam_vis
                    img_vis = visualize_from_image_arr(vis_sweep_img[id_referred])
                
                    # show_3d_depth_map(depth_monoflex[id_referred], vis_source_features[id_referred], id="b{}-cam{}-depth_img_feats".format(rand_batch, id_cam_vis))
                    show_3d_depth_map(
                        depth_monoflex[id_referred],img_vis, id="b{}-cam{}-depth_sweeps".format(rand_batch, id_cam_vis)
                    )
               

            if self.depth_fused_mode == 'direct_replacement':
                mask = depth_monoflex > 0
                depth_fused[mask] = depth_monoflex[mask]

            elif self.depth_fused_mode in ['weighted_fusion', 'roi_refinement']:
                
                if self.depth_blend_weight_mode == 'fixed':
                    conf_map = torch.full((B, D, H, W), 0.9) 
                elif self.depth_blend_weight_mode == 'geometry':
                    # based on distance
                    depth_dim = depth_fused.shape[1]
                    conf_map = 1/ (1+ torch.arange(depth_dim).float())
                    conf_map = conf_map.unsqueeze(1).view(1, -1, 1, 1).repeat(B, 1, H, W)
                elif self.depth_blend_weight_mode == 'learned':
                    conf_map = self.conf_map_net(depth_monoflex) if self.depth_fused_mode == 'roi_refinement' else self.conf_map_net(depth_monoflex + depth_fused)
                else:
                    raise NotImplementedError(
                        "Not Implemented yet: {}".format(
                            self.depth_blend_weight_mode 
                        )
                    )
                    
                conf_map = conf_map.cuda().half()
                
                if self.depth_fused_mode == 'weighted_fusion':
                    depth_fused = conf_map * depth_monoflex + (1-conf_map) * depth

                elif self.depth_fused_mode == 'roi_refinement':
                    mask = depth_monoflex > 0                    
                    depth_fused[mask] = conf_map[mask] * depth_monoflex[mask] + (1-conf_map)[mask] * depth[mask]


        if self.training or self.use_da:
            img_feat_with_depth = depth_fused.unsqueeze(
                1) * depth_feature[:, self.depth_channels:(
                    self.depth_channels + self.output_channels)].unsqueeze(2)

            if depth_monoflex is not None and self.depth_fused_mode == 'hard_combine': 
                img_feat_with_depth_monoflex = depth_monoflex.unsqueeze(
                    1) * depth_feature[:, self.depth_channels:(
                        self.depth_channels + self.output_channels)].unsqueeze(2)
                img_feat_with_depth += img_feat_with_depth_monoflex
    
            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

            img_feat_with_depth = img_feat_with_depth.reshape(
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )  # [b, n, c, d, h, w]

            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
            # [b, n, d, h, w, c]

            # TODO: check if geom_xyz is the info about ego vechicle frame points in each sweep batch and camera 
            feature_map = voxel_pooling_train(geom_xyz,
                                              img_feat_with_depth.contiguous(),
                                              self.voxel_num.cuda())

        else:
            feature_map = voxel_pooling_inference(
                geom_xyz, depth_fused, depth_feature[:, self.depth_channels:(
                    self.depth_channels + self.output_channels)].to(torch.float32).contiguous(),
                self.voxel_num.cuda())

        if is_return_depth:
            # final_depth has to be fp32, otherwise the depth
            # loss will colapse during the traing process.
            return feature_map.contiguous(
            ), depth_feature[:, :self.depth_channels].softmax(dim=1)
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                monoflex_props=None,
                sweep_intrins = None,
                is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
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
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        
        key_img_feats = self.get_cam_feats(sweep_imgs[:, 0:1, ...])

        key_frame_res = self._forward_single_sweep(
            key_img_feats,
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            monoflex_props=monoflex_props,
            sweep_intrins=sweep_intrins,
            is_return_depth=is_return_depth)
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                img_feats = self.get_cam_feats(sweep_imgs[:, sweep_index:sweep_index + 1, ...])
                feature_map = self._forward_single_sweep(
                    img_feats,
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    monoflex_props=monoflex_props,
                    is_return_depth=False)
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)


    def get_depth_probs_monoflex(
            self, monoflex_props, ref_depth
        ):
        # cs_rec is calibrated_sensor info from nuscenes
        # the rotation info for Box object is gained from vehicle ego coor's
        heatmap_shape = monoflex_props['heatmap_shape']
        ref_depth_shape = ref_depth.shape[-2:]

        cat_3D = monoflex_props['cat_3D']  # the actual cam coors
        bboxes_2D = monoflex_props['bboxes_2d_mini']
        confs_score = monoflex_props['confs_score']
        batch_idxs = monoflex_props['batch_idxs']

        h_ratio = ref_depth_shape[-2] / heatmap_shape[-2]  # 16/64 = 0.25
        w_ratio = ref_depth_shape[-1] / heatmap_shape[-1]  # 44/176 = 0.25
        ratios = torch.tensor([w_ratio, h_ratio]).to(bboxes_2D.device)
        
        if bboxes_2D.numel() == 0:
            return torch.zeros_like(ref_depth)

        bboxes_2D = bboxes_2D * ratios.repeat(1, 2).repeat(bboxes_2D.shape[0], 1)

        # Step:
        # change to key ego coor
        # apply BDA mat

        num_objs, _ = cat_3D.shape

        _, d_channel, d_h, d_w = ref_depth.shape

        pred_locations_3D = cat_3D[..., :3]

        map_depths_3D = torch.zeros_like(ref_depth)

        for idx_point in range(0, num_objs):
            center_3d = pred_locations_3D[idx_point]
            bbox_2d = bboxes_2D[idx_point]
            conf_score = confs_score[idx_point]
            batch_idx = batch_idxs[idx_point]
            point_depth = self.get_obj_lvl_depth(center_3d, bbox_2d)  # after resize ratio & before downsample ratio
            depth_map_3d = depth_transform_3dmap(
                ref_depth[batch_idx], 
                point_depth,
                self.depth_channels,
                self.monoflex_depth_mode,
                conf_score,
                dbound=self.d_bound,
            )

            tmp_mask = map_depths_3D[batch_idx] == 0
            
            map_depths_3D[batch_idx] = torch.where(tmp_mask, depth_map_3d, map_depths_3D[batch_idx])


        return map_depths_3D


        """
        for point in points:

            b_idx = batch_indices[idx]
            c_idx = cam_indices[idx]
            d_idx = depth_indices[idx]
            h_idx = h_indices[idx]
            w_idx = w_indices[idx]
            
            # Assign probability
            depth_probabilities[b_idx, c_idx, d_idx, h_idx, w_idx] = 1.0
        

        for idx_point in range(0, num_points):
            idx_batch = int(idx_point / num_sweeps / num_cams)
            idx_cam = int(idx_point % num_cams)

            assert (idx_point - idx_batch * num_sweeps * num_cams) >= 0
    
            idx_sweep = int((idx_point - idx_batch * num_sweeps * num_cams) / num_cams)

            cam2ego_per_cam_rotation = sweep_cam2ego_per_cam_rotations[idx_batch][idx_sweep][idx_cam]

            target_location_3D = pred_locations_3D[idx_point] 
            target_dimension_3D = target_dimensions_3D[idx_point] 
            target_roty_3D = target_rotys_3D[idx_point] 

            # Get Sensor Calibration Data
            rotation_cam_to_ego = Quaternion(cam2ego_per_cam_rotation)

            # Transform the Rotation Quaternion to Ego Vehicle Coordinates
            rotation_cam = Quaternion(axis=[0, 1, 0], angle=target_roty_3D[idx_point])
            rotation_ego = rotation_cam_to_ego * rotation_cam

            if target_location_3D[-1] <= min_depth_dist:
                continue
        
            box = Box(
                target_location_3D,
                target_dimension_3D,
                Quaternion(axis=[0, 1, 0], angle=rotation_ego),
                # no velocity
            )

            corners_3d = box.corners()
            centers_3d = box.center
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            corners_3d_front = corners_3d[:, in_front]

            intrins_per_cam = sweep_intrins[idx_batch][idx_cam][idx_sweep]

            # Project 3d box to 2d.
            corner_coors_depth = view_points(corners_3d, intrins_per_cam, True).T
            corner_coords = np.array(corner_coors_depth[:, :2])

            corner_coords_front_depth = view_points(corners_3d_front, intrins_per_cam, True).T
            corner_coords_front = np.array(corner_coords_front_depth[:, :2])

            center_coords_depth = view_points(np.array([centers_3d]).T, intrins_per_cam, True).T
            center_coords = np.array(center_coords_depth[:, :2])[0]


            # Keep only corners that fall within the image.
            final_coords = post_process_coords(corner_coords_front, imsize=(self.final_dim[1], self.final_dim[0]))

            # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
            if final_coords is None or centers_3d[2] <= 0:
                assert centers_3d[2] > 0
            else:
                min_x, min_y, max_x, max_y = final_coords
            
            final_coords = np.array(final_coords)

            # min_x and min_y valid range = final
            if min_x < 0 or min_x >= ogfW or min_y < 0 or min_y >= ogfH:
                continue
            
            min_x //= self.downsample_factor
            max_x //= self.downsample_factor
            min_y //= self.downsample_factor
            max_y //= self.downsample_factor


            # TODO: correct the coors

            map_depths[min_x:max_x, min_y:max_y] = target_location_3D[-1]

             Complex interpolation bye bye

            # Create a grid of pixel coordinates
            width = self.final_dim[1]
            height = self.final_dim[0]
            grid_u, grid_v = np.meshgrid(np.arange(width), np.arange(height))


            # Flatten the grid and mask
            grid_points = np.vstack((grid_u.ravel(), grid_v.ravel())).T

            # Known points and their depths
            u_coords = np.concatenate(center_coords, corner_coords)[:, 0]
            v_coords = np.concatenate(center_coords, corner_coords)[:, 1]
            depths =  np.concatenate(center_coords, corner_coords)[:, 2]
            points = np.vstack((u_coords, v_coords)).T
            values = depths

            # Interpolate over the grid
            grid_depths = griddata(points, values, (grid_u, grid_v), method='linear')

            # Handle NaNs outside convex hull
            grid_depths[np.isnan(grid_depths)] = 0

            """

    def get_obj_lvl_depth(self, centers_3d, box2d):
        # get the coordinates of all pixels in bbox
        min_x, min_y, max_x, max_y = box2d
        min_x, min_y, max_x, max_y = min_x.int(), min_y.int(), (max_x+1).int(), (max_y+1).int() 
        depth = centers_3d[-1]

        x_size = max_x-min_x
        y_size = max_y-min_y
        x_coords = torch.arange(min_x, max_x, 1, dtype=torch.int).view(1, x_size).expand(y_size, x_size).cuda()
        y_coords = torch.arange(min_y, max_y, 1, dtype=torch.int).view(y_size, 1).expand(y_size, x_size).cuda()
        depth_coords = torch.tensor(depth).unsqueeze(-1).expand(y_size, x_size)

        # n_points x 3
        grid = torch.stack((x_coords, y_coords, depth_coords), dim=-1)
        return grid.reshape(-1, 3).cpu().numpy()



def depth_transform_3dmap(ref_depth_map, point_depth, depth_channel, monoflex_depth_mode, conf_score, dbound):
    """Transform depth based on ida augmentation configuration.

    Args:
        ref_depth_map D * H * W , value consist of 0 or 1
        point_depth n_points x 3 : x, y, depth

    Returns:
        np array: [d, h/down_ratio, w/down_ratio]
    """

    D, H, W = ref_depth_map.shape
    point_depth = torch.from_numpy(point_depth).cuda()

    depth_in_map = (point_depth[:, -1] - (dbound[0] - dbound[2])) / dbound[2]

    valid_mask = ((point_depth[:, 1] < H)
                  & (point_depth[:, 0] < W)
                  & (point_depth[:, 1] >= 0)
                  & (point_depth[:, 0] >= 0)
                   & (depth_in_map.long() >= 1)  # eliminate the points that are too close to the camera
                  & (depth_in_map.long() < depth_channel)
                  )
    point_depth = point_depth[valid_mask]
    depth_in_map = depth_in_map[valid_mask]

    # add depth dimension
    # input H * W
    # return D * H * W

    depth_map_3d = torch.zeros_like(ref_depth_map)

    h_indices = point_depth[:, 1].long()
    w_indices = point_depth[:, 0].long()
    d_indices = (depth_in_map-1).long()

    if monoflex_depth_mode == 'fixed':
        depth_map_3d[d_indices, h_indices, w_indices] = conf_score.half().cuda()
    elif monoflex_depth_mode == 'gaussian':
        raise NotImplementedError(
            "Not Implemented yet: {}".format(
                monoflex_depth_mode
            )
        )
    else:
        raise NotImplementedError(
            "Not Implemented yet: {}".format(
                monoflex_depth_mode
            )
        )

    return depth_map_3d

