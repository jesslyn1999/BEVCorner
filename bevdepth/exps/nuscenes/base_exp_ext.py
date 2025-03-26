H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)


depth_refinement_cfgs = {
    'monoflex_depth_mode': 'fixed',  # either ['fixed', 'gaussian']
    'depth_fused_mode': 'weighted_fusion',  # either ['direct_replacement', 'weighted_fusion', 'roi_refinement', 'hard_combine']
    'depth_blend_weight_mode': 'fixed',  # either ['fixed', 'geometry', 'learned', '2d_loss']
}


backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'z_bound': [-5, 3, 8],
    # 'd_bound': [2.0, 58.0, 0.5],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512),
    'depth_refinement_cfgs': depth_refinement_cfgs
    
}

monoflex_backbone_conf={
    'conv_body': 'dla34',
    'pretrain': True,
    'down_ratio': 4
}

ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim':
    final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}

bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))
out_size_factor=4
down_ratio=4
grid_size=[512, 512, 1]

filter_annos=[0.9, 20]

data_root='data/nuscenes'
# data_root='data/nuscenes_mini'

# data_root_version='v1.0-mini'
data_root_version='v1.0-trainval'

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=out_size_factor,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=grid_size,
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=out_size_factor,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=out_size_factor,
    voxel_size=[0.2, 0.2, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}

regression_heads = [['2d_dim'], ['3d_offset'], ['corner_offset'], ['corner_uncertainty'], ['3d_dim'], ['ori_cls', 'ori_offset'], ['depth'], ['depth_uncertainty']]
regression_channels = [[4, ], [2, ], [20], [3], [3, ], [8, 8], [1, ], [1, ]]

orientation = 'multi-bin'
orientation_bin_size = 4
heatmap_ratio = 0.5

detector_predictor_conf = {
    'classes': CLASSES,
    'regression_heads': regression_heads,
    'regression_channels': regression_channels,
    'width_train': W,
    'height_train': H,
    'down_ratio': down_ratio,
    'num_channel': 256,
    'use_normalization': 'BN',
    'inplace_abn': True,
    'bn_momentum': 0.1,
    'init_p': 0.01,
    'uncertainty_init': True,
    'enable_edge_fusion': True,
    'edge_fusion_kernel_size': 3,
    'edge_fusion_relu': False,
    'edge_fusion_norm': 'BN',
}

anno_encoder_conf = {
    'model_device': 'cuda',
    'detect_classes': CLASSES,
    'min_radius': 0.0,
    'max_radius': 0.0,
    'center_radius_ratio': 0.1,
    'heatmap_center': '3D',
    'center_mode': 'max',
    'depth_mode': 'inv_sigmoid',
    'depth_range': [0.1, 100],
    'depth_reference': (26.494627, 16.05988),

    # Reference car size in (length, height, width)
    'dimension_mean': ((4.61866713, 1.73140302, 1.95445654), # Car
                    (6.93423535, 2.84272183, 2.50764965), # Truck
                    (6.37012576, 3.18798194, 2.8497035), # Construction Vehicle
                    (11.07509362,  3.46629955,  2.93204785), # Bus
                    (12.2858644 ,  3.86945644,  2.90172486), # Trailer
                    (0.50201487, 0.98339635, 2.5262179 ), # Barrier
                    (2.10525196, 1.47093009, 0.77147056), # Motorcycle
                    (1.70221924, 1.28309874, 0.59806316), # Bicycle
                    (0.72468841, 1.76806106, 0.66655042), # Pedestrian
                    (0.41283657, 1.06687546, 0.4069071)), # Traffic Cone

    'dimension_std': ((0.46032386, 0.24328848, 0.18679643), # Car
                (2.16968319, 0.84065833, 0.44992966), # Truck
                (3.13320673, 1.01716375, 1.0557826 ), # Construction
                (2.0754583 , 0.49955404, 0.32333505), # Bus
                (4.52220519, 0.75393834, 0.53464937), # Trailer
                (0.16676612, 0.1506985 , 0.64043416), # Barrier
                (0.32231226, 0.23121374, 0.1706047 ), # Motorcycle
                (0.25608074, 0.34426514, 0.16458265), # Bicycle
                (0.18741625, 0.18945536, 0.13819434), # Pedestrian
                (0.14222394, 0.26909892, 0.13225061)), # Traffic Cone

    'dimension_reg': ['exp', True, False],
    'orientation': orientation,
    'orientation_bin_size': orientation_bin_size,
    'regression_offset_stat': [-0.5844396972302358, 9.075032501413093],
    'height_train': H,
    'width_train': W,
    'down_ratio': down_ratio,
}

detector_loss_conf = {
    'anno_encoder_conf': anno_encoder_conf,
    'regression_heads': regression_heads,
    'regression_channels': regression_channels,
    'max_objs': 40,
    'center_sample': 'center',
    'regress_area': False,
    'heatmap_type': 'centernet',
    'supervise_corner_depth': False,
    'loss_names': ['hm_loss', 'bbox_loss', 'depth_loss', 'offset_loss', 
                   'orien_loss', 'dims_loss', 'corner_loss', 'keypoint_loss', 
                   'keypoint_depth_loss', 'trunc_offset_loss', 'weighted_avg_depth_loss'],
    'dimension_weight': [1, 1, 1],
    'uncertainty_range': [-10, 10],
    'loss_type': ["Penalty_Reduced_FocalLoss", "L1", "giou", "L1"],
    'loss_penalty_alpha': 2,
    'loss_beta': 4,
    'orientation': orientation,
    'orientation_bin_size': orientation_bin_size,
    'truncation_offset_loss': 'log',
    'init_loss_weight': [1, 1, 1, 0.5, 1, 1, 0.2, 1.0, 0.2, 0.1, 0.2],
    'uncertainty_weight': 1.0,
    'keypoint_xy_weights': [1, 1],
    'keypoint_norm_factor': 1.0,
    'modify_invalid_keypoint_depths': False,
    'corner_loss_depth': 'soft_combine'
}

detector_post_processor_conf = {
    'anno_encoder_cfg': anno_encoder_conf,
    'key2channel_cfg':{
        'keys': regression_heads,
        'channels': regression_channels
    },
    'detections_threshold': 0.2,
    'detecions_per_img': 50,
    'eval_dis_ious': False,
    'eval_depth': False,
    'width_train': W,
    'height_train': H,
    'down_ratio': down_ratio,
    'output_depth': 'soft',
    'pred_2d': True,
    'uncertainty_as_confidence': True,
}

NORM_NUM_GROUPS = 32
