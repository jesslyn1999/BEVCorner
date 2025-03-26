import numpy as np
from nuscenes.nuscenes import NuScenes
from bevdepth.datasets.nusc_det_dataset import \
    map_name_from_general_to_detection
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')

# Example: Compute std for "car"
dims_dict = {
    'car': [],
    'truck': [],
    'construction_vehicle': [],
    'bus': [],
    'trailer': [],
    'barrier': [],  
    'motorcycle': [],
    'bicycle': [],
    'pedestrian': [],
    'traffic_cone': [],
    'ignore': [],
}
for ann in nusc.sample_annotation:
    # Convert from NuScenes [L, W, H] to your LHW order [L, H, W]
    w, l, h = ann['size']
    dims_dict[map_name_from_general_to_detection[ann['category_name']]].append([l, h, w])  # LHW order

std_out = []
mean_out = []
for cls, dims in dims_dict.items():
    dims = np.array(dims)
    std_out.append(np.std(dims, axis=0))
    mean_out.append(np.mean(dims, axis=0))
    print(f"{cls} std (LHW):", np.std(dims, axis=0))
    print(f"{cls} mean (LHW):", np.mean(dims, axis=0))

print(std_out)
print()
print(mean_out)
