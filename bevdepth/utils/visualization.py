import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import cv2
import mmcv
import numpy as np
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import scipy
import matplotlib.cm as cm

# matplotlib.use('TkAgg')  # Or any other X11 back-end


def visualize_heatmap_on_image(image, heatmap):
    heatmap_resized = cv2.resize(heatmap, (image.shape[-1], image.shape[-2]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    return overlay

def draw_bounding_boxes(image, anno_boxes):
    for box in anno_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    return image

def plot_indices(image, inds):
    for ind in inds:
        x, y = ind
        cv2.circle(image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    return image

def apply_masks(image, masks):
    masked_image = image.copy()
    for mask in masks:
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        masked_image[mask_resized > 0.5] = [0, 0, 255]  # Highlight masked areas in red
    return masked_image

def visualize(image, heatmap, anno_boxes, inds, masks):
    # Visualize heatmap on image
    image_with_heatmap = visualize_heatmap_on_image(image, heatmap)
    
    # Draw bounding boxes
    image_with_boxes = draw_bounding_boxes(image_with_heatmap, anno_boxes)
    
    # Apply masks
    image_with_masks = apply_masks(image_with_boxes, masks)
    
    # Plot indices
    final_image = plot_indices(image_with_masks, inds)
    
    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title("Visualization with Heatmaps, Boxes, Masks, and Indices")
    plt.axis('off')
    # plt.show()
    plt.savefig("visualize.png")
    plt.close('all')

# Example usage:
# Assuming image is in BGR format (as loaded by OpenCV), heatmap is a 2D array, 
# anno_boxes is a list of [x1, y1, x2, y2] boxes, masks is a list of 2D mask arrays, 
# and inds is a list of [x, y] points.


# Example usage:
# Assuming heatmap is a numpy array of size (128, 128) with values between 0 and 1
heatmap = np.random.random((128, 128))  # Replace with your actual heatmap data

def visualize_heatmap(heatmap):
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    plt.title("Heatmap Visualization")
    plt.colorbar(label='Intensity')
    plt.savefig("heatmap_vis.png")
    plt.close('all')


def visualize_multiple_images(images_list,
    img_mean=np.array([123.675, 116.28, 103.53]),
    img_std=np.array([58.395, 57.12, 57.375]),
    to_rgb=True):
    # Clear previous plots
    plt.clf()
    plt.close()

    """
    Visualizes multiple sets of images in a 2x3 grid layout.
    
    Args:
    images_list: A list of numpy arrays, where each numpy array has a shape (6, 3, 256, 704).
                 Each array represents 6 images from different cameras.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Set images', fontsize=16)
    
    for idx, images in enumerate(images_list):
        row, col = divmod(idx, 3)  # Determine the row and column in the grid
        image = mmcv.imdenormalize(images.permute(1,2,0).numpy(),  img_mean,  img_std, to_rgb)
        image = image.clip(0, 255)
        print("HMM")
        print(image.max())
        print(image.min())
        # image = images.permute(1, 2, 0)  # Convert from (3, 256, 704) to (256, 704, 3)
        
        axes[row, col].imshow(image.astype(np.uint8))
        axes[row, col].set_title(f"Camera {idx+1}")
        axes[row, col].axis('off')  # Turn off axis
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate the title
    # plt.show()
    plt.savefig("tmp_sweeps.png")
    plt.close('all')
    

def visualize_from_image_arr(image,  # shpae = 3, 256, 704
    img_mean=np.array([123.675, 116.28, 103.53]),
    img_std=np.array([58.395, 57.12, 57.375]),
    to_rgb=True):
    
    image = mmcv.imdenormalize(image.permute(1,2,0).cpu().numpy(),  img_mean,  img_std, to_rgb)
    image = image.clip(0, 255)
    
    image = Image.fromarray(image.astype(np.uint8))
    
    return image



def visualize_3d_depth_depr(depth_map):
    # value from 0 to 1 basically

    matrix_3d = depth_map
    # Get indices of non-zero values (probabilities > 0)
    indices = np.argwhere(matrix_3d > 0)

    if np.prod(indices.shape) == 0:
        return

    # Extract x, y, z coordinates and corresponding probability values
    z_coords, y_coords, x_coords = indices[0], indices[1], indices[2]
    probabilities = matrix_3d[matrix_3d > 0]

    # Create a 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=2,
            color=probabilities,
            colorscale='Viridis',
            colorbar=dict(title='Probability'),
            opacity=0.5,
        )
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
        ),
        title='3D Visualization of Object Probabilities',
    )

    # fig.show()
    fig.write_image("tmp_draw_depth.png")


def visualize_3d_depth(depth_map, need_ratio_down=True):
    # value from 0 to 1 basically

    # Assuming 'depth_map' is your PyTorch tensor
    # depth_map = torch.randn(112, 256, 704)  # Replace with your actual data

    # Convert the tensor to a NumPy array
    if need_ratio_down:
        volume = depth_map[:, ::4, ::4].cpu().detach().numpy()
    else:
        volume = depth_map.cpu().detach().numpy()

    # Create coordinate axes
    x, y, z = np.meshgrid(
        np.arange(volume.shape[2]),
        np.arange(volume.shape[1]),
        np.arange(volume.shape[0]),
        indexing='ij'
    )

    # Flatten the arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    value_flat = np.transpose(volume, (2, 1, 0)).flatten()

    # Create a 3D volume rendering
    fig = go.Figure(data=go.Volume(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=value_flat,
        isomin=0.001,  # Set minimum threshold for visibility
        isomax=1.0,  # Maximum value in your data
        opacity=0.1,  # Adjust for better visualization
        surface_count=20,  # Number of isosurfaces
        colorscale='Viridis',
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
        )
    )

    # fig.show()
    fig.write_image("tmp_draw_depth.png")



def visualize_image_with_bboxes_and_points(img, t_bboxes=[], t_points=[], id=""):
    # Clear previous plots
    plt.clf()
    plt.close()

    # Convert PIL image to an ImageDraw object to draw shapes
    image = img.copy()
    draw = ImageDraw.Draw(image)
    
    # Draw bounding boxes
    for bbox in t_bboxes:
        x_min, y_min, x_max, y_max = bbox
        print("HMMM")
        print(x_min, y_min, x_max, y_max)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        print("HMM2")
    
    # Draw points
    for t_points_per_unit in t_points:
        for point in t_points_per_unit:
            x, y = point[:2]
            draw.ellipse((x-3, y-3, x+3, y+3), fill="blue", outline="blue")
    
    # Display the image with matplotlib
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    # plt.show()
    plt.savefig("visualization/ori_and_trans/img_{}.png".format(id))
    plt.close('all')


def visualize_heatmap_per_cam_obj(heatmap_slice, id=""):
    # Transpose to (width, height)
    heatmap_visual = heatmap_slice  # Shape: (64, 176)

    # Visualize the heatmap
    plt.figure(figsize=(10, 5))
    plt.imshow(heatmap_visual, cmap='viridis', origin='lower')  # Use 'viridis' or any other colormap
    plt.colorbar(label="Intensity")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()
    # plt.savefig("visualization/ori_and_trans/img_{}.png".format(id))
    plt.close('all')
    

# heatmap and 3D detections in monocular mode
def show_image_with_boxes_in_mono(image, output, target, visualize_preds, vis_scores=None):
	# output Tensor:
	# clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores
	image = image.numpy().astype(np.uint8)
	output = output.cpu().float().numpy()

	if vis_scores is not None:
		output[:, -1] = vis_scores.squeeze().cpu().float().numpy()
	
	# filter results with visualization threshold
	vis_thresh = cfg.TEST.VISUALIZE_THRESHOLD
	output = output[output[:, -1] > vis_thresh]
	ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}

	# predictions
	clses = output[:, 0]
	box2d = output[:, 2:6]
	dims = output[:, 6:9]
	locs = output[:, 9:12]
	rotys = output[:, 12]
	score = output[:, 13]

	proj_center = visualize_preds['proj_center'].cpu()
	keypoints = visualize_preds['keypoints'].cpu()

	# ground-truth
	calib = target.get_field('calib')
	pad_size = target.get_field('pad_size')
	valid_mask = target.get_field('reg_mask').bool()
	trunc_mask = target.get_field('trunc_mask').bool()
	num_gt = valid_mask.sum()
	gt_clses = target.get_field('cls_ids')[valid_mask]
	gt_boxes = target.get_field('gt_bboxes')[valid_mask]
	gt_locs = target.get_field('locations')[valid_mask]
	gt_dims = target.get_field('dimensions')[valid_mask]
	gt_rotys = target.get_field('rotys')[valid_mask]

	print('detections / gt objs: {} / {}'.format(box2d.shape[0], num_gt))

	pred_heatmap = visualize_preds['heat_map']
	all_heatmap = np.asarray(pred_heatmap[0, 0, ...].cpu())
	all_heatmap = cv2.resize(all_heatmap, (image.shape[1], image.shape[0]))

	img2 = Visualizer(image.copy()) # for 2d bbox
	img3 = image.copy() # for 3d bbox
	img4 = init_bev_image() # for bev

	font = cv2.FONT_HERSHEY_SIMPLEX
	pred_color = (0, 255, 0)
	gt_color = (255, 0, 0)

	# plot prediction 
	for i in range(box2d.shape[0]):
		img2.draw_box(box_coord=box2d[i], edge_color='g')
		img2.draw_text(text='{}, {:.3f}'.format(ID_TYPE_CONVERSION[clses[i]], score[i]), position=(int(box2d[i, 0]), int(box2d[i, 1])))

		corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
		corners_2d, depth = calib.project_rect_to_image(corners3d)
		img3 = draw_projected_box3d(img3, corners_2d, cls=ID_TYPE_CONVERSION[clses[i]], color=pred_color, draw_corner=False)

		corners3d_lidar = calib.project_rect_to_velo(corners3d)
		img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=pred_color, scores=None)

	# plot ground-truth
	for i in range(num_gt):
		img2.draw_box(box_coord=gt_boxes[i], edge_color='r')

		# 3d bbox template
		l, h, w = gt_dims[i]
		x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
		y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
		z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

		# rotation matirx
		roty = gt_rotys[i]
		R = np.array([[np.cos(roty), 0, np.sin(roty)],
					  [0, 1, 0],
					  [-np.sin(roty), 0, np.cos(roty)]])

		corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
		corners3d = np.dot(R, corners3d).T
		corners3d = corners3d + gt_locs[i].numpy() + np.array([0, h / 2, 0]).reshape(1, 3)

		corners_2d, depth = calib.project_rect_to_image(corners3d)
		img3 = draw_projected_box3d(img3, corners_2d, color=gt_color, draw_corner=False)

		corners3d_lidar = calib.project_rect_to_velo(corners3d)
		img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=gt_color, scores=None)

	img2 = img2.output.get_image()
	heat_mixed = img2.astype(np.float32) / 255 + all_heatmap[..., np.newaxis] * np.array([1, 0, 0]).reshape(1, 1, 3)
	img4 = cv2.resize(img4, (img3.shape[0], img3.shape[0]))
	stack_img = np.concatenate([img3, img4], axis=1)

	plt.figure(figsize=(12, 8))
	plt.subplot(211)
	plt.imshow(all_heatmap); plt.title('heatmap'); plt.axis('off')
	plt.subplot(212)
	plt.imshow(stack_img); plt.title('2D/3D boxes'); plt.axis('off')
	plt.suptitle('Detections')
	plt.show()
 
 
def show_3d_depth_map(depth_map, img, id=""):
    # img is PIL.Image object
    # Load data
    # depth_map = torch.load('depth_map.pt')  # Assume this is [112, 16, 44]
    # image = cv2.imread('image.png')  # Ensure 16x44 or resize accordingly

    # Convert to NumPy
    w, h  = img.size
    img = img.copy()
    img_array = np.ascontiguousarray(img, np.uint8)[:, :, ::-1].astype(np.uint8)  # make it suitable for opencv
    depth_map_np = depth_map.cpu().numpy()
    ratio_h, ratio_w = w / depth_map_np.shape[2], h / depth_map_np.shape[1]
    
    # Resize depth map
    resized_depth_map = np.empty((depth_map_np.shape[0], h, w), dtype=np.int8)
    for d in range(depth_map_np.shape[0]):
        layer = depth_map_np[d, :, :]
        resized_layer = np.repeat(np.repeat(layer, ratio_h, axis=0), ratio_w, axis=1)
        resized_depth_map[d, :, :] = resized_layer

    # Label connected components in 3D
    labeled_array, num_objects = scipy.ndimage.label(resized_depth_map)

    # Collect object information
    objects = []
    for label in range(1, num_objects + 1):
        voxels = np.argwhere(labeled_array == label)
        depths = voxels[:, 0]
        heights = voxels[:, 1]
        widths = voxels[:, 2]
        min_h, max_h = np.min(heights), np.max(heights)
        min_w, max_w = np.min(widths), np.max(widths)
        avg_depth = np.mean(depths)
        normalized_depth = avg_depth / (depth_map_np.shape[0]-1)
        color = (np.array(cm.viridis(normalized_depth)[:3]) * 255).astype(np.int64)
        color = tuple([int(color[0]), int(color[1]), int(color[2])])
        objects.append((min_w, min_h, max_w, max_h, color, avg_depth))
        
    
    if False and len(objects) >3 or len(objects) < 2:
        return

    # Sort by depth descending
    objects.sort(key=lambda x: x[5], reverse=True)

    # Draw bounding boxes
    for min_w, min_h, max_w, max_h, color, _ in objects:
        cv2.rectangle(img_array, (min_w, min_h), (max_w + 1, max_h + 1), color, 2)

    # Output
    cv2.imwrite('visualization/ori_and_trans/img_{}.png'.format(id), img_array)
    
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Display the image
    ax.imshow(img_array[:, : , ::-1])

    # Create ScalarMappable for the colorbar
    norm = plt.Normalize(vmin=0, vmax=112)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    # Add a smaller colorbar
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Depth')
    cbar.set_ticks([0, 28, 56, 84, 112])  # Optional: specific ticks

    # Hide axes and show plot
    # ax.axis('off')

    plt.savefig("visualization/ori_and_trans/img_{}_depthbin.png".format(id))
    plt.close('all')
    

# keypoints_2D = keypoints_2D_raw.copy()
# center_coords = center_coords_raw.copy() # before ida aug
# target_proj_center = target_proj_center_raw.copy()
# box2d  = box2d_raw.copy()
                
# visualize_image_with_bboxes_and_points(ori_img, t_bboxes=[final_coords], t_points=np.append(keypoints_2D, [center_coords], axis=0))
# visualize_image_with_bboxes_and_points(sweep_img, t_bboxes=[box2d], t_points=np.append(keypoints_2D, [center_coords], axis=0))

if __name__ == '__main__':
    # Dummy example data
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    heatmap = np.random.random((64, 64))  # Smaller than image, will be resized
    anno_boxes = [(100, 100, 200, 200), (300, 300, 400, 400)]  # Example boxes
    masks = [np.random.random((64, 64))]  # Example mask
    inds = [(150, 150), (350, 350)]  # Example indices

    # Visualize the result
    visualize(image, heatmap, anno_boxes, masks, inds)
