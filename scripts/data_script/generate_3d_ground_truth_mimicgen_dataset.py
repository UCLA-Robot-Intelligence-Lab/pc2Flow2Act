import open3d as o3d
import os
import numpy as np
from einops import rearrange
import h5py
from tqdm import tqdm
from matplotlib import pyplot as plt
from im2flow2act.diffusion_model.grounded_sam import *
from PIL import Image, ImageDraw


datadir = "/data/yuanhong/mimicgen_data/core_datasets/coffee/demo_src_coffee_task_D1"
dataset_path = f"{datadir}/demo_3d.hdf5"
pattern_path = f"{datadir}/coffee_pod.png"
seg_savedir = f"{datadir}/seg_images"


def segment_object_to_manipulate(image, pattern):
    """
    Input: 
        image: Image.Image
        pattern: Image.Image
    Output:
        mask: torch.tensor
    """
    detections = detect_pattern(image, pattern)
    detections = segment(image, detections)
    return detections[0].mask


def main():
    with h5py.File(dataset_path, 'r+') as f:
        for i in tqdm(range(1000)):
            depth = f[f"data/demo_{i}/obs/agentview_depth"]
            image = f[f"data/demo_{i}/obs/agentview_image"]
            traj = f[f"data/demo_{i}/datagen_info/object_poses/coffee_pod"]

            depth_np = np.array(depth)
            image_np = np.array(image)
            traj_np = np.array(traj)

            # Create the mask file with Grounded SAM
            if not os.path.exists(f'{seg_savedir}/mask_{i}.npy'):
                print(f'{seg_savedir}/mask_{i}.npy', "does not exist, creating")

                pattern = Image.open(pattern_path).convert("RGB").resize((int(49*0.182), int(47*0.182)))
                detection_image = Image.fromarray(image_np[0]) # initial frame

                # some trick to avoid mis-detecting the cup on coffee machine
                draw = ImageDraw.Draw(detection_image)
                width, height = detection_image.size
                draw.rectangle((0, 0, width // 2+3, height), fill="white")
                draw.rectangle((0, 0, width, 12), fill="white")

                mask = segment_object_to_manipulate(detection_image, pattern)
                plt.figure()
                plt.imshow(image_np[0])
                plt.imshow(mask, cmap='Reds', alpha=0.9)
                plt.savefig(f'{seg_savedir}/seg_{i}.png')
                plt.close()
                np.save(f'{seg_savedir}/mask_{i}', mask)

            # Generate trajectories
            mask = np.load(f'{seg_savedir}/mask_{i}.npy')

            fx = fy = 101.39696962
            cx = cy = 42
            width = height = 84
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            extrinsic = np.array([
                [ 0.        ,  0.70614784, -0.70806442,  0.5       ],
                [ 1.        ,  0.        ,  0.        ,  0.        ],
                [ 0.        , -0.70806442, -0.70614784,  1.35      ],
                [ 0.        ,  0.        ,  0.        ,  1.        ]
            ])

            depth_frame = np.squeeze(depth_np[0])
            rgb_frame = image_np[0]
            depth_o3d = o3d.geometry.Image(depth_frame.astype(np.float32))
            rgb_o3d = o3d.geometry.Image(rgb_frame.astype(np.uint8))

            # scene reconstruction from the initial frame
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=rgb_o3d,
                depth=depth_o3d,
                depth_scale=1.0,
                depth_trunc=10,
                convert_rgb_to_intensity=False
            )

            # object to manipulate
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                intrinsic,
                np.linalg.inv(extrinsic) # camera -> world
            )

            T = traj_np.shape[0]
            N_samples = np.sum(mask)
            sampled_points = np.asarray(pcd.points)[mask.flatten()]

            # x, y, z -> x, y, z, 1
            sampled_points = np.append(sampled_points, np.ones((N_samples, 1)), axis=1)
            flow_3d = np.zeros((T, N_samples, 4))
            flow_3d[0] = sampled_points

            sampled_points_centered = np.linalg.inv(traj_np[0]) @ sampled_points.T

            for j in range(1, T):
                flow_3d[j] = (traj_np[j] @ sampled_points_centered).T

            # Write the generated ground truth 3d-flow and point cloud to the dataset
            f.create_dataset(f"data/demo_{i}/3d_flow", data=flow_3d[:, :, :3])
            f.create_dataset(f"data/demo_{i}/pcd", data=np.asarray(pcd.points))
            f.create_dataset(f"data/demo_{i}/pcd_rgb", data=np.asarray(pcd.colors))
            f.create_dataset(f"data/demo_{i}/mask", data=mask)

if __name__ == "__main__":
    main()