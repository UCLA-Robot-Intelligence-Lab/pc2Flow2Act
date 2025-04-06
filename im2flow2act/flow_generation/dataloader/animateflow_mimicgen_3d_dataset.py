import random

import numpy as np
import torch
import h5py
from einops import rearrange, repeat
from torchvision.transforms import v2
from tqdm import tqdm
from scipy.interpolate import interp1d

from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.tapnet.utility.utility import get_buffer_size

register_codecs()


class AnimateFlowMimicgen3DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        # grid_size=32,
        # frame_resize_shape=[224, 224],
        # point_tracking_img_size=[256, 256],
        diff_flow=True,
        # max_episode=None,
        # start_episode=0,
        num_points=30,
        flow_resize_length=50,
        seed=0
    ):
        self.data_path = data_path
        self.set_seed(seed)
        self.train_data = []
        self.num_points = num_points
        self.flow_resize_length = flow_resize_length
        self.diff_flow = diff_flow
        self.construct_dataset()

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def construct_dataset(self):
        with h5py.File(self.data_path, 'r') as dataset:
            for demo in iter(dataset["data/"]):
                sample = dataset[f"data/{demo}"]

                # process the point_tracking_sequence aka 3d_flow
                point_tracking_sequence = np.array(sample["3d_flow"]) # (T, num_points, 3)
                T, num_points, _ = point_tracking_sequence.shape
                
                # resample the tracking sequence so each has the same num of points
                indices = np.random.choice(
                    num_points, 
                    self.num_points, replace=False)
                point_tracking_sequence = point_tracking_sequence[:, indices, :]

                # interpolate the the sequence o each has the same T
                interp = interp1d(
                    np.linspace(0, 1, T), point_tracking_sequence, axis=0, kind='linear')
                
                # (flow_resize_len, self.num_points, 3)
                if self.diff_flow:
                    seq = interp(
                        np.linspace(0, 1, self.flow_resize_length+1))
                    
                    point_tracking_sequence = seq[1:,:,:] - seq[:-1,:,:]
                else:
                    point_tracking_sequence = interp(
                        np.linspace(0, 1, self.flow_resize_length))
                
                # process the point clouds and rgb values
                pcd, color = np.array(sample["pcd"]), np.array(sample["pcd_rgb"])
                global_obs = np.concatenate([pcd, color], axis=1)  # (84 * 84, 6)

                # point cloud and rgb of the object to manipulate
                mask = np.array(sample["mask"])
                first_frame_object_points = global_obs[mask.flatten()][indices]

                # (3, flow_resize_len, self.num_points)
                point_tracking_sequence = rearrange(point_tracking_sequence, "... c -> c ...")

                self.train_data.append({
                    "global_obs": torch.from_numpy(global_obs),
                    "point_tracking_sequence": torch.from_numpy(point_tracking_sequence),
                    "first_frame_object_points": torch.from_numpy(first_frame_object_points)
                })
        
        print(f"Data loaded from {self.data_path}")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]
