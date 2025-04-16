import random
import pickle
import os
import numpy as np
import torch
import h5py
from einops import rearrange
from torchvision.transforms import v2
from scipy.interpolate import interp1d

from im2flow2act.common.imagecodecs_numcodecs import register_codecs

register_codecs()


class AnimateFlowMimicgen3DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        diff_flow=True,
        normalization=True,
        num_points_object=30, # number of points sampled on the object
        num_points_scene=84*84,
        flow_resize_length=50,
        seed=0
    ):
        """
        If diff_flow is set then the point_tracking_sequence[0] will be abs position
        and point_tracking_sequence[1:] will be delta positions
        """
        self.data_path = data_path
        self.set_seed(seed)
        self.num_points_object = num_points_object
        self.num_points_scene = num_points_scene
        self.flow_resize_length = flow_resize_length
        self.diff_flow = diff_flow
        self.normalization = normalization
        self.len_dataset = 0
        self.dataset = {}
        self.construct_dataset()

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def construct_dataset(self):
        with h5py.File(self.data_path, 'r') as dataset:
            self.len_dataset = len(dataset["data/"])

            point_tracking_sequences = np.empty(
                shape=(self.len_dataset, self.flow_resize_length, self.num_points_object, 3)
            )
            global_obs_dataset = np.empty(
                shape=(self.len_dataset, self.num_points_scene, 6)
            )
            first_frame_object_dataset = np.empty(
                shape=(self.len_dataset, self.num_points_object, 6)
            )
            for i, demo in enumerate(dataset["data/"]):
                sample = dataset[f"data/{demo}"]

                # process the point_tracking_sequence aka 3d_flow
                point_tracking_sequence = np.array(sample["3d_flow"]) # (T, num_points_object, 3)
                T, num_points_object, _ = point_tracking_sequence.shape

                # resample the tracking sequence so each has the same num of points
                indices = np.random.choice(
                    num_points_object, 
                    self.num_points_object, replace=False)
                point_tracking_sequence = point_tracking_sequence[:, indices, :]

                # interpolate the the sequence o each has the same T
                interp = interp1d(
                    np.linspace(0, 1, T), point_tracking_sequence, axis=0, kind='linear')

                # (flow_resize_len, num_points_object, 3)
                if self.diff_flow:
                    seq = interp(
                        np.linspace(0, 1, self.flow_resize_length+1))
                    point_tracking_sequence = seq[1:] - seq[:-1]

                else:
                    point_tracking_sequence = interp(
                        np.linspace(0, 1, self.flow_resize_length))

                # process the point clouds and rgb values
                pcd, color = np.array(sample["pcd"]), np.array(sample["pcd_rgb"])
                global_obs = np.concatenate([pcd, color], axis=1)  # (84 * 84, 6)

                # point cloud and rgb of the object to manipulate
                mask = np.array(sample["mask"])
                first_frame_object_points = global_obs[mask.flatten()][indices]

                global_obs_dataset[i] = global_obs
                point_tracking_sequences[i] = point_tracking_sequence
                first_frame_object_dataset[i] = first_frame_object_points

        # compute normalization statistic and perform normalization
        if self.normalization:
            self.mean = np.mean(point_tracking_sequences, axis=(0,1,2))
            self.std = np.std(point_tracking_sequences, axis=(0,1,2))
            self.transform = v2.Compose([
                v2.Normalize(self.mean, self.std)
            ])

        self.dataset["global_obs"] = torch.from_numpy(global_obs_dataset)
        self.dataset["point_tracking_sequence"] = torch.from_numpy(point_tracking_sequences)
        self.dataset["first_frame_object_points"] = torch.from_numpy(first_frame_object_dataset)

        print(f"Data loaded from {self.data_path}")

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        global_obs = self.dataset["global_obs"][idx]
        point_tracking_sequence = self.dataset["point_tracking_sequence"][idx]
        first_frame_object_points = self.dataset["first_frame_object_points"][idx]

        point_tracking_sequence = point_tracking_sequence.permute(2, 0, 1) # (3, flow_resize, num_points)

        if self.normalization:
            point_tracking_sequence = self.transform(point_tracking_sequence)

        return {
            "global_obs": global_obs,
            "point_tracking_sequence": point_tracking_sequence,
            "first_frame_object_points": first_frame_object_points
        }

    def save_normalization_parameters(self, save_path):
        normalization_params = {}
        if self.normalization:
            normalization_params = {
                "mean": self.mean,
                "std": self.std,
            }

        os.makedirs(save_path, exist_ok=True)
        save_dir = f"{save_path}/dataset_normalization_params.pickle"
        with open(save_dir, "wb") as handle:
            pickle.dump(normalization_params, handle, pickle.HIGHEST_PROTOCOL)

        print(f"normalization parameters saved to {save_dir}")

