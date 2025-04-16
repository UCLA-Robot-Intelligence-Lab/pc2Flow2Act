import numpy as np
import torch
from torch import nn
# from transformers import CLIPVisionModel

from im2flow2act.diffusion_model.pointnet import PointNetEncoder, MLPEncoder
# from im2flow2act.common.utility.model import freeze_model


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class AnimateFlow3D(nn.Module):
    def __init__(
        self,
        unet,
        uni3d,
        in_channels=6, # (x, y, z, R, G, B)
        flow_shape=(200, 30), # (T, num_points)
    ) -> None:
        super().__init__()
        self.scene_encoder = uni3d
        self.obj_encoder = MLPEncoder(in_channels=in_channels)
        self.unet = unet
        self.flow_shape = flow_shape

        for param in self.scene_encoder.parameters():
            param.requires_grad = False
        
        self.scene_encoder.eval()


    def forward(
        self,
        noisy_latents,
        timesteps,
        global_obs,
        first_frame_object_points,
    ):
        global_condition = self.scene_encoder.encode_pc(global_obs)
        global_condition = global_condition / global_condition.norm(dim=-1, keepdim=True)
        local_condition = self.obj_encoder(first_frame_object_points)
        # concat with text feature
        encoder_hidden_states = torch.cat(
            [
                global_condition,
                local_condition,
            ],
            axis=1,
        )  # (B, 1024 + 256)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1) # (B, seq_len, 1024+256)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return model_pred

    def load_model(self, path):
        print("Loading complete model...")
        self.load_state_dict(torch.load(path))
        print(">> loaded model")
