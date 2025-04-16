import os

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"

import math
import pickle
import hydra
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# models and dataset
from diffusers import UNet2DConditionModel
from im2flow2act.diffusion_model.uni3d.models.uni3d import create_uni3d
from im2flow2act.flow_generation.AnimateFlow3D import AnimateFlow3D
from im2flow2act.flow_generation.AnimateFlow3DPipeline import AnimationFlow3DPipeline
from im2flow2act.flow_generation.dataloader.animateflow_mimicgen_3d_dataset import AnimateFlowMimicgen3DDataset


@hydra.main(
    version_base=None,
    config_path="../../config/flow_generation",
    config_name="train_3D_flow_generation",
)
def train(cfg: DictConfig):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps
    )
    output_dir = HydraConfig.get().runtime.output_dir  # hydra will automatically assign a working directory
    ckpt_save_dir = os.path.join(output_dir, "checkpoints")
    eval_save_dir = os.path.join(output_dir, "evaluations")
    if accelerator.is_local_main_process:
        wandb.init(project=cfg.project_name)
        wandb.config.update(OmegaConf.to_container(cfg))
        accelerator.print("Logging dir", output_dir)
        os.makedirs(ckpt_save_dir, exist_ok=True)
        wandb.define_metric("epoch val loss", step_metric="val epoch")

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**cfg.noise_scheduler_kwargs)

    # Train UNet2DConditionModel from scratch
    unet = UNet2DConditionModel(in_channels=3, out_channels=3)

    weight_dtype = torch.float32

    # Load Uni3D model
    uni3d = create_uni3d(cfg.uni3d)
    uni3d_ckpt = torch.load(cfg.uni3d.ckpt_path)
    if "module" in uni3d_ckpt:
        uni3d_ckpt = uni3d_ckpt["module"]
    uni3d.load_state_dict(uni3d_ckpt)

    # AnimateFlow
    model = AnimateFlow3D(unet=unet, uni3d=uni3d)

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2),
        weight_decay=cfg.optimizer.adam_weight_decay,
        eps=cfg.optimizer.adam_epsilon,
    )
    dataset = hydra.utils.instantiate(cfg.dataset)
    if accelerator.is_local_main_process:
        if isinstance(dataset, AnimateFlowMimicgen3DDataset):
            dataset.save_normalization_parameters(ckpt_save_dir)

    # do train test split
    dataset_size = len(dataset)
    train_size = int(0.95 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=cfg.training.shuffle,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        drop_last=cfg.training.drop_last
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_size
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if cfg.training.max_train_steps is None:
        max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.training.num_train_epochs = math.ceil(
        max_train_steps / num_update_steps_per_epoch
    )
    # Train!
    total_batch_size = (
        cfg.training.batch_size
        * accelerator.num_processes
        * cfg.training.gradient_accumulation_steps
    )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(dataset)}")
    accelerator.print(f"  Num Epochs = {cfg.training.num_train_epochs}")
    accelerator.print(
        f"  Instantaneous batch size per device = {cfg.training.batch_size}"
    )
    accelerator.print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    accelerator.print(
        f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}"
    )
    accelerator.print(f"  Total optimization steps = {max_train_steps}")
    # for epoch in range(cfg.training.num_train_epochs):

    progress_bar = tqdm(total=max_train_steps, desc='Total Steps')
    for epoch in range(cfg.training.num_train_epochs):
        model.train()
        epoch_loss = []
        for step, batch in enumerate(train_dataloader):
            ### >>>> Training >>>> ###
            progress_bar.update(1)
            global_obs = batch["global_obs"].to(
                dtype=weight_dtype
            )
            first_frame_object_points = batch["first_frame_object_points"].to(
                dtype=weight_dtype
            )
            # Convert flows to latent space
            point_tracking_sequence = batch["point_tracking_sequence"].to(
                dtype=weight_dtype
            )

            # unlike im2flow2act we don't do vae to reduce the dimension
            latents = point_tracking_sequence

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )
            model_pred = model(
                noisy_latents, timesteps, global_obs, first_frame_object_points
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # logging
            epoch_loss.append(loss.item())

        if accelerator.is_local_main_process and not cfg.debug:
            wandb.log(
                {
                    "epoch loss": np.mean(epoch_loss),
                    "epoch": epoch,
                }
            )
        if epoch % cfg.training.ckpt_frequency == 0 and epoch > 0 or cfg.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                ckpt_save_path = os.path.join(ckpt_save_dir, f"epoch_{epoch}")
                os.makedirs(ckpt_save_path, exist_ok=True)
                ckpt_model = accelerator.unwrap_model(model)
                accelerator.save(
                    ckpt_model.state_dict(),
                    os.path.join(ckpt_save_path, f"epoch_{epoch}.ckpt"),
                )
                accelerator.print(f"Saved checkpoint at epoch {epoch}.")

        if epoch % cfg.evaluation.eval_frequency == 0 and epoch > 0 or cfg.debug:
            accelerator.wait_for_everyone()
            accelerator.print(f"Evaluate at epoch {epoch}.")
            if accelerator.is_local_main_process:
                eval_model = accelerator.unwrap_model(model)
                eval_model.eval()
                evaluation_save_path = os.path.join(eval_save_dir, f"epoch_{epoch}")
                os.makedirs(eval_save_dir, exist_ok=True)

                pipeline = AnimationFlow3DPipeline(
                    model=eval_model,
                    device=accelerator.device,
                    scheduler=noise_scheduler,
                    dataset=dataset
                )
                # TODO: use this pipeline to generate some evaluation result and save it
                with torch.no_grad():
                    val = next(iter(val_dataloader))

                    global_obs = val["global_obs"].to(weight_dtype)
                    first_frame_object_points = val["first_frame_object_points"].to(weight_dtype)

                    flow = pipeline(global_obs, first_frame_object_points, cfg.flow_shape)

                    results = {
                        "global_obs": global_obs,
                        "first_frame_object_points": first_frame_object_points,
                        "point_tracking_sequence": val["point_tracking_sequence"],
                        "generated_flow": flow
                    }

                    # save the resuls for visualization
                    os.makedirs(evaluation_save_path, exist_ok=True)
                    with open(f"{evaluation_save_path}/outputs.pickle" , "wb") as handle:
                        pickle.dump(results, handle, pickle.HIGHEST_PROTOCOL)

                    val_loss = F.mse_loss(flow.cpu(), val["point_tracking_sequence"], reduction="mean")
                    wandb.log(
                        {
                            "epoch val loss": val_loss,
                            "val epoch": epoch,
                        }
                    )

                # it should be fine but set the timestep back to the training timesteps
                noise_scheduler.set_timesteps(
                    cfg.noise_scheduler_kwargs.num_train_timesteps
                )
                eval_model.train()
    
    progress_bar.close()
 

if __name__ == "__main__":
    train()
