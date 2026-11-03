# -*- coding: utf-8 -*-
"""Train a Canny ControlNet with a UNet LoRA adapter and metric conditioning.

The script fine-tunes ControlNet, adds LoRA adapters to the UNet, projects
numeric metrics into cross-attention, and uses gradient checkpointing to reduce
memory use.

Example:
    python canny_controlnet_train_lora.py --data_root data/control_generation --base_model runwayml/stable-diffusion-v1-5 --unet_path out/treatment_models_weight_choose/unet_3000.pth --controlnet_canny_path lllyasviel/control_v11p_sd15_canny --output_dir out/canny_unet_lora_models_weight_choose --validation_image_dir data/test_image --lr 1e-4 --batch_size 1 --max_steps 6000 --save_every 500 --validation_steps 25000 --gradient_accumulation_steps 4 --amp
"""

import os
import random
import sys
try:
    original_stderr = sys.stderr
    with open(os.devnull, 'w') as f:
        sys.stderr = f
        import xformers
        import xformers.ops
    sys.stderr = original_stderr
except Exception:
    pass

import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Diffusers & Transformers
from diffusers import (
    AutoencoderKL, DDPMScheduler, UNet2DConditionModel, 
    ControlNetModel
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

# PEFT (LoRA)
try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    print("错误: 未安装 PEFT 库。请运行: pip install peft")
    exit(1)

# Dataset imports
try:
    from datasets_cepha import ControlNetPairDataset
    from past_try.old.canny_controlnet_validator import TrainingValidator 
except ImportError as e:
    print("="*50)
    print(f"错误: 无法导入项目模块: {e}")
    print("请确保 datasets_cepha.py 和 canny_controlnet_validator.py 在当前目录下")
    # exit(1)

# --------------------- Core component: Metric Projector ---------------------
class MetricProjector(nn.Module):
    def __init__(self, input_dim=36, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(), 
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        x = self.net(x) 
        return x.unsqueeze(1)

# --------------------- Utility functions ---------------------
def seed_all(s: int = 42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# Public source snapshot: do not embed patient identifiers. Supply exclusions only
# in a private local configuration when working with authorized data.
DEFAULT_EXCLUDE_PATIENTS = ""

def parse_patient_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [name.strip() for name in value.replace("，", ",").split(",") if name.strip()]

# --------------------- Training configuration ---------------------
@dataclass
class TrainConfig:
    data_root: str
    base_model: str
    output_dir: str
    unet_path: str 
    
    controlnet_canny_path: str = "lllyasviel/control_v11p_sd15_canny"
    
    # LoRA learning rates are often higher than full-finetuning rates (1e-4 is a typical value).
    lr: float = 1e-4 
    batch_size: int = 1
    max_steps: int = 20000
    seed: int = 42
    size: int = 512
    use_xformers: bool = True
    use_amp: bool = True

    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Dataset parameters
    rotation_csv: Optional[str] = None
    pre_csv: Optional[str] = None
    post_csv: Optional[str] = None
    information_csv: Optional[str] = None
    movement_csv: str = "data/movement.csv"
    exclude_patients: Optional[List[str]] = None
    
    crop_margin: float = 0.10
    target_mm_per_px: float = 0.1
    ruler_mm: float = 10.0
    
    # Image params
    canny_low: int = 25
    canny_high: int = 40
    mask_dilate_kernel_size: int = 15
    min_contour_area: int = 15
    heatmap_sigma: float = 8.0 

    # Validation and checkpointing
    validation_image_dir: str = "data/test_image"
    landmark_weights: str = "model/best_model.pth"
    save_every: int = 5000
    validation_steps: int = 1000
    gradient_accumulation_steps: int = 4

# --------------------- Trainer ---------------------
class CannyControlNetLoraTrainer:
    def __init__(self, cfg: TrainConfig, dataset: Dataset):
        self.cfg = cfg
        self.ds = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Model] Loading components from {cfg.base_model}...")

        # --- 1. Load components ---
        self.vae = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.base_model, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")

        # --- 2. Load UNet ---
        print(f"[UNet] Loading UNet architecture...")
        self.unet = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet")
        
        # Load previously trained weights.
        if cfg.unet_path and os.path.exists(cfg.unet_path):
            print(f"[UNet] Loading pretrained weights from {cfg.unet_path}")
            try:
                state_dict = torch.load(cfg.unet_path, map_location="cpu")
                if "state_dict" in state_dict: # Support checkpoints wrapped in a state_dict field.
                    state_dict = state_dict["state_dict"]
                self.unet.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"[Warning] Error loading UNet weights: {e}")
        
        self.unet.to(self.device)

        # --- 3. Configure LoRA ---
        print(f"[LoRA] Adding LoRA adapters to UNet (Rank={cfg.lora_rank})...")
        # Freeze the UNet before adding LoRA adapters.
        self.unet.requires_grad_(False)
        
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Add LoRA adapters to attention layers.
            lora_dropout=cfg.lora_dropout,
            bias="none"
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters() # Print parameter counts to verify LoRA setup.

        # --- 4. Load ControlNet ---
        print(f"[ControlNet] Loading Canny weights from {cfg.controlnet_canny_path}...")
        self.controlnet = ControlNetModel.from_pretrained(cfg.controlnet_canny_path).to(self.device)

        # --- 5. Initialize the Metric Projector ---
        # Infer the metric dimension from the dataset instead of hard-coding it.
        if getattr(self.ds, "movement_info", None) is not None:
            metric_dim = len(self.ds.movement_info["names"])
        else:
            metric_dim = 36
        print(f"[Projector] Initializing Metric Projector... (input_dim={metric_dim})")
        self.projector = MetricProjector(input_dim=metric_dim, output_dim=768).to(self.device)

        # --- 6. Configure frozen modules and optimization ---
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()

        # Train all ControlNet and Projector parameters.
        self.controlnet.requires_grad_(True)
        self.projector.requires_grad_(True)
        
        # Enable gradient checkpointing to reduce memory use.
        # Apply it to both the UNet LoRA model and ControlNet.
        print("[Memory] Enabling Gradient Checkpointing for UNet and ControlNet")
        try:
            self.unet.enable_gradient_checkpointing()
            self.controlnet.enable_gradient_checkpointing()
        except Exception as e:
            print(f"[Warning] Gradient Checkpointing failed: {e}")

        if cfg.use_xformers and is_xformers_available():
            print("[xFormers] Enabling memory efficient attention")
            self.unet.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()

        self.controlnet.train()
        self.unet.train() # Keep the PEFT model in training mode.
        self.projector.train()

        # Optimize the UNet LoRA parameters, ControlNet, and Projector.
        # Filtering by requires_grad selects only trainable UNet LoRA parameters.
        params = list(filter(lambda p: p.requires_grad, self.unet.parameters())) + \
                 list(self.controlnet.parameters()) + \
                 list(self.projector.parameters())
                 
        print(f"[Optimizer] Total trainable params: {sum(p.numel() for p in params)}")
        self.opt = torch.optim.AdamW(params, lr=cfg.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        os.makedirs(cfg.output_dir, exist_ok=True)

        # --- 7. Validator ---
        self.validator = None 
        if os.path.exists(cfg.validation_image_dir):
            print(f"[Trainer] 初始化验证器...")
            try:
                # Initialize the validator.
                self.validator = TrainingValidator(cfg, self.device)
            except Exception as e:
                print(f"[Warning] 验证器初始化失败: {e}")

        self.loss_history = []
        self.val_history = {
            'step': [], 'mae': [], 'mse': [], 'lpips': []
        }
        
    def encode_prompt(self, prompt_list):
        inputs = self.tokenizer(
            prompt_list, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return self.text_encoder(inputs.input_ids.to(self.device))[0]

    def plot_validation_metrics(self):
        if len(self.val_history['step']) == 0: return
        steps = self.val_history['step']
        metrics_config = [
            ('mae', 'Validation MAE (mm)', 'val_mae.png', 'red'),
            ('mse', 'Validation MSE',      'val_mse.png', 'blue'),
            ('lpips', 'Validation LPIPS',  'val_lpips.png', 'green')
        ]
        for key, title, filename, color in metrics_config:
            plt.figure(figsize=(10, 5))
            plt.plot(steps, self.val_history[key], label=key.upper(), color=color, marker='o')
            plt.title(title)
            plt.xlabel('Steps')
            plt.ylabel(key.upper())
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(self.cfg.output_dir, filename))
            plt.close()

    def plot_loss(self):
        if len(self.loss_history) == 0: return
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.savefig(os.path.join(self.cfg.output_dir, "loss_curve.png"))
        plt.close()

    def train(self):
        seed_all(self.cfg.seed)
        dl = DataLoader(self.ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
        step = 0
        
        print(f"[Trainer] 开始 LoRA 训练...")
        print(f"[Trainer] Max Steps: {self.cfg.max_steps}, Accumulation: {self.cfg.gradient_accumulation_steps}")

        while step < self.cfg.max_steps:
            for batch in dl:
                # ------------------------- Data Loading -------------------------
                pixel_values = batch["pixel_values"].to(self.device)
                cond_canny = batch["conditioning_pixel_values"].to(self.device)
                metrics = batch["conditioning_metrics"].to(self.device).float()
                prompts = batch["prompt"]

                # ------------------------- Latent Encoding -------------------------
                with torch.no_grad():
                    latents = self.vae.encode(pixel_values * 2 - 1).latent_dist.sample() * 0.18215
                    encoder_hidden_states = self.encode_prompt(prompts)

                # Metric Injection
                metric_emb = self.projector(metrics)
                encoder_hidden_states = torch.cat([encoder_hidden_states, metric_emb], dim=1)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.size(0),), device=self.device).long()
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

                # ------------------------- Forward Pass -------------------------
                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    
                    # 1. ControlNet
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=cond_canny,
                        return_dict=False,
                    )

                    # 2. UNet (LoRA)
                    pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    loss = F.mse_loss(pred, noise)

                # ------------------------- Backward -------------------------
                loss = loss / self.cfg.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                
                step += 1
                self.loss_history.append(loss.item() * self.cfg.gradient_accumulation_steps)

                # ------------------------- Logging & Saving -------------------------
                if step % 100 == 0:
                    print(f"[Step {step}/{self.cfg.max_steps}] Loss: {self.loss_history[-1]:.4f}")

                need_viz = (step % self.cfg.save_every == 0)

                if step % self.cfg.save_every == 0:
                    self.save_checkpoints(step)
                    self.plot_loss()

                if self.validator is not None and step % self.cfg.validation_steps == 0:
                    print(f"[Validator] Running validation at step {step} (Save Viz: {need_viz})...")
                    
                    # Switch to evaluation mode.
                    self.controlnet.eval()
                    self.unet.eval() 
                    self.projector.eval()
                    
                    # Run validation with the UNet containing the LoRA adapters.
                    val_mae, val_mse, val_lpips = self.validator.validate(
                        vae=self.vae,
                        unet=self.unet, 
                        controlnet=self.controlnet,
                        text_encoder=self.text_encoder,
                        tokenizer=self.tokenizer,
                        projector=self.projector,
                        step_idx=step,
                        output_dir=self.cfg.output_dir,
                        save_visuals=need_viz
                    )
                    
                    self.val_history['step'].append(step)
                    self.val_history['mae'].append(val_mae)
                    self.val_history['mse'].append(val_mse)
                    self.val_history['lpips'].append(val_lpips)
                    self.plot_validation_metrics()
                    
                    # Restore training mode.
                    self.controlnet.train()
                    self.unet.train()
                    self.projector.train()

                if step >= self.cfg.max_steps:
                    break
        
        self.save_checkpoints("final")
        self.plot_loss()
        print("[Trainer] Training finished.")

    def save_checkpoints(self, tag):
        print(f"\n[Step {tag}] Saving models...")
        
        # 1. Save ControlNet
        save_path_canny = os.path.join(self.cfg.output_dir, f"controlnet_canny_{tag}")
        self.controlnet.save_pretrained(save_path_canny)
        
        # 2. Save UNet LoRA (Key Change: Only save adapters, not full UNet)
        save_path_unet = os.path.join(self.cfg.output_dir, f"unet_lora_{tag}")
        self.unet.save_pretrained(save_path_unet)
        
        # 3. Save Projector
        save_path_proj = os.path.join(self.cfg.output_dir, f"metric_projector_{tag}.pth")
        torch.save(self.projector.state_dict(), save_path_proj)
        
        print(f"Models saved to {self.cfg.output_dir}")

# --------------------- Main Execution ---------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Canny ControlNet + UNet LoRA")
    
    # --- Paths ---
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--unet_path", type=str, default="", help="Optional pre-trained UNet path")
    parser.add_argument("--controlnet_canny_path", type=str, default="lllyasviel/control_v11p_sd15_canny")
    parser.add_argument("--output_dir", type=str, required=True)
    
    # --- Training hyperparameters ---
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate is usually higher")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    
    # --- LoRA Params ---
    parser.add_argument("--lora_rank", type=int, default=16)
    
    # --- Validation and checkpointing ---
    parser.add_argument("--validation_image_dir", type=str, default="data/test_image")
    parser.add_argument("--landmark_weights", type=str, default="model/best_model.pth")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--validation_steps", type=int, default=1000)

    # --- Dataset CSV files ---
    parser.add_argument("--rotation_csv", type=str, default=None)
    parser.add_argument("--pre_csv", type=str, default=None)
    parser.add_argument("--post_csv", type=str, default=None)
    parser.add_argument("--information_csv", type=str, default="data/Information.csv")
    parser.add_argument("--movement_csv", type=str, default="data/movement.csv")
    parser.add_argument(
        "--exclude_patients",
        type=str,
        default=DEFAULT_EXCLUDE_PATIENTS,
        help="Comma-separated patient names to exclude from training, validation, and movement statistics. Use empty string to disable.",
    )
    
    # --- Image params ---
    parser.add_argument("--heatmap_sigma", type=float, default=8.0)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg = TrainConfig(
        data_root=args.data_root,
        base_model=args.base_model,
        output_dir=args.output_dir,
        unet_path=args.unet_path,
        controlnet_canny_path=args.controlnet_canny_path,
        
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        use_amp=args.amp,
        
        lora_rank=args.lora_rank,
        
        validation_image_dir=args.validation_image_dir,
        landmark_weights=args.landmark_weights,
        save_every=args.save_every,
        validation_steps=args.validation_steps,
        
        rotation_csv=args.rotation_csv,
        pre_csv=args.pre_csv,
        post_csv=args.post_csv,
        information_csv=args.information_csv,
        movement_csv=args.movement_csv,
        exclude_patients=parse_patient_list(args.exclude_patients),
        heatmap_sigma=args.heatmap_sigma
    )
    
    print(f"[Dataset] Loading ControlNetPairDataset...")
    ds = ControlNetPairDataset(
        root=cfg.data_root,
        size=cfg.size,
        rotation_csv=cfg.rotation_csv,
        pre_csv=cfg.pre_csv,
        post_csv=cfg.post_csv,
        information_csv=cfg.information_csv,
        movement_csv=cfg.movement_csv,
        exclude_patients=cfg.exclude_patients,
        heatmap_sigma=cfg.heatmap_sigma 
    )
    print(f"[Dataset] Size: {len(ds)}")
    
    trainer = CannyControlNetLoraTrainer(cfg, ds)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Stop] Interrupted by user. Saving current state...")
        trainer.save_checkpoints("interrupted")
        print("[Stop] Saved.")

if __name__ == "__main__":
    main()
