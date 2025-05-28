import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import tqdm.auto as tqdm
from tqdm import tqdm
import glob
import datetime
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import GradScaler, autocast
from pytorch_msssim import ssim
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = 'hw4_release_dataset'
train_dir = os.path.join(ROOT, 'train')
test_dir = os.path.join(ROOT, 'test/degraded')
degraded_dir = os.path.join(train_dir, 'degraded')
clean_dir = os.path.join(train_dir, 'clean')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
device, dtype

class RestorationDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, return_prompt=False, transform = None):
        self.degraded_paths = sorted(glob.glob(os.path.join(degraded_dir, "*.png")))
        print(f"Found {len(self.degraded_paths)} degraded images")
        self.clean_dir = clean_dir
        self.return_prompt = return_prompt
        self.transform = transform

        if self.clean_dir:
            self.pairs = []
            for dp in self.degraded_paths:
                filename = os.path.basename(dp)
                prefix, number = filename.split('-')
                clean_name = prefix + '_clean-' + number
                clean_path = os.path.join(clean_dir, clean_name)
                if os.path.exists(clean_path):
                    self.pairs.append((dp, clean_path))
        else:
            self.pairs = [(p, None) for p in self.degraded_paths]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        degraded_path, clean_path = self.pairs[idx]

        degraded_img = Image.open(degraded_path).convert("RGB")
        if self.transform is None:
            degraded_tensor = T.ToTensor()(degraded_img)
        else:
            degraded_tensor = self.transform(degraded_img)

        if clean_path:
            clean_img = Image.open(clean_path).convert("RGB")
            if self.transform is None:
                clean_tensor = T.ToTensor()(clean_img)
            else:
                clean_tensor = self.transform(clean_img)
        else:
            clean_tensor = None

        # Determine prompt token from filename
        prompt_token = None
        if self.return_prompt:
            fname = os.path.basename(degraded_path)
            if fname.startswith("rain"):
                prompt_token = 0
            elif fname.startswith("snow"):
                prompt_token = 1
            else:
                prompt_token = -1  # unknown or test

        if self.return_prompt:
            return degraded_tensor, clean_tensor, prompt_token
        else:
            return degraded_tensor, clean_tensor

transform_def = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

transform_1 = T.Compose([
    T.Resize((512, 512)),
    T.RandomGrayscale(p=0.1),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(degrees=15),  
    T.GaussianBlur(kernel_size=1, sigma=(0.1, 2.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor()
])

train_dataset = RestorationDataset(degraded_dir = degraded_dir, clean_dir = clean_dir, return_prompt = True, transform = transform_1)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_dataset = RestorationDataset(test_dir, None, True, None)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class PromptedUNet(nn.Module):
    def __init__(self, prompt_dim=2, in_channels=3, out_channels=3):
        super().__init__()
        self.prompt_embedding = nn.Embedding(2, prompt_dim)  # 0: rain, 1: snow
        self.prompt_dim = prompt_dim

        in_ch = in_channels + prompt_dim

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
            ResidualBlock(64)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            ResidualBlock(128)
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            ResidualBlock(256)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            ResidualBlock(128)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            ResidualBlock(64)
        )

        self.final = nn.Conv2d(64, out_channels, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, prompt_token):
        B, C, H, W = x.shape
        prompt = self.prompt_embedding(prompt_token).view(B, self.prompt_dim, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([x, prompt], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))

        # Decoder
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))

        out = self.activation(self.final(d2))
        return out

scaler = GradScaler('cuda')

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

def loss_fn(pred, target):
    mse = F.mse_loss(pred, target)
    ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    return mse + 0.1 * ssim_loss

def train_model(model, dataloader, optimizer, device, num_epochs=20):
    best_loss = float('inf')
    model.to(device)
    model.train()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a directory to store the checkpoints if it does not already exist
    checkpoint_dir = Path(f"{timestamp}")

    # Create the checkpoint directory if it does not already exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # The model checkpoint path
    checkpoint_path = checkpoint_dir/f"{model.name}.pth"

    print(checkpoint_path)

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        total_loss = 0
        total_psnr = 0

        for degraded, clean, prompt in tqdm(dataloader):
            degraded = degraded.to(device)
            clean = clean.to(device)
            prompt = prompt.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(degraded, prompt_token=prompt)
                # loss = loss_fn(output, clean)
                loss = F.mse_loss(output, clean)


            lr_scheduler = CosineAnnealingLR(optimizer, T_max=300000, eta_min=1e-6)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            lr_scheduler.step()

            scaler.update()
            if loss.item() < best_loss:
                best_loss = loss.item()
                
                torch.save(model.state_dict(), checkpoint_path)

            total_loss += loss.item()
            total_psnr += psnr(output, clean).item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}, PSNR: {total_psnr/len(dataloader):.2f} dB")

torch.cuda.empty_cache()


model = PromptedUNet(prompt_dim=2)
model.name = "prompted_unet"
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

train_model(model, train_loader, optimizer, device, num_epochs=20)
modelpath = "model.pth"
model.load_state_dict(torch.load(modelpath))
model.to(device)
model.eval()

transform_test = T.Compose([
    T.ToTensor()
])

# Set your image folder path
folder_path = 'hw4_release_dataset/test/degraded'
output_npz = 'pred.npz'

# Initialize dictionary to hold image arrays
images_dict = {}

# Loop through all files in the folder
with torch.no_grad():
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)

            # Load image and convert to RGB
            image = Image.open(file_path).convert('RGB')
            image_tensor = transform_test(image).unsqueeze(0).to(device)
            prompt = 0
            output = model(image_tensor, prompt_token=torch.tensor([prompt]).to(device))  # Expect (1, 3, H, W)

            # Postprocess output to uint8 RGB image
            pred = output[0].clamp(0, 1).cpu().numpy()  # (3, H, W), float32
            pred = (pred * 255).astype(np.uint8)  # (3, H, W), uint8

            images_dict[filename] = pred

# Save to .npz file
np.savez(output_npz, **images_dict)

print(f"Saved {len(images_dict)} images to {output_npz}")