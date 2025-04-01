import os
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from model.MDCNet import MDCNet_fcn


# --- Loss Function ---
class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

    def forward(self, pred, mask):
        # Ref: https://github.com/Xiaoqi-Zhao-DLUT/MSNet-M2SNet/blob/main/train.py
        weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
        wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

        pred  = torch.sigmoid(pred)
        inter = ((pred*mask)*weit).sum(dim=(2,3))
        union = ((pred+mask)*weit).sum(dim=(2,3))
        wiou  = 1-(inter+1)/(union-inter+1)
        return (wbce+wiou).mean()

# --- Helper for Counting ---
def get_counts_from_mask(mask_tensor):
    """ Calculates connected components count from a batch of binary masks. """
    batch_counts = []
    # Ensure mask is on CPU and is numpy uint8
    mask_np_batch = mask_tensor.squeeze(1).cpu().numpy().astype(np.uint8)
    # Binarize (assume mask values are 0 or >0)
    mask_np_batch[mask_np_batch > 0] = 1

    for i in range(mask_np_batch.shape[0]):
        mask_np = mask_np_batch[i]
        # Use cv2.connectedComponents. The first output is the number of labels (including background)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        # Subtract 1 for the background label
        count = max(0, num_labels - 1)
        batch_counts.append(count)

    # Return as a float tensor on the original device
    return torch.tensor(batch_counts, dtype=torch.float32, device=mask_tensor.device).unsqueeze(1) # Shape [B, 1]

# --- Dataset ---
class PBDDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(352, 352)):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img_crop')
        self.neg_line_mask_dir = os.path.join(root_dir, 'neg_line_mask_crop')
        self.neg_point_mask_dir = os.path.join(root_dir, 'neg_point_mask_crop')
        self.pos_line_mask_dir = os.path.join(root_dir, 'pos_line_mask_crop')
        self.pos_point_mask_dir = os.path.join(root_dir, 'pos_point_mask_crop')

        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        self.transform = transform
        self.target_size = target_size

        # Basic resize and ToTensor for masks
        self.mask_transform = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST), # Use NEAREST for masks
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        neg_line_path = os.path.join(self.neg_line_mask_dir, img_name)
        neg_point_path = os.path.join(self.neg_point_mask_dir, img_name)
        pos_line_path = os.path.join(self.pos_line_mask_dir, img_name)
        pos_point_path = os.path.join(self.pos_point_mask_dir, img_name)

        # Load image (ensure RGB)
        image = Image.open(img_path).convert('RGB')

        # Load masks (ensure Grayscale - L mode)
        neg_line_mask = Image.open(neg_line_path).convert('L')
        neg_point_mask = Image.open(neg_point_path).convert('L')
        pos_line_mask = Image.open(pos_line_path).convert('L')
        pos_point_mask = Image.open(pos_point_path).convert('L')

        # Apply image transform (includes resize, augmentations, ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)
        else:
            # Default minimal transform if none provided
            temp_transform = T.Compose([T.Resize(self.target_size), T.ToTensor()])
            image = temp_transform(image)

        # Apply mask transform (just resize and ToTensor)
        neg_line_mask = self.mask_transform(neg_line_mask)
        neg_point_mask = self.mask_transform(neg_point_mask)
        pos_line_mask = self.mask_transform(pos_line_mask)
        pos_point_mask = self.mask_transform(pos_point_mask)

        # Normalize masks to [0, 1] if loaded as 0-255
        neg_line_mask = (neg_line_mask > 0.5).float() # Binarize
        neg_point_mask = (neg_point_mask > 0.5).float()
        pos_line_mask = (pos_line_mask > 0.5).float()
        pos_point_mask = (pos_point_mask > 0.5).float()

        return {
            'image': image,
            'neg_line_mask': neg_line_mask,
            'neg_point_mask': neg_point_mask,
            'pos_line_mask': pos_line_mask,
            'pos_point_mask': pos_point_mask,
            'name': img_name # For potential debugging/logging
        }

# --- Main Training Script ---
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    # Define transforms - Adjust normalization based on dataset stats if known
    img_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(), # As mentioned in paper
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats typical starting point
    ])

    dataset = PBDDataset(root_dir=args.data_root, transform=img_transform, target_size=(args.img_size, args.img_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # --- Load Fixed Anchor Image ---
    try:
        anchor_img = Image.open(args.anchor_path).convert('RGB')
        anchor_tensor = img_transform(anchor_img).unsqueeze(0).to(device) # Add batch dim and move to device
        print(f"Loaded anchor image from: {args.anchor_path}")
    except Exception as e:
        print(f"Error loading anchor image: {e}")
        print("Ensure --anchor_path points to a valid image file.")
        return

    # --- Model ---
    model = MDCNet_fcn().to(device)

    # --- Optimizer and Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # StepLR as mentioned: decay LR by 0.9 every 30 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    # --- Loss Functions ---
    seg_loss_fn = SegmentationLoss().to(device)
    count_loss_fn = nn.L1Loss().to(device) # L1 loss for counts

    # --- Training Loop ---
    print("Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            images = batch['image'].to(device)
            neg_line_gt = batch['neg_line_mask'].to(device)
            neg_point_gt = batch['neg_point_mask'].to(device)
            pos_line_gt = batch['pos_line_mask'].to(device)
            pos_point_gt = batch['pos_point_mask'].to(device)

            # Calculate ground truth counts from point masks
            neg_count_gt = get_counts_from_mask(neg_point_gt)
            pos_count_gt = get_counts_from_mask(pos_point_gt)

            # Repeat anchor tensor to match batch size
            current_batch_size = images.size(0)
            anchor_batch = anchor_tensor.repeat(current_batch_size, 1, 1, 1)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # output_fpn_p: point logits [B, 2, H, W] (neg, pos)
            # regression_neg/pos: count predictions [B, 1]
            # line_neg/pos: line logits [B, 1, H, W]
            output_fpn_p, regression_neg, regression_pos, line_neg, line_pos = model(images, anchor_batch)

            # Separate point predictions
            pred_point_neg_logits = output_fpn_p[:, 0, :, :].unsqueeze(1) # Keep channel dim [B, 1, H, W]
            pred_point_pos_logits = output_fpn_p[:, 1, :, :].unsqueeze(1) # Keep channel dim [B, 1, H, W]

            # Calculate losses
            loss_p_neg = seg_loss_fn(pred_point_neg_logits, neg_point_gt)
            loss_p_pos = seg_loss_fn(pred_point_pos_logits, pos_point_gt)
            loss_l_neg = seg_loss_fn(line_neg, neg_line_gt)
            loss_l_pos = seg_loss_fn(line_pos, pos_line_gt)
            loss_c_neg = count_loss_fn(regression_neg, neg_count_gt)
            loss_c_pos = count_loss_fn(regression_pos, pos_count_gt)

            # Combine losses (adjust weights here if needed)
            total_loss = (loss_p_neg + loss_p_pos) + \
                         (loss_l_neg + loss_l_pos) + \
                         (loss_c_neg + loss_c_pos)

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())

        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Step the scheduler
        scheduler.step()

        # --- Save Checkpoint ---
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, f"mdcnet_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

    print("Training finished.")
    model.eval()
    os.makedirs('saved_model', exist_ok=True) # Ensure save directory exists
    torch.save(model.state_dict(), 'saved_model/MDCNet_seg.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MDCNet for Power Battery Detection")
    parser.add_argument('--data_root', type=str, required=True, help='Path to the root directory of the training dataset (containing img_crop, etc.)')
    parser.add_argument('--anchor_path', type=str, required=True, help='Path to the fixed anchor (prompt) image file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--img_size', type=int, default=352, help='Image size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=12, help='Training batch size') # Default from paper
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate') # Default from paper
    parser.add_argument('--lr_decay_step', type=int, default=30, help='Step size for learning rate decay') # Default from paper
    parser.add_argument('--lr_decay_gamma', type=float, default=0.9, help='Learning rate decay factor') # Default from paper
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')

    args = parser.parse_args()

    train(args)
