import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import random
from PIL import Image

import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse

import Helpers as hf
from RES_VAE import VAE
from vgg19 import VGG19

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=True)
parser.add_argument("--target_root", "-tr", help="Target root dir", type=str, required=True)

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=128)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=64)

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=256)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)

# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--feature_scale", "-fs", help="Feature loss scale", type=float, default=1)
parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")

# Create dataloaders
# This code assumes there is no pre-defined test/train split and will create one for you
'''
transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])
'''

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

def split_dataset(files):

    train_im_ids = []
    test_im_ids = []    

    # Get all .png files in the folder
    png_files = files

    # Shuffle the list of files randomly
    # random.shuffle(png_files)

    # Calculate the split index for train/test
    split_index = int(0.8 * len(png_files))  #80%training

    # Assign 80% as train_im_ids and 20% as test_im_ids
    train_im_ids = png_files[:split_index]
    test_im_ids = png_files[split_index:]

    return train_im_ids, test_im_ids

# training code
train_files = sorted([file for file in os.listdir(args.dataset_root) if file.endswith('.png')])
train_ids, test_ids = split_dataset(train_files)
print('num train_images:', len(train_ids))
print('num test_images:', len(test_ids))

target_files = sorted([file for file in os.listdir(args.target_root) if file.endswith('.png')])
target_ids, target_test_ids = split_dataset(target_files)

import json

with open('/home/qw3971/cnn-vae/run2_saccade.json', 'r') as json_file:
    saccade_dict = json.load(json_file)

# Convert the values (the saccade vectors) to tensors
saccade_dict = {k: torch.tensor(v).float() for k, v in saccade_dict.items()} 

# heavy cpu load, light memory load
class ImageDiskLoader(torch.utils.data.Dataset):

    def __init__(self, im_ids, root, saccade_dict):
        self.transform = transform
        self.im_ids = im_ids
        self.root = root
        self.saccade_dict = saccade_dict

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im_path = args.dataset_root + self.im_ids[idx]
        im = Image.open(im_path).convert('RGB')
        #im = crop(im, 30, 0, 178, 178)
        saccade_vector = self.saccade_dict[self.im_ids[idx]]
        data = self.transform(im)

        return data, self.im_ids[idx], saccade_vector
    
# heavy cpu load, light memory load
class ImageDiskLoaderTarget(torch.utils.data.Dataset):

    def __init__(self, im_ids, root):
        self.transform = transform
        self.im_ids = im_ids
        self.root = root

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im_path = self.root + self.im_ids[idx]
        im = Image.open(im_path).convert('RGB')
        #im = crop(im, 30, 0, 178, 178)
        data = self.transform(im)

        return data, self.im_ids[idx]
    
data_train = ImageDiskLoader(train_ids, args.dataset_root, saccade_dict)
data_test = ImageDiskLoader(test_ids, args.dataset_root, saccade_dict)
data_target =  ImageDiskLoaderTarget(target_ids, args.target_root)
data_target_test =  ImageDiskLoaderTarget(target_test_ids, args.target_root)

kwargs = {'num_workers': 1,
          'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, **kwargs)
target_loader = torch.utils.data.DataLoader(data_target, batch_size=args.batch_size, shuffle=False, **kwargs)
target_test_loader = torch.utils.data.DataLoader(data_target_test, batch_size=args.batch_size, shuffle=False, **kwargs)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(test_loader)
test_images, image_ids, saccades = next(dataiter) 

dataiter = iter(target_test_loader)
target_test_images, target_test_image_ids = next(dataiter) 

# Create AE network.
vae_net = VAE(channel_in=test_images.shape[1],
              ch=args.ch_multi,
              latent_channels=args.latent_channels).to(device)

# Setup optimizer
optimizer = optim.Adam(vae_net.parameters(), lr=args.lr)

# AMP Scaler
scaler = torch.cuda.amp.GradScaler()

# Create the feature loss module if required
if args.feature_scale > 0:
    feature_extractor = VGG19().to(device)

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in vae_net.parameters():
    num_model_params += param.flatten().shape[0]
print("This model has %d (approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))

# Create the save directory if it does not exist
if not os.path.isdir(args.save_dir + "/Models"):
    os.makedirs(args.save_dir + "/Models")
if not os.path.isdir(args.save_dir + "/Results"):
    os.makedirs(args.save_dir + "/Results")
if not os.path.isdir(args.save_dir + "/Recon"):
    os.makedirs(args.save_dir + "/Recon")
if not os.path.isdir(args.save_dir + "/Original"):
    os.makedirs(args.save_dir + "/Original")

for i in range(test_images.size(0)):
                        vutils.save_image(test_images[i],
                                      "%s/%s/%s_%d_test_%d_%d.png" % (args.save_dir,
                                                                       "Original",
                                                                       args.model_name,
                                                                       args.image_size,
                                                                       1,
                                                                       i),
                                      normalize=True)

# Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
# If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
# it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
save_file_name = args.model_name + "_" + str(args.image_size)
if args.load_checkpoint:
    checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt",
                            map_location="cpu")
    print("Checkpoint loaded")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    vae_net.load_state_dict(checkpoint['model_state_dict'])

    if not optimizer.param_groups[0]["lr"] == args.lr:
        print("Updating lr!")
        optimizer.param_groups[0]["lr"] = args.lr

    start_epoch = checkpoint["epoch"]
    data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
else:
    # If checkpoint does exist raise an error to prevent accidental overwriting
    if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
        raise ValueError("Warning Checkpoint exists")
    else:
        print("Starting from scratch")
        start_epoch = 0
        # Loss and metrics logger
        data_logger = defaultdict(lambda: [])

# Start training loop
for epoch in trange(start_epoch, args.nepoch, leave=False):
    vae_net.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_batches = 0

    # Whether to train with SaccadeNet
    train_with_saccade = True

    for i, ((images, ids, saccades), (target_images, target_ids)) in enumerate(tqdm(zip(train_loader, target_loader), 
                                                            total = len(train_loader),  leave=False)):
        current_iter = i + epoch * len(train_loader)
        images = images.to(device)
        target_images = target_images.to(device)
        saccade = saccades.to(device)
        bs, c, h, w = images.shape
        
        # We will train with mixed precision!
        with torch.cuda.amp.autocast():
            if train_with_saccade:
                recon_img, mu, log_var = vae_net(images, saccade)

            else:
                dummy_saccade_data = torch.zeros_like(images)
                recon_img, mu, log_var = vae_net(images, dummy_saccade_data)

            kl_loss = hf.kl_loss(mu, log_var)
            mse_loss = F.mse_loss(recon_img, target_images)
            loss = args.kl_scale * kl_loss + mse_loss

            # Perception loss
            if args.feature_scale > 0:
                feat_in = torch.cat((recon_img, target_images), 0)
                feature_loss = feature_extractor(feat_in)
                loss += args.feature_scale * feature_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 40)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses over the epoch
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_batches += 1

        avg_loss = total_loss / total_batches
        avg_mse_loss = total_mse_loss / total_batches
        print(f'Epoch: {epoch+1}, Avg Loss: {avg_loss}, Avg MSE Loss: {avg_mse_loss}')

        # Log losses and other metrics for evaluation!
        data_logger["mu"].append(mu.mean().item())
        data_logger["mu_var"].append(mu.var().item())
        data_logger["log_var"].append(log_var.mean().item())
        data_logger["log_var_var"].append(log_var.var().item())

        data_logger["kl_loss"].append(kl_loss.item())
        data_logger["img_mse"].append(mse_loss.item())
        data_logger["feature_loss"].append(feature_loss.item())

            # In eval mode the model will use mu as the encoding instead of sampling from the distribution
        vae_net.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Save an example from testing and log a test loss
                recon_img, mu, log_var = vae_net(test_images.to(device))
                data_logger['test_mse_loss'].append(F.mse_loss(recon_img,
                                                                   target_test_images.to(device)).item())

                img_cat = torch.cat((recon_img.cpu(), target_test_images), 2).float()
                vutils.save_image(img_cat,
                                      "%s/%s/%s_%d_test_%d.png" % (args.save_dir,
                                                                "Results",
                                                                args.model_name,
                                                                args.image_size,
                                                                epoch),
                                      normalize=True)
                # Only save images on the final epoch
                if epoch == args.nepoch - 1:
                    for i in range(recon_img.size(0)):
                        vutils.save_image(recon_img[i],
                                      "%s/%s/%s_%d_test_%d_%d.png" % (args.save_dir,
                                                                       "Recon",
                                                                       args.model_name,
                                                                       args.image_size,
                                                                       epoch,
                                                                       i),
                                      normalize=True)

                # Keep a copy of the previous save in case we accidentally save a model that has exploded...
            if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
                shutil.copyfile(src=args.save_dir + "/Models/" + save_file_name + ".pt",
                                    dst=args.save_dir + "/Models/" + save_file_name + "_copy.pt")

            # Save a checkpoint
            torch.save({
                        'epoch': epoch + 1,
                        'data_logger': dict(data_logger),
                        'model_state_dict': vae_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, args.save_dir + "/Models/" + save_file_name + ".pt")

            # Set the model back into training mode!!
            vae_net.train() 