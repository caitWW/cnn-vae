import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url

import os,glob
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import trange, tqdm

from RES_VAE import VAE as VAE
from vgg19 import VGG19

import json

batch_size = 16
image_size = 64 #QH edited from 64 to 128
lr = 1e-4
nepoch = 100
start_epoch = 0
#dataset_root = "/media/luke/Quick Storage/Data"
#dataset_root = "/scratch/gpfs/qh8777/qhhome/celebA"
save_dir = "/home/qw3971/cnn-vae/test_shifted/"
model_name = "test_run"
load_checkpoint = True

use_cuda = torch.cuda.is_available()
GPU_indx  = 1
device = torch.device(GPU_indx if use_cuda else "cpu")
device

#only if not using GPU
device = torch.device("cpu")
device

dataset_root = '/home/qw3971/clevr/image_generation/new_transform_shifted/'
transform = transforms.Compose([# transforms.Resize(image_size),
                                # transforms.CenterCrop(image_size),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

def split_dataset():

    train_im_ids = []
    test_im_ids = []    

    # Get all .png files in the folder
    png_files = [file for file in os.listdir(dataset_root) if file.endswith('.png')]

    # Shuffle the list of files randomly
    random.shuffle(png_files)

    # Calculate the split index for train/test
    # split_index = int(0.8 * len(png_files))  #80%training
    split_index = int(0.99* len(png_files))

    # Assign 80% as train_im_ids and 20% as test_im_ids
    train_im_ids = png_files[:split_index]
    test_im_ids = png_files[split_index:]

    return train_im_ids, test_im_ids

# training code
train_ids, test_ids = split_dataset()
print('num train_images:', len(train_ids))
print('num test_images:', len(test_ids))

# heavy cpu load, light memory load
class ImageDiskLoader(torch.utils.data.Dataset):

    def __init__(self, im_ids):
        self.transform = transform
        self.im_ids = im_ids

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im_name = self.im_ids[idx]
        im_path = dataset_root + self.im_ids[idx]
        im = Image.open(im_path).convert('RGB')
        #im = crop(im, 30, 0, 178, 178)
        data = self.transform(im)

        return data, im_name
    
data_train = ImageDiskLoader(train_ids)
data_test = ImageDiskLoader(test_ids)

kwargs = {'num_workers': 1,
          'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, **kwargs)

# Create the feature loss module
feature_extractor = VGG19().to(device)

#Create VAE network
vae_net = VAE(channel_in=3, ch=64, latent_channels=256).to(device)
# setup optimizer
optimizer = optim.Adam(vae_net.parameters(), lr=lr, betas=(0.5, 0.999))
#Loss function
loss_log = []

#Create the save directory if it does note exist
if not os.path.isdir(save_dir + "/Models"):
    os.makedirs(save_dir + "/Models")
if not os.path.isdir(save_dir + "/Results"):
    os.makedirs(save_dir + "/Results")

if load_checkpoint:
    checkpoint = torch.load(save_dir + "/Models/" + model_name + "_" + str(image_size) + ".pt", map_location = "cpu")
    print("Checkpoint loaded")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    vae_net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint["epoch"]
    #loss_log = checkpoint["loss_log"]
else:
    #If checkpoint does exist raise an error to prevent accidental overwriting
    if os.path.isfile(save_dir + "/Models/" + model_name + "_" + str(image_size) + ".pt"):
        raise ValueError("Warning Checkpoint exists")
    else:
        print("Starting from scratch")

dataiter = iter(train_loader)
images, image_ids = next(dataiter)
len(dataiter)

recon_img, mu, logvar = vae_net(images)

# Initialize an accumulator for the difference images
diff_image_accumulator = np.zeros((240, 320))  # assuming your images are square

# Initialize a counter for the number of images processed
num_images = 0

with open('/home/qw3971/cv-2023/retina_data.json', 'r') as f:
    fixation_data = json.load(f)

# Initialize empty lists to hold all distance and error values
distances = []
errors = []

# Iterate over the dataset
for images, image_ids in train_loader:
    # Move the images to the device
    images = images.to(device)
    
    # Generate the reconstructed images
    with torch.no_grad():
        recon_img, _, _ = vae_net(images)

    '''
    image_id = image_ids[i]

    fixation = fixation_data[image_id]
    x_fixation = fixation['xc']
    y_fixation = fixation['yc'] 
    '''

    x_fixation = 160
    y_fixation = 120

    for i in range(images.shape[0]):

        # Create coordinate arrays
        Y, X = np.ogrid[:240, :320]

        # Compute the distance from the fixation point for each pixel
        distance = np.sqrt((X - x_fixation)**2 + (Y - y_fixation)**2)

        # Compute the original image
        img = images[i].cpu().numpy().squeeze()
        img_o = np.zeros((img.shape[1], img.shape[2], img.shape[0]))
        img_o[:,:,0] = img[0,:,:].squeeze()
        img_o[:,:,1] = img[1,:,:].squeeze()
        img_o[:,:,2] = img[2,:,:].squeeze()
        img_o =  (img_o - img_o.min()) / (img_o.max() - img_o.min()) #normalize between 0 and 1

        # Compute the reconstructed image
        img = recon_img[i].cpu().numpy().squeeze()
        img_t = np.zeros((img.shape[1], img.shape[2], img.shape[0]))
        img_t[:,:,0] = img[0,:,:].squeeze()
        img_t[:,:,1] = img[1,:,:].squeeze()
        img_t[:,:,2] = img[2,:,:].squeeze()
        img_t =  (img_t - img_t.min()) / (img_t.max() - img_t.min()) #normalize between 0 and 1

        # Compute the difference image
        diff_image = (img_o - img_t)**2
        diff_image = 0.2989 * diff_image[:,:,0] + 0.5870 * diff_image[:,:,1] + 0.1140 * diff_image[:,:,2]
        diff_image_normalized = (diff_image - diff_image.min()) / (diff_image.max())
        diff_image_inverted = 1.0 - diff_image_normalized

        '''
        # Accumulate the difference image
        diff_image_accumulator += diff_image_inverted

        # Increment the counter
        num_images += 1
        '''

        # Flatten the distance and error arrays and append them to the lists
        distances.append(distance.flatten())
        errors.append(diff_image.flatten())

# Convert the lists to numpy arrays
distances = np.concatenate(distances)
errors = np.concatenate(errors)

# Compute the average difference image
# avg_diff_image = diff_image_accumulator / num_images

# Plot error against distance
plt.scatter(distances, errors)
plt.xlabel('Distance from fixation point (image size of 320x240)')
plt.ylabel('Sum of Squared Difference')

# Save the plot
plt.savefig('scatter_plot.png')