""" saves mean and std of R,G,B channels of the dataset """
import os
import numpy as np
import cv2

from utils_aich.open_files import *

img_path = '/media/aich/DATA/road_extraction/train/images';
th_low = -1; # low threshold for black region
val_max_img = 255; # maximum possible value in an 8-bit RGB image

mean_file = 'mean';
std_file = 'std';

mean = [0.] * 3;
std = [0.] * 3;
count = 0;
for img_name in os.listdir(img_path) :
    if not is_image_file(img_name) :
        continue;
    print img_name;
    count += 1;
    im = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_UNCHANGED);
    im = im[:,:,0:3];
    ind_valid = np.logical_and(im[:,:,0]>th_low, im[:,:,1]>th_low);
    ind_valid = np.logical_and(im[:,:,2]>th_low, ind_valid);
    mean += im[ind_valid].mean(0);
    std += im[ind_valid].std(0);

mean /= (count * val_max_img);
std /= (count * val_max_img);

# swap R and B channel values
tmp = mean[0];
mean[0] = mean[2];
mean[2] = tmp;
tmp = std[0];
std[0] = std[2];
std[2] = tmp;

print mean;
print std;

save_pickle(mean_file, mean.tolist());
save_pickle(std_file, std.tolist());
