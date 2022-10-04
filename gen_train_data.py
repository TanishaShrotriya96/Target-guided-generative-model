from shutil import copyfile
import os
import argparse
from utils.helper import create_folder, remove_files, load_file_names

'''
manually choose a target images in the folder ('./temp') saving the cropped images
and select all or a part of cropped images for the training phase
if the computation resources are limited, please select a part of cropped images
'''

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='COWC')
parser.add_argument("--loc_idx", type=str, default='Toronto_03559.8.2')
parser.add_argument("--obj_name", type=str, default='car')
parser.add_argument("--pos_names", type=list, default=['100_100'])
parser.add_argument("--sample_portion", type=float, default=1.0)
args = parser.parse_args()

unlabeled_imgs_path = './temp'
# the name of chosen target images, named by top_left coordinate
pos_names = args.pos_names
img_type = '.png'
sample_portion = args.sample_portion

# the dir to save manually labeled and classified positive images
dataset_name = args.dataset_name
obj_name = args.obj_name
#location name or index in the dataset
loc_idx = args.loc_idx
                    
sample_path = os.path.join('data', dataset_name, obj_name)
loc_name = '_'.join([loc_idx, obj_name])

pos_folder_path = os.path.join(sample_path, loc_name+'_samples', 'positive')
subset_folder_path = os.path.join(sample_path, loc_name+'_samples', 'subset')

# create a folder to save the target images, if exist, clean it first
create_folder(pos_folder_path)
remove_files(pos_folder_path, 'png')


for img_name in pos_names:
    src = os.path.join(unlabeled_imgs_path, img_name+img_type)
    dst = os.path.join(pos_folder_path, img_name+img_type)
    copyfile(src, dst)
    
num_imgs = len([name for name in os.listdir(unlabeled_imgs_path) if os.path.isfile(os.path.join(unlabeled_imgs_path, name))])
num_samples = int(num_imgs*sample_portion)
print('num of cropped images: ', num_imgs)


create_folder(subset_folder_path)
remove_files(subset_folder_path, 'png')

img_names = load_file_names(unlabeled_imgs_path)
img_names = np.array(img_names)
np.random.shuffle(img_names)

for i in range(num_samples):
    img_name = img_names[i]
    src = os.path.join(unlabeled_imgs_path, img_name+img_type)
    dst = os.path.join(subset_folder_path, img_name+img_type)
    copyfile(src, dst)
