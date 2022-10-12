import numpy as np
import argparse
import os
import cv2
import tensorflow as tf
import keras
import TGG
import argparse
from keras.optimizers import *
from utils import load_data, data_augmentation, helper

def process_inputs(_x_u, _x_l, batch_size):
    np.random.shuffle(_x_u)
    num_samples = _x_u.shape[0]//batch_size * batch_size
    print('num samples: ',  num_samples)
    print('unlabeled samples shape: ', _x_u.shape)
    print('labeled samples shape: ', _x_l.shape)
    _x_l_aug = data_augmentation.data_aug(_x_l, SHIFT_LIST, ROTATION_ANGLE)
    np.random.shuffle(_x_l_aug)
    print('augmented labeled samples shape: ', _x_l_aug.shape)

    _x_l = np.reshape(_x_l, [-1, IMG_SIZE*IMG_SIZE*3])
    _x_l_aug = np.reshape(_x_l_aug, [-1, IMG_SIZE*IMG_SIZE*3])
    _x_u = np.reshape(_x_u, [-1, IMG_SIZE*IMG_SIZE*3])

    _x_u = _x_u.astype('float32') / 255
    _x_l = _x_l.astype('float32') / 255
    _x_l_aug = _x_l_aug.astype('float32') / 255


    while _x_l_aug.shape[0] < _x_u.shape[0]:
        _x_l_aug = np.vstack((_x_l_aug, _x_l_aug[:_x_u.shape[0]-_x_l_aug.shape[0]]))

    np.random.shuffle(_x_l_aug)
    np.random.shuffle(_x_u)
    _x_l_aug = _x_l_aug[:num_samples]
    _x_u = _x_u[:num_samples]
    print('target samples shape: ', _x_l_aug.shape)
    print('all samples shape: ', _x_u.shape)
    return _x_u, _x_l_aug
    
def inference(model, x_test, x_test_idx, x_l_test, thres, pos_folder_path, save_dir, img_size=30):
    print('processing {:s}'.format(loc_name))
    while x_l_test.shape[0] < x_test.shape[0]:
        x_l_test = np.vstack((x_l_test, x_l_test[:x_test.shape[0]-x_l_test.shape[0]])) 

    x_l_test = x_l_test[:x_test.shape[0]]

    x_u_pred0, x_u_pred1, x_l_pred, y_u_pred, _,_,_,_,_,_,_,_,_,_ \
        = model.predict([x_test, x_l_test, np.array([[1,0]]*x_test.shape[0]), np.array([[0,1]]*x_test.shape[0])])

    detected_idx = helper.load_file_names(pos_folder_path)
    print('num of detected: ', len(detected_idx))
    pred_pos_idx = []
    num_detected = len(helper.load_file_names(save_dir))
    while num_detected == 0 and thres>0.5:   
        for i in range(x_test.shape[0]): 
            if y_u_pred[i, 0] > thres:
                pred_pos_idx.append(x_test_idx[i])
                temp_name = loc_name + '_' + x_test_idx[i]
                if temp_name not in detected_idx:
                    cv2.imwrite(os.path.join(save_dir, x_test_idx[i]+'.png']), x_test[i].reshape((img_size, img_size, 3))*255)
                    print('save new detected image in %s'%(save_dir))
        thres = thres - 0.0019
        num_detected = len(helper.load_file_names(save_dir))
        print('the threshold and num of detected image: ', thres, num_detected)
    print('total number of detected in is {:d}'.format(num_detected))

    
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='COWC')
parser.add_argument("--obj_name", type=str, default='car')
parser.add_argument("--loc_idx", type=str, default='car')
parse.add_argument("--map_path", type=str, default='./data/COWC/Toronto_03553.7.8/Toronto_03553.7.8.jpg')
parser.add_argument("--augmentation", type=boolean, default=True)
parser.add_argument("--image_size", type=int, default=50)
parser.add_argument("--stride", type=int, default=20)
parser.add_argument("--num_epochs", type=int, default=400)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=400)
parser.add_argument("--weight", type=int, default=500)
parser.add_argument("--saved_model_path", type=str, default='TGG.h5df')
parser.add_argument("--save_detection_folder", type=str, default='./detected_img')
args = parser.parse_args()
#######################################################    
# hyperparamters
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
obj_name = args.obj_name
dataset_name = args.dataset_name
loc_idx = args.loc_idx
# DATA_DIR = '_'.join([obj_name, args.dataset_name, 'test'])

if args.augmentation:
    SHIFT_LIST = [-2, -1 ,0, 1, 2] 
    ROTATION_ANGLE = [90, 180, 270, 360]
else: 
    SHIFT_LIST = [] 
    ROTATION_ANGLE = []
    
IMG_SIZE = args.image_size
STRIDE = args.stride
EPOCHS = args.num_epochs
LEARNING_RATE = args.learnign_rate
BATCH_SIZE = args.batch_size

latent_dim = 16 #32, 5
intermediate_dim = 256 #256 for dior ships
num_cls = 2
optimizer = Adam(lr=LEARNING_RATE)
# optimizer = RMSprop(lr=LEARNING_RATE)
initializer = 'glorot_normal'#'random_uniform'#
original_dim = IMG_SIZE*IMG_SIZE*3
w_recons, w_kl, w_ce = IMG_SIZE*IMG_SIZE*3.0, 1.0, args.weight
threshold = 0.5

sample_path = os.path.join('data', dataset_name, obj_name)
loc_name = '_'.join([loc_idx, obj_name])
pos_samples_path = os.path.join(sample_path, loc_name+'_samples', 'positive')
map_path = args.map_path
save_detection_path = args.save_detection_folder
SAVE_MODEL_PATH = args.saved_model_path
print('directory for labeled images: ', pos_samples_path)
print('path to the map: ', map_path)
#######################################################

#######################################################
# Data Processing
labele_samples, _ = load_data.load_wetland_samples(pos_samples_path)
all_samples, _ = load_data.load_all_data(map_path, '', IMG_SIZE, STRIDE)
_x_u, _x_l_aug = process_inputs(all_samples, labele_samples, BATCH_SIZE)
print('unlabeled img, augmented labeled img shape: ', _x_u.shape, _x_l_aug.shape)
#######################################################

#######################################################
# Training Phase
vae = TGG.initialize_tgg(original_dim, num_cls, latent_dim, intermediate_dim,w_recons, w_kl, w_ce)
optimizer = Adam(lr=LEARNING_RATE)
TGG.train_tgg(vae, optimizer, _x_u, _x_l_aug, EPOCHS, BATCH_SIZE, SAVE_MODEL_PATH, LOAD_MODEL_PATH)
#######################################################

#######################################################
# Testing Phase
vae = TGG.initialize_tgg(original_dim, num_cls, latent_dim, intermediate_dim, w_recons, w_kl, w_ce)
vae.load_weights(SAVE_MODEL_PATH)
helper.create_folder(save_detection_path)
helper.remove_files(save_detection_path)
x_test, x_test_idx = load_data.load_all_data(map_path, '', IMG_SIZE, STRIDE)
x_test = np.array(x_test) / 255.0
x_test = np.reshape(x_test, (-1, IMG_SIZE*IMG_SIZE*3))
threshold = 0.501
inference(vae, x_test,  x_test_idx, _x_l_aug, threshold, pos_samples_path, save_detection_path)
#######################################################
