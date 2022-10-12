# Target Guided Generative Model (TGGM)

## Paper link: https://arxiv.org/abs/2112.05786

## Paper citation:

@article{duanguided,<br/>
  title={Guided Generative Models using Weak Supervision for Detecting Object Spatial Arrangement in Overhead Images},  <br/>
  author={Duan, Weiwei and Chiang, Yao-Yi and Leyk, Stefan and Uhl, Johannes H and Knoblock, Craig A},  <br/>
  booktitle={2021 IEEE International Conference on Big Data (Big Data)},<br/>
  year={2021},  <br/>
  organization={IEEE}
}


## The goal is to detect the approximate locations of target objects within a region-of-interest using only one or a few manually labeled target object


### 1) Rerequired libraries
Python == 3.5

Keras == 2.1.6 

tensorflow-gpu == 1.2.0 

cuda == 8.0 

numpy, cv2, os, shutil, 

### Alternatively, all the required packages and libaries are in the docker imagery. 
**Here is the command to run the docker imagery**

<code>sudo nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888  spatialcomputing/map_text_recognition_gpu </code>

### 2) Run TGG
### Step 1: Data Generation
**Firstly, cropping images within the region level annotation**

<code>python3 crop_images.py --data_dir {path-to-dataset} --loc_idx {name-or-index-of-image} --mask_name {region-annotation} --obj_name {target-object-name} --stride {sliding-window-stride} --win_size {sliding-window-size} </code>
  
**For example,** <code>python3 crop_images.py --data_dir '/data/weiweidu/COWC/DetectionPatches_800x800/Toronto_ISPRS' --loc_idx 'Toronto_03559.8.2' --mask_name None --obj_name='car' --stride=20 --win_size 50 </code>

**Secondly, choosing target images as labeled data from the cropped images**

<code>python3 gen_train_data.py --dataset_name {dataset-name} --loc_idx {name-or-index-of-image} --obj_name {target-object-name} --pos_names {list-of-target-images-names} --sample_portion {percentage-of-cropped-images} </code>

**For example,** <code>python3 gen_train_data.py --dataset_name 'COWC' --loc_idx 'Toronto_03559.8.2' --obj_name='car' --pos_names ['100_100'] --sample_portion 1.0</code>

### Step 2: Iteratively Training TGG
**TGG takes cropped images and labeled target image(s) as inputs.<br/>
The recognition results are top-left coordinates of cropped images, saved in a txt file.**

<code>python3 train.py --dataset_name {dataset-name} --loc_idx {name-or-index-of-image} --obj_name {target-object-name} --map_path {path_to_map} --augmentation {True/False} --image_size {sliding-window-size} --stride {sliding-window-stride} --num_epochs {number-of-epochs} --learning_rate {learning-rate} --batch_size {batch-size} --weight {weight-for-multiloss} --saved_model_path {path-to-save-model} --save_detected_images </code>
  
**For example,** <code>python3 train.py --dataset_name 'COWC' --loc_idx 'Toronto_03559.8.2' --obj_name='car' --map_path='./data/COWC/Toronto_03553.7.8/Toronto_03553.7.8.jpg' --augmentation True --image_size 50 --stride 20 --num-epochs 500 --learning_rate 0.0001 --batch_size 200 --weight 500 --saved_model_path 'TGG.hd5f' </code>

### Step 3: Evaluation
**The evaluation script takes bounding-box level ground truth and image-level recognition results as inputs.<br/> 
The results are evaluated by precision, recall, and F1 score in the grid level.**

<code>python3 cal_eval.py --annotation_dir {path-to-ground-truth-folder} --pred_dir {path-to-TGG-ressult-folder} --obj_name {target-object-name} --loc_idx 'Toronto_03559.8.2' --grid_size {grid-size-for-eval} --image-size {crop-image-size}</code>
