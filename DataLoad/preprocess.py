from ctypes.wintypes import POINTL
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import sys

def checkCoordinates(data_dir):
    # This function checks how many points of the polygon bounding box are provided. 
    diagonal = 0
    four = 0
    unknown = 0
    dlabels = []
    flabels = []
    ulabels = []

    for file_name in tqdm(os.listdir(data_dir)):

        # get the .tif files
        if '.tif' in file_name:
            filename=file_name.replace('.tif', '')
            print('Working on map:', file_name)
            file_path=os.path.join(data_dir, file_name)
            test_json=file_path.replace('.tif', '.json')
            
            # read the legend annotation file
            with open(test_json) as f:
                data = json.load(f)

            # load image into an array
            im=cv2.imread(file_path)
            im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
            #plt.imshow(im)
            #plt.show()
            
           
            for shape in data['shapes']:
                # read labels and bounding box coordinates
                label = shape['label']
                points = shape['points']

                if(len(points)==2):
                    diagonal+=1
                    dlabels.append(label)
                elif(len(points)==4):
                    four+=1
                    flabels.append(label)
                else: 
                    unknown+=1
                    ulabels.append(label)

    textfile = open("labelStructure.txt", "w")
    textfile.write(str(diagonal) + " : " + str(dlabels) + "\n\n")
    textfile.write(str(four) + " : " + str(flabels) + "\n\n")
    textfile.write(str(unknown) + " : " + str(ulabels) + "\n\n")
    textfile.close()

def process(data_dir):
    # create directory to store results
    os.makedirs('ValidationLabels', exist_ok=True)

    for file_name in tqdm(os.listdir(data_dir)):

        # get the .tif files
        if '.tif' in file_name:
            filename=file_name.replace('.tif', '')
            print('Working on map:', file_name)
            file_path=os.path.join(data_dir, file_name)
            test_json=file_path.replace('.tif', '.json')
            
            # read the legend annotation file
            with open(test_json) as f:
                data = json.load(f)

            # load image into an array
            im=cv2.imread(file_path)
            im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
           
            for shape in data['shapes']:
                # read labels and bounding box coordinates
                label = shape['label']
                points = shape['points']
                
                # extract the minimum and maximum points.
                x_min = y_min = sys.float_info.max
                x_max = y_max =0
                for each in points:
                    x_t,y_t= each
                    if(x_min > x_t):
                        x_min = x_t
                        
                    if(y_min > y_t):
                        y_min = y_t
                
                    if(x_max < x_t):
                        x_max = x_t
                        
                    if(y_max < y_t):
                        y_max = y_t
                    
                template = im[int(y_min):int(y_max), int(x_min):int(x_max)]
                h, w = template.shape[0], template.shape[1]
                print('using the following legend feature for matching...:')

                # save the raster into a .tif file
                out_file_path=os.path.join('ValidationLabels', filename+'_label_'+label+'.jpeg')
                label_image=template.astype('uint16')
                cv2.imwrite(out_file_path, label_image)

def main():
    parser = argparse.ArgumentParser(description='map_feature_extraction')
    # load data from file
    parser.add_argument('--data_path', type=str, default='ValidationImages', help='directory should contain JSON and Map Image .tif file')
    args = parser.parse_args()
    process(args.data_path)

if __name__ == "__main__":
    main()
